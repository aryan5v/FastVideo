# SPDX-License-Identifier: Apache-2.0
"""Model-independent FX region capture for optimization discovery.

The capture hook attaches to *repeated* submodule stacks (any ``nn.ModuleList``
whose children share one class) so it works for any transformer-style model
without naming a single architecture. Per-call bookkeeping inside the forward
hook is deliberately cheap — shape/dtype signatures and counters only. The
expensive FX trace runs once per distinct scope in :meth:`FXCaptureSession.finalize`,
after the profiled region has closed, so captured graphs never contaminate the
timings the profiler is collecting.

Only metadata is exported: operation names, dependency edges, tensor layout
signatures, safe scalar constants, call/shape frequencies, and capture
failures. Tensor values, weights, activations and prompts are never read or
serialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn

# Bumped independently of the profiler export's ``schema_version`` so consumers
# can tell a capture-format change from a profiler-row change.
CAPTURE_SCHEMA_VERSION = 2

# Region names and op keys must satisfy the consumer's validation patterns.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_NAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")

# Keys that must never appear anywhere in an exported payload.
FORBIDDEN_KEYS = frozenset({
    "activations",
    "credential",
    "credentials",
    "data",
    "password",
    "prompt",
    "prompts",
    "secret",
    "secrets",
    "source",
    "source_code",
    "tensor_values",
    "token",
    "values",
    "weights",
})

_SAFE_SCALAR_TYPES = (bool, int, float, str)
_CAPTURE_MODES = ("symbolic", "export", "dynamo")


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, _SAFE_SCALAR_TYPES):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    raise ValueError(f"unsupported fingerprint value type: {type(value).__name__}")


def graph_fingerprint(
    *,
    operations: list[str],
    input_signatures: list[dict[str, Any]],
    output_signatures: list[dict[str, Any]] | None = None,
    safe_constants: dict[str, Any] | None = None,
    parent_module: str | None = None,
) -> str:
    """Stable content hash of a captured region.

    Mirrors the canonical discovery fingerprint: derived only from op names,
    tensor signatures and safe constants, so two runs of the same region agree
    and the consumer can recompute it as an integrity check.
    """
    payload = {
        "operations": list(operations),
        "inputs": list(input_signatures),
        "outputs": list(output_signatures or ()),
        "constants": dict(safe_constants or {}),
        "parent_module": parent_module or "",
    }
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _tensor_meta(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Layout metadata for one tensor. Never touches tensor storage."""
    shape = [int(dim) for dim in tensor.shape]
    try:
        stride = [int(step) for step in tensor.stride()]
    except (RuntimeError, NotImplementedError):
        # Sparse / nested layouts have no stride; synthesize a contiguous one.
        stride = []
        running = 1
        for dim in reversed(shape):
            stride.append(running)
            running *= max(dim, 1)
        stride.reverse()
    return {
        "name": name,
        "shape": shape,
        "stride": stride,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device_type": tensor.device.type,
        "requires_grad": bool(tensor.requires_grad),
    }


def _signature_dict(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "name"}


def _shape_key(metas: list[dict[str, Any]]) -> str:
    return "|".join("{}:{}:{}".format(meta["name"], "x".join(str(dim) for dim in meta["shape"]), meta["dtype"])
                    for meta in metas)


def _safe_name(value: str) -> str:
    cleaned = _NAME_SANITIZE.sub("_", value).strip("._-")
    return (cleaned or "tensor")[:96]


def _tensor_leaves(
    value: Any,
    *,
    prefix: str,
    seen: set[int] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Find tensor leaves in supported containers without reading values."""
    if isinstance(value, torch.Tensor):
        return [(_safe_name(prefix), value)]
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    leaves: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(
                item,
                prefix=f"{prefix}_{index}",
                seen=seen,
            ))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            leaves.extend(_tensor_leaves(
                item,
                prefix=f"{prefix}_{_safe_name(str(key))}",
                seen=seen,
            ))
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            leaves.extend(
                _tensor_leaves(
                    getattr(value, item.name),
                    prefix=f"{prefix}_{_safe_name(item.name)}",
                    seen=seen,
                ))
    return leaves


def _input_metas(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    leaves: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for index, value in enumerate(args):
        leaves.extend(_tensor_leaves(value, prefix=f"input_{index}", seen=seen))
    for key, value in kwargs.items():
        leaves.extend(_tensor_leaves(
            value,
            prefix=f"kwarg_{_safe_name(str(key))}",
            seen=seen,
        ))
    return [_tensor_meta(name, tensor) for name, tensor in leaves]


def _output_metas(output: Any) -> list[dict[str, Any]]:
    return [_tensor_meta(name, tensor) for name, tensor in _tensor_leaves(output, prefix="output")]


def _input_shape_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build the same shape key as capture without retaining tensor metadata."""
    return _shape_key(_input_metas(args, kwargs))


def _region_name(scope: str, shape_key: str) -> str:
    digest = hashlib.sha256(shape_key.encode("utf-8")).hexdigest()[:8]
    base = _NAME_SANITIZE.sub("_", scope).strip("._-") or "region"
    name = f"{base}.{digest}"[:128]
    if not _NAME_PATTERN.fullmatch(name):
        name = f"region.{digest}"
    return name


def _is_safe_constant(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        # Short enum-like tags only — never free-form text.
        return bool(re.fullmatch(r"[A-Za-z0-9_.:/-]{1,64}", value))
    if isinstance(value, list | tuple):
        return len(value) <= 32 and all(_is_safe_constant(item) for item in value)
    return False


def _sanitize_constants(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and key and key.lower() not in FORBIDDEN_KEYS and _is_safe_constant(value)
    }


class DataclassRegistrationError(RuntimeError):
    """Input/output dataclass types could not be prepared for export."""


# Bounds for the pre-export scan. A malformed or adversarial input structure
# must not turn discovery into an unbounded walk.
_MAX_DATACLASS_SCAN_DEPTH = 16
_MAX_DATACLASS_SCAN_NODES = 4096

# Backstop dedup cache for the (unlikely) case where pytree's registry cannot
# be consulted. Registration is already a permanent global, so holding the
# class here adds no lifetime that ``register_dataclass`` did not already take.
_REGISTERED_DATACLASSES: set[type] = set()


def _is_pytree_registered(cls: type) -> bool:
    """True when ``cls`` is already a pytree node, however it was registered.

    Consulting the live registry (rather than only our own cache) keeps us from
    overwriting a registration made by the model, ``diffusers``, or an earlier
    process-wide call — an overwrite both warns and replaces the other party's
    serialized type name.
    """
    try:
        from torch.utils import _pytree

        return cls in _pytree.SUPPORTED_NODES
    except Exception:  # noqa: BLE001 — private registry is best-effort only
        return cls in _REGISTERED_DATACLASSES


def _iter_dataclass_types(
    value: Any,
    *,
    seen: set[int],
    depth: int,
    budget: list[int],
) -> Iterator[type]:
    """Yield dataclass types reachable from ``value``.

    Only ``type(...)`` is read. Field values are traversed for *structure*
    alone: they are never compared, formatted, hashed, or exported, and tensor
    storage is never touched.
    """
    if depth > _MAX_DATACLASS_SCAN_DEPTH:
        raise DataclassRegistrationError("dataclass_scan_depth_exceeded")
    budget[0] -= 1
    if budget[0] < 0:
        raise DataclassRegistrationError("dataclass_scan_budget_exceeded")

    # Leaves: tensors and safe scalars can never contain a dataclass. Skipping
    # them before the identity check also avoids interned-scalar id collisions.
    if value is None or isinstance(value, (torch.Tensor, *_SAFE_SCALAR_TYPES)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if isinstance(value, tuple | list):
        for item in value:
            yield from _iter_dataclass_types(item, seen=seen, depth=depth + 1, budget=budget)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_dataclass_types(item, seen=seen, depth=depth + 1, budget=budget)
    elif is_dataclass(value) and not isinstance(value, type):
        yield type(value)
        for item in fields(value):
            try:
                child = getattr(value, item.name)
            except AttributeError:
                # An uninitialized ``field(init=False)`` carries no type to
                # register; a later export failure is recorded normally.
                continue
            yield from _iter_dataclass_types(child, seen=seen, depth=depth + 1, budget=budget)


def dataclass_types_in(value: Any) -> tuple[type, ...]:
    """Ordered, de-duplicated dataclass types reachable from ``value``."""
    seen: set[int] = set()
    budget = [_MAX_DATACLASS_SCAN_NODES]
    found = _iter_dataclass_types(value, seen=seen, depth=0, budget=budget)
    return tuple(dict.fromkeys(found))


def _register_dataclasses(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        extra_types: tuple[type, ...] = (),
) -> tuple[str, ...]:
    """Register every dataclass type the traced call needs, once each.

    ``torch.export`` flattens inputs and outputs through pytree, so a module
    whose signature carries a dataclass is rejected unless that type is a
    registered node. This is the model-independent fix: the types come from the
    observed call itself, so no architecture, field name, or class is named
    here. Returns the sanitized class names that were newly registered.
    """
    try:
        discovered: list[type] = []
        seen: set[int] = set()
        budget = [_MAX_DATACLASS_SCAN_NODES]
        for value in (*args, *kwargs.values()):
            discovered.extend(_iter_dataclass_types(value, seen=seen, depth=0, budget=budget))
        discovered.extend(extra_types)
    except DataclassRegistrationError:
        raise
    except Exception as exc:  # noqa: BLE001 — traversal must fail closed
        raise DataclassRegistrationError("dataclass_scan_failed") from exc

    registered: list[str] = []
    for cls in dict.fromkeys(discovered):
        if _is_pytree_registered(cls):
            continue
        try:
            torch.export.register_dataclass(cls)
        except Exception as exc:  # noqa: BLE001 — registration must fail closed
            raise DataclassRegistrationError("dataclass_register_failed") from exc
        _REGISTERED_DATACLASSES.add(cls)
        registered.append(_safe_name(cls.__name__))
    return tuple(registered)


def _capture_failure_reason(mode: str, exc: Exception) -> str:
    """Classify failures without exporting exception text or source snippets."""
    text = str(exc).lower()
    if isinstance(exc, DataclassRegistrationError):
        code = "dataclass_registration"
    elif any(marker in text for marker in (
            "data-dependent",
            "data dependent",
            "guardondatadependentsymnode",
            ".item()",
    )):
        code = "data_dependent_control_flow"
    elif any(marker in text for marker in (
            "control flow",
            "proxy object",
            "symbolically traced variables",
            "cannot be iterated",
    )):
        code = "dynamic_python_control_flow"
    elif "alias" in text:
        code = "unknown_aliasing"
    elif "unsupported" in text or "not supported" in text:
        code = "unsupported_graph"
    else:
        code = "trace_error"
    return f"capture_failed:{mode}:{code}:{type(exc).__name__}"


def assert_metadata_only(payload: Any, path: str = "capture") -> None:
    """Fail loudly if a payload carries a forbidden key or a live tensor."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise RuntimeError(f"forbidden key {key!r} at {path}")
            assert_metadata_only(value, f"{path}.{key}")
    elif isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            assert_metadata_only(item, f"{path}[{index}]")
    elif isinstance(payload, torch.Tensor):
        raise RuntimeError(f"tensor value at {path}")
    elif payload is not None and not isinstance(payload, _SAFE_SCALAR_TYPES):
        raise RuntimeError(f"non-JSON value of type {type(payload).__name__} at {path}")


@contextmanager
def _capture_ready_module(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Temporarily bypass FastVideo's eager-only layer-offload wrapper.

    ``ModuleHookManager`` replaces ``forward`` with ``functools.partial``.
    Non-strict export requires a Python function with ``__code__``, while
    Dynamo intentionally refuses to inline the offload hook. Run that
    state loader eagerly, expose the original forward only for graph
    capture, then restore the exact parameter/offload state and wrapper.
    Calling the hook itself is not observational: its pre-hook prefetches
    the next layer and can leave that layer populated after a failed trace.
    Unknown hooks fail closed because bypassing them could change model
    semantics.
    """
    manager = getattr(module, "_hook_manager", None)
    if manager is None:
        yield args, kwargs
        return

    forward_hooks = getattr(manager, "forward_hooks", {})
    unsupported_hooks = [str(name) for name in forward_hooks if str(name) != "LayerwiseOffloadHook"]
    if unsupported_hooks:
        raise RuntimeError("unsupported_module_hooks")

    wrapped_forward = module.forward
    original_forward = getattr(manager, "original_forward", None)
    if original_forward is None:
        raise RuntimeError("missing_original_forward")

    offload_hook = forward_hooks.get("LayerwiseOffloadHook")
    state = getattr(offload_hook, "state", None)
    if offload_hook is not None and state is None:
        raise RuntimeError("missing_offload_state")

    parameter_data = {name: parameter.data for name, parameter in module.named_parameters()}
    gpu_parameters = (dict(getattr(state, "gpu_named_parameters", {})) if state is not None else {})
    try:
        if state is not None:
            state.wait_and_replace_params()
        module.forward = original_forward
        yield tuple(args), dict(kwargs)
    finally:
        module.forward = wrapped_forward
        if state is not None:
            named_parameters = dict(module.named_parameters())
            for name, data in parameter_data.items():
                parameter = named_parameters.get(name)
                if parameter is not None:
                    parameter.data = data
            state.gpu_named_parameters.clear()
            state.gpu_named_parameters.update(gpu_parameters)


@contextmanager
def _capture_forward_context(observed_context: tuple[Any, Any] | None, ) -> Iterator[None]:
    """Re-establish the bounded runtime context needed by attention layers."""
    if observed_context is None:
        yield
        return

    from fastvideo.forward_context import set_forward_context

    current_timestep, attention_metadata = observed_context
    with set_forward_context(
            current_timestep=current_timestep,
            attn_metadata=attention_metadata,
            forward_batch=None,
    ):
        yield


@contextmanager
def _traceable_module_forwards(module: nn.Module) -> Iterator[None]:
    """Temporarily expose forwards hidden behind ``torch.compiler.disable``.

    FastVideo deliberately keeps attention eager during normal compiled
    execution. Discovery is different: export/Dynamo must see the attention
    call to describe the dominant block. PyTorch's disable wrapper retains the
    original callable, so expose it only during capture and restore every
    module instance afterward.
    """
    original_forwards: list[tuple[nn.Module, Any]] = []
    try:
        for child in module.modules():
            forward = child.forward
            function = getattr(forward, "__func__", forward)
            if not getattr(function, "_torchdynamo_disable", False):
                continue
            original = getattr(
                function,
                "_torchdynamo_orig_callable",
                None,
            )
            if original is None:
                raise RuntimeError("missing_compiler_disabled_forward")
            original_forwards.append((child, forward))
            child.forward = MethodType(original, child)
        yield
    finally:
        for child, forward in reversed(original_forwards):
            child.forward = forward


# FX lowers Python operators (``a * b``) to ``operator.mul`` and friends. They
# are the same aten ops as the method form, so normalize them together.
_OPERATOR_ALIASES = {
    "add": "add",
    "iadd": "add",
    "sub": "sub",
    "isub": "sub",
    "mul": "mul",
    "imul": "mul",
    "truediv": "div",
    "itruediv": "div",
    "matmul": "matmul",
    "neg": "neg",
    "pow": "pow",
    "getitem": "select",
}


def _op_key(target: Any) -> str:
    """Normalize an FX call target to an ``aten::``-style op key."""
    text = str(target)
    if "aten::" in text:
        return "aten::" + text.split("aten::", 1)[1].split(".")[0].split("(")[0]
    if "aten." in text:
        return "aten::" + text.split("aten.", 1)[1].split(".")[0]
    name = getattr(target, "__name__", None) or text
    module = getattr(target, "__module__", "") or ""
    if module in {"_operator", "operator"} and name in _OPERATOR_ALIASES:
        return f"aten::{_OPERATOR_ALIASES[name]}"
    if module.startswith("torch"):
        return f"aten::{name}"
    return _NAME_SANITIZE.sub("_", str(name))[:256] or "unknown"


def _extract_graph(graph: Any) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    """Return (operations, dependencies, safe_constants, structural_notes)."""
    operations: list[str] = []
    dependencies: list[str] = []
    constants: dict[str, Any] = {}
    notes: list[str] = []
    index_of: dict[str, int] = {}

    def node_names(value: Any) -> Iterator[str]:
        name = getattr(value, "name", None)
        if name is not None and hasattr(value, "op"):
            yield name
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from node_names(item)
        elif isinstance(value, tuple | list):
            for item in value:
                yield from node_names(item)

    def record_deps(node: Any, index: int) -> None:
        seen: set[str] = set()
        for arg_name in node_names((node.args, node.kwargs or {})):
            if arg_name in index_of and arg_name not in seen:
                dependencies.append(f"{index_of[arg_name]}->{index}")
                seen.add(arg_name)

    def record_constants(node: Any, key: str) -> None:
        for arg_index, value in enumerate(node.args):
            if hasattr(value, "op"):
                continue
            if _is_safe_constant(value):
                constants[f"{node.name}.arg{arg_index}"] = value
            elif value is not None and not tuple(node_names(value)):
                notes.append(f"{key}: unsafe positional constant arg{arg_index}")
        for kwarg, value in (node.kwargs or {}).items():
            if hasattr(value, "op"):
                continue
            if _is_safe_constant(value):
                constants[f"{node.name}.{kwarg}"] = value
            elif value is not None and not tuple(node_names(value)):
                notes.append(f"{key}: non-scalar constant {kwarg!r} dropped")

    for node in graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            target = str(node.target)
            notes.append(f"get_attr:{target}: lifted attribute value not exported")
            continue
        if node.op == "call_function":
            key = _op_key(node.target)
        elif node.op == "call_method":
            key = f"aten::{node.target}"
        elif node.op == "call_module":
            key = f"module::{node.target}"
            notes.append(f"{key}: nested module not expanded")
        else:
            notes.append(f"unknown_fx_op:{node.op}")
            continue
        index = len(operations)
        index_of[node.name] = index
        operations.append(key)
        record_deps(node, index)
        record_constants(node, key)
        if "as_strided" in key or key.endswith("::alias"):
            notes.append(f"capture_safety:unknown_aliasing:{key}")

    return operations, dependencies, _sanitize_constants(constants), notes


def _ir_tensor_meta(value: Any) -> dict[str, Any] | None:
    """Return value metadata accepted by MotionKernel's executable IR."""
    if not isinstance(value, torch.Tensor):
        return None
    try:
        shape = [int(dim) for dim in value.shape]
    except (TypeError, ValueError, RuntimeError):
        return None
    return {
        "shape": shape,
        "dtype": str(value.dtype).replace("torch.", ""),
        "requires_grad": bool(value.requires_grad),
    }


def _ir_target(node: Any) -> str:
    if node.op == "call_method":
        return f"aten.{node.target}.default"
    if node.op == "call_module":
        return f"module::{str(node.target)[:128]}"
    target = node.target
    name = getattr(target, "__name__", None)
    module = getattr(target, "__module__", "") or ""
    if module in {"_operator", "operator"} and name == "getitem":
        return "operator.getitem"
    text = str(target)
    if text.startswith("aten."):
        return text
    # The type category is enough for a fail-closed consumer; never serialize
    # repr/source text for unknown callables.
    return f"unsupported::{_safe_name(type(target).__name__)}"


def _ir_argument(
    value: Any,
    node_expressions: Mapping[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Encode graph wiring and safe metadata constants, never tensor values."""
    try:
        if value in node_expressions:
            return dict(node_expressions[value])
    except (TypeError, RuntimeError):
        pass
    if value is None or isinstance(value, bool | int):
        return {"const": value}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"const": value}
        return {"unsupported": "non_finite_float"}
    if isinstance(value, str):
        if re.fullmatch(r"[A-Za-z0-9_.:/-]{1,64}", value):
            return {"const": value}
        return {"unsupported": "string"}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).replace("torch.", "")}
    if isinstance(value, torch.device):
        return {"device": "runtime"}
    if isinstance(value, tuple | list):
        kind = "tuple" if isinstance(value, tuple) else "list"
        return {kind: [_ir_argument(item, node_expressions) for item in value]}
    # Static SymInts are shape metadata and safe to materialize.
    if type(value).__module__.startswith("torch") and type(value).__name__ in {
            "SymInt",
            "SymBool",
            "SymFloat",
    }:
        try:
            scalar = int(value) if type(value).__name__ != "SymFloat" else float(value)
            return {"const": scalar}
        except (TypeError, ValueError, RuntimeError):
            return {"unsupported": type(value).__name__}
    return {"unsupported": _safe_name(type(value).__name__)}


def _extract_executable_ir(exported: Any, graph: Any) -> dict[str, Any]:
    """Build an operand-aware, metadata-only IR from an ExportedProgram.

    Export is required because its graph signature distinguishes runtime
    tensors from lifted parameters. Lifted inputs receive generic names; model
    parameter paths and values are deliberately excluded.
    """
    graph_signature = getattr(exported, "graph_signature", None)
    if graph_signature is None:
        raise RuntimeError("missing_graph_signature")
    input_specs = list(getattr(graph_signature, "input_specs", ()))
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(input_specs) != len(placeholders):
        raise RuntimeError("graph_signature_input_mismatch")

    node_expressions: dict[Any, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    runtime_index = 0
    lifted_index = 0
    for index, (node, input_spec) in enumerate(zip(placeholders, input_specs, strict=True)):
        placeholder_value = (node.meta or {}).get("val")
        meta = _ir_tensor_meta(placeholder_value)
        if meta is None:
            # A specialized non-tensor graph input. Strings arriving as *inputs*
            # are caller data — model ids, paths, captions, dataclass tag fields
            # — not op enums, so they never enter the IR even when they match
            # the enum-shaped safe-string pattern. Registering dataclasses makes
            # their string fields reachable here, so this fails closed.
            if isinstance(placeholder_value, str):
                raise RuntimeError("unsupported_non_tensor_graph_input")
            constant = _ir_argument(placeholder_value, {})
            if set(constant) == {"unsupported"}:
                raise RuntimeError("unsupported_non_tensor_graph_input")
            node_expressions[node] = constant
            continue
        kind_name = getattr(getattr(input_spec, "kind", None), "name", "")
        if kind_name == "USER_INPUT":
            name = f"input_{runtime_index}"
            kind = "runtime"
            runtime_index += 1
        else:
            name = f"lifted_{lifted_index}"
            kind = "lifted"
            lifted_index += 1
        node_id = f"p{index}"
        node_expressions[node] = {"ref": node_id}
        inputs.append({"id": node_id, "name": name, "kind": kind, "meta": meta})

    nodes: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            raise RuntimeError("unlifted_graph_attribute")
        if node.op not in {"call_function", "call_method", "call_module"}:
            raise RuntimeError("unsupported_graph_node_kind")
        node_id = f"n{len(nodes)}"
        node_expressions[node] = {"ref": node_id}
        item: dict[str, Any] = {
            "id": node_id,
            "target": _ir_target(node),
            "args": [_ir_argument(value, node_expressions) for value in node.args],
            "kwargs": {
                _safe_name(str(key)): _ir_argument(value, node_expressions)
                for key, value in (node.kwargs or {}).items()
            },
        }
        meta = _ir_tensor_meta((node.meta or {}).get("val"))
        if meta is not None:
            item["meta"] = meta
        nodes.append(item)

    output_nodes = [node for node in graph.nodes if node.op == "output"]
    if len(output_nodes) != 1 or not output_nodes[0].args:
        raise RuntimeError("invalid_graph_output")
    encoded_output = _ir_argument(output_nodes[0].args[0], node_expressions)
    outputs = (encoded_output["tuple"] if set(encoded_output) == {"tuple"} else [encoded_output])
    return {
        "schema_version": 1,
        "inputs": inputs,
        "nodes": nodes,
        "outputs": outputs,
    }


@dataclass
class _ShapeVariant:
    """One observed input/output signature for a scope."""

    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    # Bounded, in-memory references used only after the profiler window closes.
    # They are cleared before the JSON payload is returned and never serialized.
    example_args: tuple[Any, ...] | None = field(default=None, repr=False)
    example_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    observed_context: tuple[Any, Any] | None = field(
        default=None,
        repr=False,
    )
    # Export rejects an unregistered dataclass in the *output* as well as the
    # input, so the observed return structure contributes its types. Only the
    # types are kept — never the returned objects or their tensors.
    output_dataclass_types: tuple[type, ...] = field(default=(), repr=False)
    calls: int = 0


@dataclass
class _Scope:
    """A repeated module stack being captured, e.g. ``transformer.blocks``."""

    scope: str
    class_name: str
    module: nn.Module
    variants: dict[str, _ShapeVariant] = field(default_factory=dict)
    calls: int = 0
    dropped_variants: int = 0


def default_capture_targets(root: nn.Module) -> list[tuple[str, nn.Module]]:
    """Select repeated block stacks to hook, with no per-architecture knowledge.

    A stack of identically-typed children under an ``nn.ModuleList`` is the
    model-independent signature of a transformer/DiT block list. Every member
    is hooked (so call and shape frequencies reflect the whole stack), but they
    share one scope name, so exactly one FX trace is taken per stack.
    """
    targets: list[tuple[str, nn.Module]] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        classes = {type(child) for child in module}
        if len(classes) != 1:
            continue
        scope = name or "root"
        targets.extend((scope, child) for child in module)
    return targets


class FXCaptureSession:
    """Hook repeated module stacks, then trace them after the timed region.

    Forward hooks only record signatures and counters. ``finalize`` performs
    the FX traces and returns a JSON-ready, metadata-only payload. Any failure
    in either phase is recorded as data rather than raised at the caller.
    """

    def __init__(
        self,
        *,
        tracer: str = "auto",
        max_scopes: int = 64,
        max_shape_variants: int = 8,
    ) -> None:
        self.tracer = tracer
        self.max_scopes = max_scopes
        self.max_shape_variants = max_shape_variants
        self._scopes: dict[str, _Scope] = {}
        self._handles: list[Any] = []
        self._graph_breaks: list[dict[str, Any]] = []
        self._unsupported: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._dropped_scopes = 0

    # -- attach ---------------------------------------------------------

    def attach(self, root: nn.Module, *, prefix: str = "") -> int:
        """Hook every repeated block stack under ``root``. Returns hook count."""
        try:
            targets = default_capture_targets(root)
        except Exception as exc:  # noqa: BLE001 — capture never breaks generation
            self._record_exception("target_selection_failed", exc)
            return 0
        hooked = 0
        for scope, module in targets:
            qualified = f"{prefix}.{scope}" if prefix else scope
            if qualified not in self._scopes and len(self._scopes) >= self.max_scopes:
                self._dropped_scopes += 1
                continue
            try:
                self._hook(qualified, module)
                hooked += 1
            except Exception as exc:  # noqa: BLE001
                self._record_exception("hook_failed", exc, scope=qualified)
        return hooked

    def attach_modules(self, modules: dict[str, Any] | None) -> int:
        """Hook every ``nn.Module`` in a pipeline's module mapping."""
        if not modules:
            return 0
        hooked = 0
        for name, module in modules.items():
            if isinstance(module, nn.Module):
                hooked += self.attach(module, prefix=str(name))
        return hooked

    def _hook(self, scope: str, module: nn.Module) -> None:
        record = self._scopes.get(scope)
        if record is None:
            # The first member of the stack is the trace representative.
            record = _Scope(scope=scope, class_name=type(module).__name__, module=module)
            self._scopes[scope] = record

        active_ranges: list[Any] = []

        def pre_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Add a shape-specific range that torch.profiler can attribute."""
            try:
                name = _region_name(scope, _input_shape_key(args, kwargs))
                profiler_range = torch.profiler.record_function(f"motionkernel::{name}")
                profiler_range.__enter__()
                active_ranges.append(profiler_range)
            except Exception as exc:  # noqa: BLE001 — never disturb the forward
                self._record_exception("profile_range_failed", exc, scope=scope)

        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            try:
                self._observe(record, args, kwargs, output)
            except Exception as exc:  # noqa: BLE001 — never disturb the forward
                self._record_exception("observe_failed", exc, scope=scope)
            finally:
                if active_ranges:
                    try:
                        active_ranges.pop().__exit__(None, None, None)
                    except Exception as exc:  # noqa: BLE001
                        self._record_exception(
                            "profile_range_close_failed",
                            exc,
                            scope=scope,
                        )

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True, always_call=True))

    def _observe(self, record: _Scope, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        record.calls += 1
        inputs = _input_metas(args, kwargs)
        if not inputs:
            self._graph_breaks.append({"scope": record.scope, "reason": "no_tensor_inputs", "count": 1})
            return

        outputs = _output_metas(output)

        key = _shape_key(inputs)
        variant = record.variants.get(key)
        if variant is None:
            if len(record.variants) >= self.max_shape_variants:
                record.dropped_variants += 1
                return
            try:
                from fastvideo.forward_context import get_forward_context

                forward_context = get_forward_context()
                observed_context = (
                    forward_context.current_timestep,
                    forward_context.attn_metadata,
                )
            except AssertionError:
                observed_context = None
            try:
                output_dataclass_types = dataclass_types_in(output)
            except DataclassRegistrationError:
                # A pathological return structure must not break the forward;
                # export will fail closed and be recorded during finalize.
                output_dataclass_types = ()
            variant = _ShapeVariant(
                inputs=inputs,
                outputs=outputs,
                example_args=args,
                example_kwargs=dict(kwargs),
                observed_context=observed_context,
                output_dataclass_types=output_dataclass_types,
            )
            record.variants[key] = variant
        variant.calls += 1

    # -- finalize -------------------------------------------------------

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                self._record_exception("detach_failed", exc)
        self._handles.clear()

    def _record_error(self, message: str) -> None:
        if len(self._errors) < 64:
            self._errors.append(message)

    def _record_exception(
        self,
        code: str,
        exc: Exception,
        *,
        scope: str | None = None,
    ) -> None:
        location = f"[{scope}]" if scope else ""
        self._record_error(f"{code}{location}:{type(exc).__name__}")

    def _mode_order(self) -> tuple[str, ...]:
        if self.tracer in {"auto", "fallback"}:
            return _CAPTURE_MODES
        if self.tracer in _CAPTURE_MODES:
            return (self.tracer, )
        raise ValueError(f"unsupported tracer {self.tracer!r}; "
                         "use auto, symbolic, export, or dynamo")

    def _trace(
        self,
        module: nn.Module,
        variant: _ShapeVariant,
        mode: str,
    ) -> Any:
        args = variant.example_args
        kwargs = variant.example_kwargs
        if args is None or kwargs is None:
            raise RuntimeError("example arguments unavailable")
        with _capture_ready_module(module, args, kwargs) as (
                capture_args,
                capture_kwargs,
        ), _capture_forward_context(variant.observed_context), _traceable_module_forwards(module):
            if mode == "symbolic":
                # Symbolic tracing builds proxies from the forward signature and
                # never pytree-flattens the example inputs, so it needs no
                # dataclass registration.
                return torch.fx.symbolic_trace(module)
            if mode == "export":
                _register_dataclasses(
                    capture_args,
                    capture_kwargs,
                    variant.output_dataclass_types,
                )
                return torch.export.export(
                    module,
                    capture_args,
                    capture_kwargs,
                    strict=False,
                )
            if mode == "dynamo":
                _register_dataclasses(
                    capture_args,
                    capture_kwargs,
                    variant.output_dataclass_types,
                )
                exported = torch._dynamo.export(module, aten_graph=True)(
                    *capture_args,
                    **capture_kwargs,
                )
                graph_module = getattr(exported, "graph_module", None)
                if graph_module is None and isinstance(exported, tuple):
                    graph_module = exported[0]
                return (graph_module if graph_module is not None else exported)
        raise ValueError(f"unsupported capture mode {mode!r}")

    def _regions_for(self, record: _Scope) -> list[dict[str, Any]]:
        regions: list[dict[str, Any]] = []
        for shape_key, variant in record.variants.items():
            capture_mode: str | None = None
            failures: list[str] = []
            operations: list[str] = []
            dependencies: list[str] = []
            constants: dict[str, Any] = {}
            notes: list[str] = []
            executable_ir: dict[str, Any] | None = None
            attempts = self._mode_order()
            try:
                for mode in attempts:
                    try:
                        traced = self._trace(record.module, variant, mode)
                        graph = getattr(traced, "graph", None)
                        if graph is None:
                            raise RuntimeError("trace result has no FX graph")
                        operations, dependencies, constants, notes = _extract_graph(graph)
                        if not operations:
                            raise RuntimeError("trace result has no tensor operations")
                        capture_mode = mode
                        if mode == "export":
                            try:
                                executable_ir = _extract_executable_ir(
                                    traced,
                                    graph,
                                )
                            except Exception as exc:  # noqa: BLE001
                                known_reason = str(exc)
                                if known_reason not in {
                                        "graph_signature_input_mismatch",
                                        "invalid_graph_output",
                                        "missing_graph_signature",
                                        "unlifted_graph_attribute",
                                        "unsupported_graph_node_kind",
                                        "unsupported_non_tensor_graph_input",
                                }:
                                    known_reason = type(exc).__name__
                                self._graph_breaks.append({
                                    "scope": record.scope,
                                    "reason": ("executable_ir_unavailable:"
                                               f"{known_reason}"),
                                    "count": max(variant.calls, 1),
                                })
                        break
                    except Exception as exc:  # noqa: BLE001
                        reason = _capture_failure_reason(mode, exc)
                        failures.append(reason)
                        self._graph_breaks.append({
                            "scope": record.scope,
                            "reason": reason,
                            "count": max(variant.calls, 1),
                        })
            finally:
                # Drop all live tensor/object references before serialization.
                variant.example_args = None
                variant.example_kwargs = None
                variant.observed_context = None

            if capture_mode is None:
                continue

            for note in notes:
                self._graph_breaks.append({
                    "scope": record.scope,
                    "reason": note[:512],
                    "count": 1,
                })
                if "nested module" in note or "unknown_fx_op" in note:
                    self._unsupported.append({
                        "op_name": note.split(":", 1)[0][:256],
                        "reason": note[:512],
                        "count": 1,
                        "scope": record.scope,
                    })

            fingerprint = graph_fingerprint(
                operations=operations,
                input_signatures=[_signature_dict(meta) for meta in variant.inputs],
                output_signatures=[_signature_dict(meta) for meta in variant.outputs],
                safe_constants=constants,
                parent_module=record.scope,
            )
            region: dict[str, Any] = {
                "name": _region_name(record.scope, shape_key),
                "fingerprint": fingerprint,
                "operations": operations,
                "dependencies": dependencies,
                "inputs": variant.inputs,
                "outputs": variant.outputs,
                "cuda_time_us": 0.0,
                "self_cuda_time_us": 0.0,
                "calls": max(variant.calls, 1),
                "rejection_reasons": sorted({note[:512]
                                             for note in notes}),
                "shape_frequency": {
                    shape_key: max(variant.calls, 1)
                },
                "parent_module": record.scope,
                "attributes": {
                    "module_class": record.class_name,
                    "tracer": self.tracer,
                    "capture_mode": capture_mode,
                    "capture_attempts": list(attempts)[:list(attempts).index(capture_mode) + 1],
                    "capture_failures": failures,
                },
            }
            if executable_ir is not None:
                region["attributes"]["executable_ir"] = executable_ir
            if constants:
                region["safe_constants"] = constants
            regions.append(region)
        return regions

    def finalize(self) -> dict[str, Any]:
        """Trace hooked scopes and build the metadata-only capture payload."""
        self.detach()
        regions: list[dict[str, Any]] = []
        for record in self._scopes.values():
            if not record.variants:
                continue
            try:
                regions.extend(self._regions_for(record))
            except Exception as exc:  # noqa: BLE001
                self._record_exception(
                    "finalize_failed",
                    exc,
                    scope=record.scope,
                )

        breaks = _coalesce(self._graph_breaks, ("scope", "reason"))
        unsupported = _coalesce(self._unsupported, ("op_name", "reason", "scope"))
        payload = {
            "capture": {
                "capture_schema_version":
                CAPTURE_SCHEMA_VERSION,
                "tracer":
                self.tracer,
                "scopes":
                sorted(self._scopes),
                "scope_calls": {
                    record.scope: record.calls
                    for record in self._scopes.values()
                },
                "dropped_scopes":
                self._dropped_scopes,
                "dropped_shape_variants":
                sum(record.dropped_variants for record in self._scopes.values()),
                "errors":
                list(self._errors),
                "capture_mode_breakdown":
                dict(
                    sorted((
                        mode,
                        sum(1 for region in regions if region.get("attributes", {}).get("capture_mode") == mode),
                    ) for mode in _CAPTURE_MODES)),
            },
            "regions": regions,
            "graph_breaks": breaks,
            "unsupported": unsupported,
        }
        assert_metadata_only(payload)
        return payload


def _coalesce(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Merge duplicate records, summing their ``count`` fields."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for record in records:
        identity = tuple(record.get(key) for key in keys)
        merged.setdefault(identity, dict(record))
        counts[identity] += int(record.get("count", 1))
    result = []
    for identity, record in merged.items():
        record["count"] = counts[identity]
        result.append(record)
    result.sort(key=lambda item: tuple(str(item.get(key)) for key in keys))
    return result
