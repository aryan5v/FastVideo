# SPDX-License-Identifier: Apache-2.0
"""Fail-closed replacement of an exported graph subregion."""

from __future__ import annotations

import copy
import operator
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from torch import fx, nn
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import tree_unflatten

from fastvideo.optimization import timing
from fastvideo.optimization.artifact import ArtifactManifest, signature_key


class SubgraphRewriteError(RuntimeError):
    """The live exported graph does not satisfy an artifact rewrite recipe."""


def _get_attr(root: Any, target: str) -> Any:
    """Resolve one FX qualified target without evaluating arbitrary source."""
    value = root
    for atom in target.split("."):
        if not atom:
            raise AttributeError(target)
        value = getattr(value, atom)
    return value


def _set_attr(root: Any, target: str, value: Any) -> None:
    """Replace one existing qualified attribute without changing its path."""
    atoms = target.split(".")
    if not atoms or any(not atom for atom in atoms):
        raise AttributeError(target)
    owner = root
    for atom in atoms[:-1]:
        owner = getattr(owner, atom)
    setattr(owner, atoms[-1], value)


def _tensor_key(value: Any) -> tuple[tuple[int, ...], str] | None:
    """Return metadata-only tensor identity without reading tensor values."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        return tuple(int(dim) for dim in shape), str(dtype)
    except (TypeError, ValueError, RuntimeError):
        return None


def _tensor_anomaly(value: Any) -> dict[str, Any] | None:
    key = _tensor_key(value)
    local = getattr(value, "_local_tensor", None)
    local_key = _tensor_key(local)
    if key is None:
        return None
    shape, dtype = key
    local_shape = local_key[0] if local_key is not None else None
    if 0 not in shape and (local_shape is None or 0 not in local_shape):
        return None
    return {
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "shape": shape,
        "local_shape": local_shape,
        "dtype": dtype,
    }


def _value_metadata(value: Any) -> Any:
    """Describe runtime operands without reading or serializing tensor values."""
    key = _tensor_key(value)
    if key is not None:
        local_key = _tensor_key(getattr(value, "_local_tensor", None))
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "shape": key[0],
            "local_shape": local_key[0] if local_key is not None else None,
            "dtype": key[1],
            "device": str(getattr(value, "device", "unknown")),
        }
    if isinstance(value, tuple):
        return tuple(_value_metadata(item) for item in value)
    if isinstance(value, list):
        return [_value_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value_metadata(item) for key, item in value.items()}
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


class _MetadataInterpreter(fx.Interpreter):
    """Annotate a failing FX node with metadata-only resolved operands."""

    def run_node(self, node: fx.Node) -> Any:
        try:
            return super().run_node(node)
        except RuntimeError as exc:
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            raise SubgraphRewriteError(
                f"rewritten node {node.name!r} ({node.target}) failed; "
                f"args={_value_metadata(args)}, kwargs={_value_metadata(kwargs)}"
            ) from exc


def _placeholder_contract(
    graph: fx.Graph,
) -> tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...]:
    """Freeze the graph's placeholder contract once, at build time.

    The rewritten graph is immutable for the lifetime of the dispatcher, so
    walking every node to rediscover its placeholders on each call re-derives a
    constant. That walk is O(graph size) in Python and runs inside the region
    it is meant to be accelerating.
    """
    return tuple(
        (str(node.target), _tensor_key((node.meta or {}).get("val")))
        for node in graph.nodes
        if node.op == "placeholder"
    )


def _validate_runtime_inputs(
    contract: tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...],
    flat_inputs: list[Any],
) -> None:
    if len(contract) != len(flat_inputs):
        raise SubgraphRewriteError(
            "runtime flattened input count differs from the exported graph"
        )
    for index, ((target, expected), value) in enumerate(
        zip(contract, flat_inputs, strict=True)
    ):
        if expected is None:
            continue
        actual = _tensor_key(value)
        if actual != expected:
            raise SubgraphRewriteError(
                f"runtime input {index} ({target}) metadata changed: "
                f"expected {expected}, observed {actual}"
            )


def _lifted_attribute_pairs(
    exported: Any,
    graph: fx.Graph,
) -> list[tuple[Any, fx.Node]]:
    """Pair lifted input specs with runtime attributes without order assumptions."""
    input_specs = list(exported.graph_signature.input_specs)
    lifted_specs = [
        spec
        for spec in input_specs
        if getattr(getattr(spec, "kind", None), "name", "") != "USER_INPUT"
    ]
    attributes = [node for node in graph.nodes if node.op == "get_attr"]
    if len(attributes) != len(lifted_specs):
        raise SubgraphRewriteError("export lifted input mapping changed")

    by_target: dict[str, fx.Node] = {}
    for node in attributes:
        target = str(node.target)
        if target in by_target:
            raise SubgraphRewriteError("export contains duplicate lifted attributes")
        by_target[target] = node

    paired: list[tuple[Any, fx.Node] | None] = [None] * len(lifted_specs)
    assigned: set[fx.Node] = set()
    for index, spec in enumerate(lifted_specs):
        source_target = str(getattr(spec, "target", "") or "")
        node = by_target.get(source_target)
        if node is not None and node not in assigned:
            paired[index] = (spec, node)
            assigned.add(node)

    unmatched_specs = [
        (index, spec)
        for index, spec in enumerate(lifted_specs)
        if paired[index] is None
    ]
    unmatched_nodes = [node for node in attributes if node not in assigned]
    if len(unmatched_specs) != len(unmatched_nodes):
        raise SubgraphRewriteError("export lifted input mapping is ambiguous")
    for (index, spec), node in zip(unmatched_specs, unmatched_nodes, strict=True):
        paired[index] = (spec, node)
    if any(item is None for item in paired):
        raise SubgraphRewriteError("export lifted input mapping is incomplete")
    return [item for item in paired if item is not None]


def _graph_bindings(
    exported: Any,
    graph: fx.Graph,
    parent_module: nn.Module,
) -> dict[str, Any]:
    """Bind an exported graph to one live repeated block.

    ``ExportedProgram.module()`` turns lifted inputs into ``get_attr`` nodes.
    Parameters and buffers must resolve against the live block so every block
    keeps its own weights. Export-created constants, however, can have opaque
    names such as ``lifted_tensor_0`` that do not exist on the original block;
    those are immutable capture-time constants and must resolve against the
    representative exported module.
    """
    exported_module = exported.module()
    bindings: dict[str, Any] = {}
    for spec, node in _lifted_attribute_pairs(exported, graph):
        runtime_target = str(node.target)
        kind = getattr(getattr(spec, "kind", None), "name", "")
        source_target = str(getattr(spec, "target", ""))
        try:
            # Prefer the exact attribute requested by the runnable graph. Some
            # inference loaders expose ordinary Parameters with requires_grad
            # disabled in a way Export classifies like a lifted constant; its
            # representative storage may already have been released to an
            # empty placeholder even though the live block owns real storage.
            try:
                bindings[runtime_target] = _get_attr(
                    parent_module, runtime_target
                )
                continue
            except AttributeError:
                pass
            if kind in {"PARAMETER", "BUFFER"}:
                if not source_target:
                    raise AttributeError(runtime_target)
                bindings[runtime_target] = _get_attr(parent_module, source_target)
            else:
                bindings[runtime_target] = _get_attr(exported_module, runtime_target)
        except (AttributeError, KeyError) as exc:
            raise SubgraphRewriteError(
                "live module does not satisfy exported attributes"
            ) from exc

    # Export normally functionalizes nested modules into call_function nodes.
    # Preserve a fail-closed fallback for any call_module target that remains.
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        target = str(node.target)
        try:
            bindings[target] = _get_attr(parent_module, target)
        except AttributeError:
            try:
                bindings[target] = _get_attr(exported_module, target)
            except AttributeError as exc:
                raise SubgraphRewriteError(
                    "live module does not satisfy exported submodules"
                ) from exc
    return bindings


def _runtime_nodes(exported: Any, runnable: fx.GraphModule) -> dict[str, fx.Node]:
    """Map canonical capture refs (``p#``/``n#``) onto a runnable graph."""
    raw_graph = exported.graph_module.graph
    graph = runnable.graph
    refs: dict[str, fx.Node] = {}

    raw_placeholders = [node for node in raw_graph.nodes if node.op == "placeholder"]
    input_specs = list(exported.graph_signature.input_specs)
    if len(raw_placeholders) != len(input_specs):
        raise SubgraphRewriteError("export input signature changed")

    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    user_index = 0
    lifted_pairs = _lifted_attribute_pairs(exported, graph)
    lifted_nodes = iter(node for _spec, node in lifted_pairs)
    for index, spec in enumerate(input_specs):
        kind = getattr(getattr(spec, "kind", None), "name", "")
        if kind == "USER_INPUT":
            if user_index >= len(placeholders):
                raise SubgraphRewriteError("runtime user input mapping changed")
            refs[f"p{index}"] = placeholders[user_index]
            user_index += 1
            continue
        try:
            attribute = next(lifted_nodes)
        except StopIteration as exc:
            raise SubgraphRewriteError(
                "runtime lifted input mapping changed"
            ) from exc
        refs[f"p{index}"] = attribute
    try:
        next(lifted_nodes)
    except StopIteration:
        pass
    else:
        raise SubgraphRewriteError("runtime lifted input mapping changed")

    raw_ops = [node for node in raw_graph.nodes if node.op in {"call_function", "call_method", "call_module"}]
    runtime_ops = [
        node for node in graph.nodes if node.op in {"call_function", "call_method", "call_module"}
        and not (node.op == "call_module" and str(node.target) == "_guards_fn")
    ]
    if len(raw_ops) != len(runtime_ops):
        raise SubgraphRewriteError("runtime operation count changed")
    for index, (raw, live) in enumerate(zip(raw_ops, runtime_ops, strict=True)):
        if raw.op != live.op or str(raw.target) != str(live.target):
            raise SubgraphRewriteError("runtime operation ordering changed")
        refs[f"n{index}"] = live
    return refs


def _meta_keys(region: Mapping[str, Any], refs: tuple[str, ...]) -> tuple[tuple[Any, ...], ...]:
    attributes = region.get("attributes")
    if not isinstance(attributes, Mapping):
        raise SubgraphRewriteError("capture attributes missing")
    ir = attributes.get("executable_ir")
    if not isinstance(ir, Mapping):
        raise SubgraphRewriteError("capture executable IR missing")
    metadata: dict[str, dict[str, Any]] = {}
    for section in ("inputs", "nodes"):
        items = ir.get(section)
        if not isinstance(items, list):
            raise SubgraphRewriteError("capture executable IR is malformed")
        for item in items:
            if not isinstance(item, Mapping):
                raise SubgraphRewriteError("capture executable IR is malformed")
            item_id = item.get("id")
            meta = item.get("meta")
            if isinstance(item_id, str) and isinstance(meta, dict):
                metadata[item_id] = meta
    try:
        return tuple(signature_key(metadata[ref]) for ref in refs)
    except KeyError as exc:
        raise SubgraphRewriteError("rewrite boundary metadata missing") from exc


def subgraph_signature_keys(
    region: Mapping[str, Any],
    manifest: ArtifactManifest,
) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    """Return live boundary input/output signatures for compatibility checks."""
    return (
        _meta_keys(region, manifest.boundary_refs),
        _meta_keys(region, manifest.output_node_ids),
    )


def rewrite_exported_subgraph(
    exported: Any,
    manifest: ArtifactManifest,
    candidate: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a module-first dispatcher with exactly one subgraph replaced.

    The graph structure comes from the representative export, while every
    repeated block gets a tiny GraphModule rooted in that live block. Thus
    get-attr nodes read that block's own parameters without copying weights.
    """
    if manifest.target_kind != "subgraph" or manifest.capture_mode != "export":
        raise SubgraphRewriteError("artifact is not an export subgraph target")
    exported_graph = exported.module().graph
    input_spec = getattr(getattr(exported, "call_spec", None), "in_spec", None)
    output_spec = getattr(getattr(exported, "call_spec", None), "out_spec", None)
    if input_spec is None or output_spec is None:
        raise SubgraphRewriteError("export call specification is missing")
    # Value is (rewritten module, frozen placeholder contract): the contract is
    # derived once per build so the per-call path never walks the graph.
    cache: weakref.WeakKeyDictionary[
        nn.Module,
        tuple[fx.GraphModule, tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...]],
    ] = weakref.WeakKeyDictionary()

    def build(parent_module: nn.Module) -> fx.GraphModule:
        representative = exported.module()
        graph = copy.deepcopy(exported_graph)
        # Execute the rewritten graph with explicit flat leaves. Export's
        # generated PyTreeCodeGen wrapper is tied to module attributes and has
        # proven fragile when transplanted into a fresh GraphModule for real
        # model signatures containing tuples and specialized scalar leaves.
        graph.set_codegen(fx.graph.CodeGen())
        for node in list(graph.nodes):
            if node.op == "call_module" and str(node.target) == "_guards_fn":
                if node.users:
                    raise SubgraphRewriteError("export guard unexpectedly has users")
                graph.erase_node(node)
        bindings = _graph_bindings(exported, graph, parent_module)
        try:
            # Rooting the rewritten module in Export's GraphModule retains its
            # pytree/call machinery and opaque lifted constants. Constructing
            # from a plain mapping is sufficient for simple tensor-only CPU
            # graphs but can corrupt real exported model calling conventions.
            runnable = fx.GraphModule(representative, graph)
            for target, value in bindings.items():
                attribute_node = next(
                    (
                        node
                        for node in graph.nodes
                        if node.op == "get_attr" and str(node.target) == target
                    ),
                    None,
                )
                expected = _tensor_key(
                    (attribute_node.meta or {}).get("val")
                    if attribute_node is not None
                    else None
                )
                actual = _tensor_key(value)
                if expected is not None and actual != expected:
                    raise SubgraphRewriteError(
                        f"live attribute {target!r} metadata changed: "
                        f"expected {expected}, observed {actual}"
                    )
                _set_attr(runnable, target, value)
        except Exception as exc:  # noqa: BLE001 - graph/module mismatch fails closed
            raise SubgraphRewriteError("live module does not satisfy exported attributes") from exc
        refs = _runtime_nodes(exported, runnable)
        try:
            selected = [refs[item] for item in manifest.selected_node_ids]
            boundary = [refs[item] for item in manifest.boundary_refs]
            outputs = [refs[item] for item in manifest.output_node_ids]
        except KeyError as exc:
            raise SubgraphRewriteError("rewrite recipe references an unknown node") from exc

        selected_set = set(selected)
        output_set = set(outputs)
        positions = {node: index for index, node in enumerate(graph.nodes)}
        for node in selected:
            external_users = [user for user in node.users if user not in selected_set]
            if external_users and node not in output_set:
                raise SubgraphRewriteError("rewrite recipe omits an externally used output")
        if any(node in selected_set for node in boundary):
            raise SubgraphRewriteError("rewrite boundary must be outside the selected nodes")

        # A subgraph artifact is one call and therefore needs one legal point
        # in the parent graph: all boundary values must exist before it, and
        # every external consumer of a replaced output must follow it. A
        # dependency-connected recipe can still violate this when it spans an
        # unsupported operation. Reject that recipe before mutating the graph.
        latest_boundary = max(boundary, key=positions.__getitem__)
        external_consumers = {
            user
            for output in outputs
            for user in output.users
            if user not in selected_set
        }
        if any(
            positions[user] <= positions[latest_boundary]
            for user in external_consumers
        ):
            raise SubgraphRewriteError(
                "rewrite recipe has no valid topological insertion point"
            )

        def invoke(*values: Any) -> Any:
            return candidate(parent_module, *values)

        with graph.inserting_after(latest_boundary):
            replacement = graph.call_function(invoke, args=tuple(boundary))
        if len(outputs) == 1:
            replacements = [replacement]
        else:
            replacements = []
            cursor = replacement
            for index in range(len(outputs)):
                with graph.inserting_after(cursor):
                    cursor = graph.call_function(operator.getitem, args=(replacement, index))
                replacements.append(cursor)
        for original, new in zip(outputs, replacements, strict=True):
            original.replace_all_uses_with(new)
        for node in reversed(selected):
            if node.users:
                raise SubgraphRewriteError("selected node still has users after replacement")
            graph.erase_node(node)
        graph.lint()
        runnable.recompile()
        return runnable

    def dispatch(parent_module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        entry = cache.get(parent_module)
        if entry is None:
            runnable = build(parent_module)
            entry = (runnable, _placeholder_contract(runnable.graph))
            cache[parent_module] = entry
        runnable, contract = entry
        with timing.phase("subgraph.flatten"):
            flat_inputs = tree_flatten_spec((args, kwargs), input_spec)
        with timing.phase("subgraph.validate"):
            _validate_runtime_inputs(contract, flat_inputs)
        try:
            with timing.phase("subgraph.execute"):
                flat_output = runnable(*flat_inputs)
        except RuntimeError as exc:
            anomalies = {
                "attributes": {
                    str(node.target): detail
                    for node in runnable.graph.nodes
                    if node.op == "get_attr"
                    if (detail := _tensor_anomaly(
                        _get_attr(runnable, str(node.target))
                    ))
                    is not None
                },
                "inputs": {
                    str(index): detail
                    for index, value in enumerate(flat_inputs)
                    if (detail := _tensor_anomaly(value)) is not None
                },
            }
            try:
                _MetadataInterpreter(runnable).run(*flat_inputs)
            except SubgraphRewriteError as diagnostic:
                raise diagnostic from exc
            raise SubgraphRewriteError(
                f"rewritten export execution failed; metadata anomalies: {anomalies}"
            ) from exc
        with timing.phase("subgraph.unflatten"):
            leaves = list(flat_output) if isinstance(flat_output, tuple) else [flat_output]
            return tree_unflatten(leaves, output_spec)

    return dispatch
