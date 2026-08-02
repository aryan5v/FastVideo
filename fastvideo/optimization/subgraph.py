# SPDX-License-Identifier: Apache-2.0
"""Fail-closed replacement of an exported graph subregion."""

from __future__ import annotations

import copy
import json
import operator
import os
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import fx, nn
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import tree_unflatten

from fastvideo.logger import init_logger
from fastvideo.optimization import timing
from fastvideo.optimization.artifact import ArtifactManifest, signature_key


logger = init_logger(__name__)


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


#: Export emits a runtime metadata assertion for every tensor whose dtype or
#: device it wants to pin. They compute nothing, and the dispatcher already
#: checks the placeholder contract once per call in _validate_runtime_inputs,
#: against the same values export recorded. Replaying 68 of them per call --
#: measured on transformer.model.transformer_blocks -- is pure dispatch cost.
_ASSERTION_TARGETS = frozenset({"aten._assert_tensor_metadata.default"})


def _strip_runtime_assertions(graph: fx.Graph) -> int:
    """Remove export's tensor-metadata assertions. Returns how many went.

    Only nodes with no users are removed, so this can never drop a value the
    graph depends on. The metadata they assert is still enforced -- once per
    call, on the inputs, by the placeholder contract.
    """
    removed = 0
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function":
            continue
        if str(node.target) not in _ASSERTION_TARGETS:
            continue
        if node.users:
            continue
        graph.erase_node(node)
        removed += 1
    return removed


def _dump_graph_profile(graph: fx.Graph, manifest: ArtifactManifest) -> None:
    """Write an op histogram of the rewritten graph, when asked to.

    The rewritten graph is what the dispatcher executes in place of the
    module's own forward. When that path is slower than what it replaced, the
    first question is which operations it is actually running -- an export
    graph is decomposed, so a single high-level call in the module can become
    dozens of primitives here. Metadata only: op names and counts, never tensor
    values.
    """
    destination = os.getenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_DUMP_GRAPH", "")
    if not destination:
        return
    try:
        histogram = Counter(
            str(node.target) for node in graph.nodes if node.op == "call_function"
        )
        payload = {
            "artifact_id": manifest.artifact_id,
            "parent_module": manifest.parent_module,
            "nodes_total": sum(1 for _ in graph.nodes),
            "call_function_total": sum(histogram.values()),
            "op_counts": dict(histogram.most_common()),
            "op_kinds": {
                "placeholder": sum(1 for n in graph.nodes if n.op == "placeholder"),
                "get_attr": sum(1 for n in graph.nodes if n.op == "get_attr"),
                "call_module": sum(1 for n in graph.nodes if n.op == "call_module"),
                "call_method": sum(1 for n in graph.nodes if n.op == "call_method"),
            },
        }
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return


#: CUDA-graph replay of the rewritten subgraph. On by default: it is bitwise
#: identical to the eager replay by construction and removes the dispatch cost
#: that made subgraph artifacts uneconomic. Set to "0" to force eager replay.
_CUDA_GRAPHS_ENABLED = os.getenv(
    "FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS", "1"
) not in {"0", "false", "False", ""}


@dataclass
class _Entry:
    """Everything the dispatcher caches for one live repeated block."""

    runnable: fx.GraphModule
    contract: tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...]
    cuda_graph: "_CudaGraphRunner | None" = None


def _unflatten(flat_output: Any, output_spec: Any) -> Any:
    with timing.phase("subgraph.unflatten"):
        leaves = list(flat_output) if isinstance(flat_output, tuple) else [flat_output]
        return tree_unflatten(leaves, output_spec)


class _CudaGraphScope:
    """Static input buffers and a memory pool shared by one scope's captures.

    Every repeated block in a stack is called with the same input signature, so
    one set of static input buffers serves all of them: each block's capture
    records reads from the *same* addresses, and the dispatcher copies the live
    inputs in once before replaying whichever block it is on.

    This matters for memory, not speed. ``transformer.model.transformer_blocks``
    has 48 blocks and 27 placeholders, several of them large (a 1x4680x24576
    bfloat16 timestep tensor is 230MB). Per-block buffers would add roughly
    19GB against a 68GB baseline -- far past the workload's 5% peak-memory
    allowance -- while the tensors are largely identical across blocks anyway.
    Sharing a ``graph_pool_handle`` likewise stops each capture from reserving
    its own pool for intermediates.
    """

    def __init__(self) -> None:
        self.static_inputs: list[Any] | None = None
        self.pool: Any = None

    def prepare(self, flat_inputs: list[Any]) -> list[Any]:
        import torch

        if self.static_inputs is None:
            self.static_inputs = [value.clone().detach() for value in flat_inputs]
            # Only meaningful with a CUDA context; a pool handle is not
            # obtainable otherwise and is not needed, because capture will
            # decline first.
            if torch.cuda.is_available():
                self.pool = torch.cuda.graph_pool_handle()
        return self.static_inputs


class CudaGraphUnavailable(RuntimeError):
    """This rewritten graph cannot be replayed from a CUDA graph capture."""


class _CudaGraphRunner:
    """Replay a rewritten subgraph from a captured CUDA graph.

    Why this exists: the dispatcher executes an *export* graph, which is
    decomposed. For ``transformer.model.transformer_blocks`` that is 621
    ``call_function`` nodes per call, each paying Python and PyTorch dispatcher
    cost. Shadow timing measured the replay at 11.57ms against the module's own
    forward at 8.18ms on identical inputs -- a 3.39ms penalty that is entirely
    dispatch, not arithmetic, and that swamped the artifact's 124us saving.

    A CUDA graph replays *the same kernels, with the same parameters, in the
    same order*. It is therefore bitwise identical to the eager replay by
    construction -- unlike a compiler backend, which may fuse or reassociate
    and would put the workload's ``byte_equal`` parity at risk. All it removes
    is the host-side cost of getting there.

    Requirements, each checked rather than assumed: every runtime input is a
    CUDA tensor of fixed shape and dtype; warmup on a side stream runs cleanly;
    capture succeeds; outputs are tensors. Any failure raises
    :class:`CudaGraphUnavailable` and the caller falls back to eager replay
    permanently. Nothing here can change a result: if it cannot guarantee the
    same kernels, it declines.
    """

    #: Eager iterations before capture, so the allocator and any lazily
    #: initialized kernel state are warm when the capture is recorded.
    WARMUP_ITERATIONS = 3

    def __init__(self, runnable: fx.GraphModule, scope: _CudaGraphScope) -> None:
        self._runnable = runnable
        self._scope = scope
        self._graph: Any = None
        self._static_inputs: list[Any] = []
        self._static_outputs: tuple[Any, ...] = ()
        self._output_was_tuple = True
        self._warmups = 0

    def _capture(self, flat_inputs: list[Any]) -> None:
        import torch

        for index, value in enumerate(flat_inputs):
            if not isinstance(value, torch.Tensor):
                raise CudaGraphUnavailable(
                    f"runtime input {index} is {type(value).__name__}, not a tensor"
                )
            if not value.is_cuda:
                raise CudaGraphUnavailable(f"runtime input {index} is not on CUDA")
        if not torch.cuda.is_available():
            raise CudaGraphUnavailable("CUDA is not available")

        static_inputs = self._scope.prepare(flat_inputs)
        if len(static_inputs) != len(flat_inputs):
            raise CudaGraphUnavailable("scope input arity differs between blocks")
        for index, (static, live) in enumerate(zip(static_inputs, flat_inputs, strict=True)):
            if static.shape != live.shape or static.dtype != live.dtype:
                raise CudaGraphUnavailable(
                    f"runtime input {index} differs from the scope's captured layout"
                )
            static.copy_(live)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_ITERATIONS):
                self._runnable(*static_inputs)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        if self._scope.pool is None:
            raise CudaGraphUnavailable("no CUDA graph memory pool")
        with torch.cuda.graph(graph, pool=self._scope.pool):
            outputs = self._runnable(*static_inputs)

        leaves = outputs if isinstance(outputs, tuple) else (outputs,)
        self._output_was_tuple = isinstance(outputs, tuple)
        for position, leaf in enumerate(leaves):
            if not isinstance(leaf, torch.Tensor):
                raise CudaGraphUnavailable(
                    f"output {position} is {type(leaf).__name__}, not a tensor"
                )
        self._static_inputs = static_inputs
        self._static_outputs = tuple(leaves)
        self._graph = graph

    def __call__(self, flat_inputs: list[Any]) -> Any:
        import torch

        if self._graph is None:
            if self._warmups < self.WARMUP_ITERATIONS:
                # Let the eager path settle first; capturing on the very first
                # call would record one-time allocator and autotune work.
                self._warmups += 1
                raise CudaGraphUnavailable("warming up")
            try:
                self._capture(flat_inputs)
            except CudaGraphUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - capture is best effort
                # A failed capture must not escape as an arbitrary error: the
                # caller would read it as a candidate runtime fault and demote
                # the artifact for the rest of the run, when the kernel is
                # fine and only the acceleration is unavailable. Drop any
                # partial capture and let the eager replay proceed.
                self._graph = None
                self._static_outputs = ()
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    pass
                raise CudaGraphUnavailable(
                    f"capture failed: {type(exc).__name__}: {exc}"
                ) from exc

        if len(flat_inputs) != len(self._static_inputs):
            raise CudaGraphUnavailable("runtime input count changed after capture")
        for index, (static, live) in enumerate(
            zip(self._static_inputs, flat_inputs, strict=True)
        ):
            if not isinstance(live, torch.Tensor):
                raise CudaGraphUnavailable(f"runtime input {index} is no longer a tensor")
            if live.shape != static.shape or live.dtype != static.dtype:
                raise CudaGraphUnavailable(f"runtime input {index} changed shape or dtype")
            static.copy_(live)

        self._graph.replay()
        # The static outputs are overwritten by the next replay, so hand the
        # caller its own copies.
        copies = tuple(leaf.clone() for leaf in self._static_outputs)
        return copies if self._output_was_tuple else copies[0]


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
    cache: weakref.WeakKeyDictionary[nn.Module, "_Entry"] = weakref.WeakKeyDictionary()
    graph_scope = _CudaGraphScope()

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
        _strip_runtime_assertions(graph)
        graph.lint()
        runnable.recompile()
        _dump_graph_profile(graph, manifest)
        return runnable

    def dispatch(parent_module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        entry = cache.get(parent_module)
        if entry is None:
            runnable = build(parent_module)
            entry = _Entry(
                runnable=runnable,
                contract=_placeholder_contract(runnable.graph),
                cuda_graph=_CudaGraphRunner(runnable, graph_scope) if _CUDA_GRAPHS_ENABLED else None,
            )
            cache[parent_module] = entry
        runnable, contract = entry.runnable, entry.contract
        with timing.phase("subgraph.flatten"):
            flat_inputs = tree_flatten_spec((args, kwargs), input_spec)
        with timing.phase("subgraph.validate"):
            _validate_runtime_inputs(contract, flat_inputs)
        try:
            if entry.cuda_graph is not None:
                try:
                    with timing.phase("subgraph.execute_cuda_graph"):
                        flat_output = entry.cuda_graph(flat_inputs)
                    return _unflatten(flat_output, output_spec)
                except CudaGraphUnavailable as reason:
                    # Warmup, or this graph cannot be captured. Either way the
                    # eager replay below is the authority; a capture is only
                    # ever an accelerator for it, never a different answer.
                    if str(reason) != "warming up":
                        logger.info(
                            "CUDA graph replay unavailable for %s (%s); "
                            "continuing with eager replay",
                            manifest.artifact_id,
                            reason,
                        )
                        entry.cuda_graph = None
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
        return _unflatten(flat_output, output_spec)

    return dispatch
