# SPDX-License-Identifier: Apache-2.0
"""Fail-closed replacement of an exported graph subregion."""

from __future__ import annotations

import copy
import json
import operator
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import fx, nn
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import tree_unflatten

from fastvideo import envs
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


def _layout_identity(value: Any) -> tuple[Any, ...] | None:
    """Everything a capture bakes in about one tensor's memory.

    ``data_ptr()`` alone is not enough: it folds in ``storage_offset`` but says
    nothing about strides, so a contiguous tensor and a permuted view handed
    back the same allocator block compare equal while the captured kernels read
    elements in a different order. Device and dtype are included for the same
    reason -- the capture encodes all of it.
    """
    try:
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            value.storage_offset(),
        )
    except Exception:  # noqa: BLE001 - anything unreadable is not capturable
        return None


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
            raise SubgraphRewriteError(f"rewritten node {node.name!r} ({node.target}) failed; "
                                       f"args={_value_metadata(args)}, kwargs={_value_metadata(kwargs)}") from exc


#: Export emits a runtime metadata assertion for every tensor whose dtype or
#: device it wants to pin. They compute nothing, and the dispatcher already
#: checks the placeholder contract once per call in _validate_runtime_inputs,
#: against the same values export recorded. Replaying 68 of them per call --
#: measured on transformer.model.transformer_blocks -- is pure dispatch cost.
#: Matched on the operator's canonical name rather than ``str(node.target)``.
#: An OpOverload can stringify as "aten._assert_tensor_metadata.default" or
#: "torch.ops.aten._assert_tensor_metadata.default" depending on how the graph
#: was produced, and an OpOverloadPacket drops the ".default" entirely, so a
#: single literal comparison silently stopped matching and left the assertions
#: in the replayed graph.
_ASSERTION_TARGETS = frozenset({"aten._assert_tensor_metadata.default"})


def _canonical_target_name(target: Any) -> str:
    """Best-effort canonical ``namespace.op.overload`` name for an FX target."""
    namespace = getattr(target, "namespace", None)
    name = getattr(target, "_opname", None) or getattr(target, "__name__", None)
    overload = getattr(target, "_overloadname", None)
    if namespace and name:
        return f"{namespace}.{name}.{overload}" if overload else f"{namespace}.{name}"
    text = str(target)
    return text[len("torch.ops."):] if text.startswith("torch.ops.") else text


def _is_assertion_target(target: Any) -> bool:
    """Whether an FX call target is an export runtime metadata assertion."""
    canonical = _canonical_target_name(target)
    if canonical in _ASSERTION_TARGETS:
        return True
    # Tolerate a packet (no overload) and the torch.ops.-prefixed spelling.
    return any(canonical == declared.rsplit(".", 1)[0] for declared in _ASSERTION_TARGETS)


def _strip_runtime_assertions(graph: fx.Graph) -> int:
    """Remove export's tensor-metadata assertions. Returns how many went.

    Only nodes with no users are removed, so this can never drop a value the
    graph depends on, and the arithmetic is untouched. Note this is a genuine
    reduction in checking, not a free win: the placeholder contract re-checks
    shape and dtype on the *inputs* every call, but these assertions also
    covered intermediates and lifted parameters.
    """
    removed = 0
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function":
            continue
        if not _is_assertion_target(node.target):
            continue
        if node.users:
            continue
        graph.erase_node(node)
        removed += 1
    return removed


def _dump_graph_profile(graph: fx.Graph, manifest: ArtifactManifest, *, block_identity: str) -> None:
    """Write an op histogram of the rewritten graph, when asked to.

    The rewritten graph is what the dispatcher executes in place of the
    module's own forward. When that path is slower than what it replaced, the
    first question is which operations it is actually running -- an export
    graph is decomposed, so a single high-level call in the module can become
    dozens of primitives here. Metadata only: op names and counts, never tensor
    values.
    """
    destination = envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_DUMP_GRAPH
    if not destination:
        return
    try:
        histogram = Counter(str(node.target) for node in graph.nodes if node.op == "call_function")
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
        configured = Path(destination).expanduser()
        suffix = configured.suffix or ".json"
        output = configured.with_name(f"{configured.stem}.{manifest.artifact_id}.{block_identity}{suffix}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return


#: CUDA-graph replay of the rewritten subgraph. On by default: it is bitwise
#: identical to the eager replay by construction and removes the dispatch cost
#: that made subgraph artifacts uneconomic. Set to "0" to force eager replay.
_CUDA_GRAPHS_ENABLED = envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS


@dataclass
class _Entry:
    """Everything the dispatcher caches for one live repeated block."""

    runnable: fx.GraphModule
    contract: tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...]
    cuda_graph: _CudaGraphRunner | None = None


def _unflatten(flat_output: Any, output_spec: Any) -> Any:
    with timing.phase("subgraph.unflatten"):
        leaves = list(flat_output) if isinstance(flat_output, tuple) else [flat_output]
        return tree_unflatten(leaves, output_spec)


class _CudaGraphScope:
    """Static input buffers shared by one scope's captures.

    Every repeated block in a stack is called with the same input signature, so
    one buffer serves all of them at a given position: each block's capture
    records reads from the same address, and the dispatcher refreshes it once
    before replaying whichever block it is on.

    Memory pools are deliberately *not* shared: see the capture site. Buffers
    are allocated only for inputs that actually move. Most of a
    diffusion block's boundary tensors -- timesteps, positional embeddings,
    the text context -- are the *same tensor object* for all 48 blocks and all
    8 steps of a generation, so the capture can read them where they already
    live. On ``transformer.model.transformer_blocks`` the 27 placeholders total
    roughly 565MB per call; copying only the ones that move avoids most of
    that traffic, and avoids reserving the memory to copy into.
    """

    def __init__(self) -> None:
        self.buffers: dict[int, Any] = {}

    def buffer_for(self, index: int, template: Any) -> Any:
        """A static buffer for one moving input position, shared across blocks."""
        existing = self.buffers.get(index)
        if existing is not None:
            if (existing.shape != template.shape or existing.stride() != template.stride()
                    or existing.dtype != template.dtype or existing.device != template.device):
                raise CudaGraphUnavailable(f"scope input {index} changed layout between blocks")
            return existing
        buffer = template.clone().detach()
        self.buffers[index] = buffer
        return buffer


class CudaGraphUnavailable(RuntimeError):
    """This rewritten graph cannot be replayed from a CUDA graph capture."""


class CudaGraphWarmingUp(CudaGraphUnavailable):
    """Not yet ready to capture. Distinct from a refusal, which is permanent."""


def _classify_output_leaves(leaves: tuple[Any, ...]) -> dict[int, Any]:
    """Return immutable non-tensor outputs, declining every mutable leaf."""
    import torch

    constants: dict[int, Any] = {}
    for position, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            continue
        if leaf is None or isinstance(leaf, bool | int | float | complex | str | bytes):
            constants[position] = leaf
            continue
        raise CudaGraphUnavailable(f"output {position} is a mutable {type(leaf).__name__}; "
                                   "returning it uncopied would alias the graph's static buffers")
    return constants


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
        self._static_outputs: tuple[Any, ...] = ()
        self._output_was_tuple = True
        self._warmups = 0
        self._arity = 0
        #: Input addresses seen during warmup, used to decide which inputs are
        #: stable enough to be read in place.
        self._observed: list[list[tuple[Any, ...] | None]] = []
        #: index -> captured address, for inputs read where they live.
        self._pinned: dict[int, tuple[Any, ...]] = {}
        #: index -> static buffer, for inputs that move between calls.
        self._moving: dict[int, Any] = {}
        #: index -> value, for non-tensor leaves baked into the capture.
        self._constants: dict[int, Any] = {}
        #: position -> value, for non-tensor outputs the graph reproduces.
        self._output_constants: dict[int, Any] = {}
        #: (target, layout) for every parameter/buffer the capture baked in.
        self._attributes: list[tuple[str, tuple[Any, ...]]] = []
        #: A capture taken but not yet verified, so cleanup can release it.
        self._pending_graph: Any = None
        #: The stream on which the CUDA graph was captured and is replayed.
        self._capture_stream: Any = None

    def _capture(self, flat_inputs: list[Any]) -> None:
        import torch

        if not torch.cuda.is_available():
            raise CudaGraphUnavailable("CUDA is not available")
        for index, value in enumerate(flat_inputs):
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                raise CudaGraphUnavailable(f"runtime input {index} is not on CUDA")

        # An input whose address never moved across warmup can be read where it
        # lives; one that moved needs a static buffer refreshed per call.
        pointers = [_layout_identity(value) if isinstance(value, torch.Tensor) else None for value in flat_inputs]
        for index, (value, identity) in enumerate(zip(flat_inputs, pointers, strict=True)):
            if isinstance(value, torch.Tensor) and identity is None:
                raise CudaGraphUnavailable(f"runtime input {index} has no readable layout")
        stable = [
            all(observed[index] == pointers[index] for observed in self._observed) for index in range(len(flat_inputs))
        ]

        capture_inputs: list[Any] = []
        self._pinned = {}
        self._moving = {}
        self._constants = {}
        for index, value in enumerate(flat_inputs):
            if not isinstance(value, torch.Tensor):
                # A non-tensor leaf -- export flattens Python scalars and flags
                # through as graph inputs. It holds no device memory, so it is
                # baked into the capture as the constant it is. The graph is
                # only valid while it keeps that value, which __call__ checks.
                capture_inputs.append(value)
                self._constants[index] = value
                continue
            if stable[index]:
                identity = pointers[index]
                if identity is None:  # defensive; tensor layouts were checked above
                    raise CudaGraphUnavailable(f"runtime input {index} has no readable layout")
                capture_inputs.append(value)
                self._pinned[index] = identity
                continue
            buffer = self._scope.buffer_for(index, value)
            buffer.copy_(value)
            capture_inputs.append(buffer)
            self._moving[index] = buffer

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_ITERATIONS):
                self._runnable(*capture_inputs)
        torch.cuda.current_stream().wait_stream(stream)

        # Each capture gets its own memory pool. Sharing one across a stack's
        # blocks trips the caching allocator's `use_count > 0` assert: every
        # graph keeps its output buffers alive for the life of the run, so the
        # pool is never free of live allocations when the next capture starts.
        graph = torch.cuda.CUDAGraph()
        self._pending_graph = graph
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            outputs = self._runnable(*capture_inputs)

        leaves = outputs if isinstance(outputs, tuple) else (outputs, )
        self._output_was_tuple = isinstance(outputs, tuple)
        # Tensor outputs live in the graph's static buffers and are copied out.
        # A non-tensor output is returned as captured, which is only safe for a
        # genuinely immutable value: a list or dict leaf would hand the caller
        # the static buffers themselves, and the next replay would rewrite what
        # it is holding. Anything else declines.
        self._output_constants = _classify_output_leaves(leaves)
        # The capture also baked in every parameter and buffer the graph reads
        # through get_attr. Those are not inputs, so nothing above covers them,
        # and they are not stable in general: FSDP2's reshard frees the
        # all-gathered storage the capture just recorded pointers into, and any
        # offload or dequantization cache moves them too. Record their identity
        # and re-check it on every replay.
        self._attributes = []
        for node in self._runnable.graph.nodes:
            if node.op != "get_attr":
                continue
            target = str(node.target)
            bound = _get_attr(self._runnable, target)
            if not isinstance(bound, torch.Tensor):
                continue
            identity = _layout_identity(bound)
            if identity is None:
                raise CudaGraphUnavailable(f"bound attribute {target!r} has no readable layout")
            self._attributes.append((target, identity))

        self._arity = len(flat_inputs)
        self._static_outputs = tuple(leaves)

        # Everything above argues the capture must be bitwise identical to the
        # eager replay. This checks it. The graph contains one node that export
        # did not functionalize -- the artifact's own entry point -- so purity
        # is an assumption about third-party code, and the workload's contract
        # is byte_equal. Verify once per block, then trust the replay.
        with torch.cuda.stream(stream):
            graph.replay()
            replayed = tuple(leaf if position in self._output_constants else leaf.clone()
                             for position, leaf in enumerate(self._static_outputs))
            reference = self._runnable(*capture_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        reference_leaves = reference if isinstance(reference, tuple) else (reference, )
        if len(reference_leaves) != len(replayed):
            raise CudaGraphUnavailable("capture changed the output arity")
        for position, (captured_leaf, eager_leaf) in enumerate(zip(replayed, reference_leaves, strict=True)):
            if position in self._output_constants:
                if captured_leaf != eager_leaf:
                    raise CudaGraphUnavailable(f"replayed constant output {position} differs from eager")
                continue
            if not torch.equal(captured_leaf, eager_leaf):
                raise CudaGraphUnavailable(f"replayed output {position} is not bitwise equal to the "
                                           "eager replay; refusing to use the capture")

        # Published only once verified. Until this assignment the runner has no
        # graph to replay, so a rejected capture can never be used.
        self._graph = graph
        self._capture_stream = stream
        self._pending_graph = None

    def _release_capture(self) -> None:
        """Drop a capture and free the private memory pool it reserved."""
        aborted, self._graph = self._graph, None
        self._static_outputs = ()
        self._attributes = []
        self._capture_stream = None
        pending = self._pending_graph
        self._pending_graph = None
        for candidate in (aborted, pending):
            try:
                if candidate is not None:
                    candidate.reset()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass

    def __call__(self, flat_inputs: list[Any]) -> Any:
        import torch

        if self._graph is None:
            if self._warmups < self.WARMUP_ITERATIONS:
                # Let the eager path settle first; capturing on the very first
                # call would record one-time allocator and autotune work. The
                # addresses seen here are what decide which inputs need a
                # static buffer.
                self._warmups += 1
                self._observed.append(
                    [_layout_identity(value) if isinstance(value, torch.Tensor) else None for value in flat_inputs])
                raise CudaGraphWarmingUp("warming up")
            try:
                self._capture(flat_inputs)
            except CudaGraphUnavailable:
                # A refused capture releases its private pool here, the same as
                # the unexpected-error path below. Relying on the caller
                # dropping the runner would leave the two paths asymmetric and
                # the pool's lifetime dependent on refcounting.
                self._release_capture()
                raise
            except Exception as exc:  # noqa: BLE001 - capture is best effort
                # A failed capture must not escape as an arbitrary error: the
                # caller would read it as a candidate runtime fault and demote
                # the artifact for the rest of the run, when the kernel is fine
                # and only the acceleration is unavailable.
                self._release_capture()
                with suppress(Exception):
                    torch.cuda.synchronize()
                raise CudaGraphUnavailable(f"capture failed: {type(exc).__name__}: {exc}") from exc

        if len(flat_inputs) != self._arity:
            raise CudaGraphUnavailable("runtime input count changed after capture")

        for index, live in enumerate(flat_inputs):
            if index in self._constants:
                captured = self._constants[index]
                # The captured kernels encode this value. A different one means
                # the graph computes the wrong thing, so decline rather than
                # replay it.
                try:
                    unchanged = type(live) is type(captured) and bool(live == captured)
                except Exception:  # noqa: BLE001 - an uncomparable leaf is not trusted
                    unchanged = False
                if not unchanged:
                    raise CudaGraphUnavailable(f"runtime input {index} changed from the captured constant")
                continue
            if not isinstance(live, torch.Tensor):
                raise CudaGraphUnavailable(f"runtime input {index} is no longer a tensor")
            pinned = self._pinned.get(index)
            if pinned is not None:
                # The capture reads this tensor where it lives. If it has moved,
                # replaying would read whatever now occupies that address, so
                # decline and let the eager replay answer instead.
                if _layout_identity(live) != pinned:
                    raise CudaGraphUnavailable(f"runtime input {index} moved or changed layout after capture")
                continue
            buffer = self._moving.get(index)
            if buffer is None:
                raise CudaGraphUnavailable(f"runtime input {index} has no captured storage")
            if live.shape != buffer.shape or live.dtype != buffer.dtype:
                raise CudaGraphUnavailable(f"runtime input {index} changed shape or dtype")
            buffer.copy_(live)

        for target, identity in self._attributes:
            if _layout_identity(_get_attr(self._runnable, target)) != identity:
                # A weight moved since capture. Replaying would read whatever
                # now occupies that address -- freed memory after an FSDP
                # reshard, or another tensor entirely.
                raise CudaGraphUnavailable(f"bound attribute {target!r} moved after capture")

        capture_stream = self._capture_stream
        if capture_stream is None:
            raise CudaGraphUnavailable("capture stream is unavailable")
        current_stream = torch.cuda.current_stream()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            self._graph.replay()
        current_stream.wait_stream(capture_stream)
        # The static tensor outputs are overwritten by the next replay, so hand
        # the caller its own copies; constants are returned as captured.
        copies = tuple(leaf if position in self._output_constants else leaf.clone()
                       for position, leaf in enumerate(self._static_outputs))
        return copies if self._output_was_tuple else copies[0]


def _placeholder_contract(graph: fx.Graph, ) -> tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...]:
    """Freeze the graph's placeholder contract once, at build time.

    The rewritten graph is immutable for the lifetime of the dispatcher, so
    walking every node to rediscover its placeholders on each call re-derives a
    constant. That walk is O(graph size) in Python and runs inside the region
    it is meant to be accelerating.
    """
    return tuple((str(node.target), _tensor_key((node.meta or {}).get("val"))) for node in graph.nodes
                 if node.op == "placeholder")


def _validate_runtime_inputs(
    contract: tuple[tuple[str, tuple[tuple[int, ...], str] | None], ...],
    flat_inputs: list[Any],
) -> None:
    if len(contract) != len(flat_inputs):
        raise SubgraphRewriteError("runtime flattened input count differs from the exported graph")
    for index, ((target, expected), value) in enumerate(zip(contract, flat_inputs, strict=True)):
        if expected is None:
            continue
        actual = _tensor_key(value)
        if actual != expected:
            raise SubgraphRewriteError(f"runtime input {index} ({target}) metadata changed: "
                                       f"expected {expected}, observed {actual}")


def _lifted_attribute_pairs(
    exported: Any,
    graph: fx.Graph,
) -> list[tuple[Any, fx.Node]]:
    """Pair lifted input specs with runtime attributes without order assumptions."""
    input_specs = list(exported.graph_signature.input_specs)
    lifted_specs = [spec for spec in input_specs if getattr(getattr(spec, "kind", None), "name", "") != "USER_INPUT"]
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

    unmatched_specs = [(index, spec) for index, spec in enumerate(lifted_specs) if paired[index] is None]
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
                bindings[runtime_target] = _get_attr(parent_module, runtime_target)
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
            raise SubgraphRewriteError("live module does not satisfy exported attributes") from exc

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
                raise SubgraphRewriteError("live module does not satisfy exported submodules") from exc
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
            raise SubgraphRewriteError("runtime lifted input mapping changed") from exc
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
    # Each _Entry owns the rewritten module, frozen placeholder contract, and
    # optional CUDA-graph runner. The contract is derived once per build so the
    # per-call path never walks the graph.
    cache: weakref.WeakKeyDictionary[nn.Module, _Entry] = weakref.WeakKeyDictionary()
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
                    (node for node in graph.nodes if node.op == "get_attr" and str(node.target) == target),
                    None,
                )
                expected = _tensor_key((attribute_node.meta or {}).get("val") if attribute_node is not None else None)
                actual = _tensor_key(value)
                if expected is not None and actual != expected:
                    raise SubgraphRewriteError(f"live attribute {target!r} metadata changed: "
                                               f"expected {expected}, observed {actual}")
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
        external_consumers = {user for output in outputs for user in output.users if user not in selected_set}
        if any(positions[user] <= positions[latest_boundary] for user in external_consumers):
            raise SubgraphRewriteError("rewrite recipe has no valid topological insertion point")

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
        _dump_graph_profile(
            graph,
            manifest,
            block_identity=f"{id(parent_module):x}",
        )
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
                except CudaGraphWarmingUp:
                    timing.note("cuda_graph_warmup")
                except CudaGraphUnavailable as reason:
                    # This graph cannot be captured. The eager replay below is
                    # the authority; a capture is only ever an accelerator for
                    # it, never a different answer.
                    timing.note(f"cuda_graph_declined: {reason}")
                    logger.warning_once(f"CUDA graph replay unavailable for {manifest.artifact_id} "
                                        f"({reason}); continuing with eager replay")
                    entry.cuda_graph = None
            with timing.phase("subgraph.execute"):
                flat_output = runnable(*flat_inputs)
        except RuntimeError as exc:
            anomalies = {
                "attributes": {
                    str(node.target): detail
                    for node in runnable.graph.nodes if node.op == "get_attr"
                    if (detail := _tensor_anomaly(_get_attr(runnable, str(node.target)))) is not None
                },
                "inputs": {
                    str(index): detail
                    for index, value in enumerate(flat_inputs) if (detail := _tensor_anomaly(value)) is not None
                },
            }
            try:
                _MetadataInterpreter(runnable).run(*flat_inputs)
            except SubgraphRewriteError as diagnostic:
                raise diagnostic from exc
            raise SubgraphRewriteError(f"rewritten export execution failed; metadata anomalies: {anomalies}") from exc
        return _unflatten(flat_output, output_spec)

    return dispatch
