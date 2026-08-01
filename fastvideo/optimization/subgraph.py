# SPDX-License-Identifier: Apache-2.0
"""Fail-closed replacement of an exported graph subregion."""

from __future__ import annotations

import copy
import operator
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from torch import fx, nn

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
    input_specs = list(exported.graph_signature.input_specs)
    lifted_specs = [
        spec
        for spec in input_specs
        if getattr(getattr(spec, "kind", None), "name", "") != "USER_INPUT"
    ]
    attributes = [node for node in graph.nodes if node.op == "get_attr"]
    if len(attributes) != len(lifted_specs):
        raise SubgraphRewriteError("export lifted input mapping changed")

    bindings: dict[str, Any] = {}
    for node, spec in zip(attributes, lifted_specs, strict=True):
        runtime_target = str(node.target)
        kind = getattr(getattr(spec, "kind", None), "name", "")
        source_target = str(getattr(spec, "target", ""))
        try:
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
    attributes = [node for node in graph.nodes if node.op == "get_attr"]
    lifted_index = 0
    for index, spec in enumerate(input_specs):
        kind = getattr(getattr(spec, "kind", None), "name", "")
        if kind == "USER_INPUT":
            if user_index >= len(placeholders):
                raise SubgraphRewriteError("runtime user input mapping changed")
            refs[f"p{index}"] = placeholders[user_index]
            user_index += 1
            continue
        if lifted_index >= len(attributes):
            raise SubgraphRewriteError("runtime lifted input mapping changed")
        refs[f"p{index}"] = attributes[lifted_index]
        lifted_index += 1
    if lifted_index != len(attributes):
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
    cache: weakref.WeakKeyDictionary[nn.Module, fx.GraphModule] = weakref.WeakKeyDictionary()

    def build(parent_module: nn.Module) -> fx.GraphModule:
        graph = copy.deepcopy(exported_graph)
        for node in list(graph.nodes):
            if node.op == "call_module" and str(node.target) == "_guards_fn":
                if node.users:
                    raise SubgraphRewriteError("export guard unexpectedly has users")
                graph.erase_node(node)
        bindings = _graph_bindings(exported, graph, parent_module)
        try:
            runnable = fx.GraphModule(bindings, graph)
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
        runnable = cache.get(parent_module)
        if runnable is None:
            runnable = build(parent_module)
            cache[parent_module] = runnable
        return runnable(*args, **kwargs)

    return dispatch
