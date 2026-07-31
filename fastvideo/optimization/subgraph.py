# SPDX-License-Identifier: Apache-2.0
"""Fail-closed replacement of an exported graph subregion."""

from __future__ import annotations

import operator
import copy
import weakref
from typing import Any
from collections.abc import Callable, Mapping

from torch import fx, nn

from fastvideo.optimization.artifact import ArtifactManifest, signature_key


class SubgraphRewriteError(RuntimeError):
    """The live exported graph does not satisfy an artifact rewrite recipe."""


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
    attributes = {str(node.target): node for node in graph.nodes if node.op == "get_attr"}
    for index, spec in enumerate(input_specs):
        kind = getattr(getattr(spec, "kind", None), "name", "")
        if kind == "USER_INPUT":
            if user_index >= len(placeholders):
                raise SubgraphRewriteError("runtime user input mapping changed")
            refs[f"p{index}"] = placeholders[user_index]
            user_index += 1
            continue
        target = str(getattr(spec, "target", ""))
        node = attributes.get(target)
        if node is None:
            raise SubgraphRewriteError("runtime lifted input mapping changed")
        refs[f"p{index}"] = node

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
        try:
            runnable = fx.GraphModule(parent_module, graph)
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
        for node in selected:
            external_users = [user for user in node.users if user not in selected_set]
            if external_users and node not in output_set:
                raise SubgraphRewriteError("rewrite recipe omits an externally used output")
        if any(node in selected_set for node in boundary):
            raise SubgraphRewriteError("rewrite boundary must be outside the selected nodes")

        def invoke(*values: Any) -> Any:
            return candidate(parent_module, *values)

        first = selected[0]
        with graph.inserting_before(first):
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
