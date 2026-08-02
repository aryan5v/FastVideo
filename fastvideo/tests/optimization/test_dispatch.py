# SPDX-License-Identifier: Apache-2.0
"""CPU tests for generic artifact dispatch and native fallback.

Every kernel and module here is a CPU fake: no CUDA, no Triton, and no model
code. What is exercised is the contract -- identity matching, hash
verification, fallback behavior and diagnostics -- not any particular model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from fastvideo import envs
from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.optimization.artifact import (
    ANY,
    MANIFEST_FILENAME,
    ArtifactError,
    ArtifactManifest,
    ArtifactRegistry,
    RuntimeProfile,
    load_entry_point,
    verify_bundle,
)
from fastvideo.optimization.dispatch import (REASON_NO_COMPATIBLE_ARTIFACT, REASON_NO_SIGNATURE_MATCH, REASON_SELECTED,
                                             GraphDispatchSession, attach_graph_dispatch, detach_graph_dispatch)
from fastvideo.optimization.fx_capture import capture_export_invocation
from fastvideo.optimization.identity import (graph_identity, input_signatures, output_signatures)
from fastvideo.optimization.subgraph import (SubgraphRewriteError,
                                             rewrite_exported_subgraph)

WIDTH = 4
MARKER = 7.0

# A CPU fake kernel. It reads what it needs from the module it is handed, which
# is what lets one artifact serve every block in a stack.
FAKE_KERNEL = '''"""CPU fake kernel used by the dispatch tests."""
import torch


def fused_block(module, hidden):
    return torch.full_like(hidden, float(module.marker))
'''

RAISING_KERNEL = '''"""CPU fake kernel that fails at call time."""


def fused_block(module, hidden):
    raise RuntimeError("candidate exploded")
'''

OFFLOADED_KERNEL = '''"""CPU fake requiring hook-materialized parameters."""
import torch


def fused_block(module, hidden):
    if module.proj.weight.numel() == 0:
        raise RuntimeError("candidate observed offloaded parameters")
    return torch.full_like(hidden, float(module.marker))
'''

IMPORT_FAILURE_KERNEL = '''"""CPU fake kernel that fails at import time."""
raise RuntimeError("import side effect")


def fused_block(module, hidden):
    return hidden
'''

SUBGRAPH_KERNEL = '''"""CPU fake fused epilogue used by subgraph dispatch tests."""
import torch


def fused_subgraph(module, projected, hidden):
    module.subgraph_calls = getattr(module, "subgraph_calls", 0) + 1
    return torch.nn.functional.silu(projected) + hidden
'''

LIFTED_CONSTANT_KERNEL = '''"""CPU fake for an export-lifted tensor constant."""


def fused_subgraph(module, value, offset):
    module.subgraph_calls = getattr(module, "subgraph_calls", 0) + 1
    return value + offset
'''


class _Block(nn.Module):
    """One traceable block, standing in for a transformer block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.marker = MARKER

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(hidden)) + hidden


class _UntraceableBlock(nn.Module):
    """A block whose data-dependent branch defeats symbolic tracing."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.marker = MARKER

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if bool(hidden.sum() > 0):
            return self.proj(hidden)
        return self.proj(hidden) * 2


class _LiftedConstantBlock(nn.Module):
    """A plain tensor attribute becomes ``lifted_tensor_*`` under export."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)
        # Deliberately not a Parameter or registered buffer. Export owns the
        # opaque lifted name; the live module only owns ``offset``.
        self.offset = torch.tensor(0.25)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(hidden)) + hidden + self.offset


class _TopologicalGapBlock(nn.Module):
    """A selected producer and consumer separated by an unselected op."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        early = torch.neg(hidden)
        gap = torch.sin(early)
        return gap + early


class _StructuredInputBlock(nn.Module):
    """A block with a tuple input, mirroring rotary-frequency arguments."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # Registration deliberately differs from execution order. Export is
        # free to order get_attr nodes by first use, not registration order.
        self.offset = nn.Parameter(torch.full((width, ), 3.0))
        self.scale = nn.Parameter(torch.full((width, ), 2.0))

    def forward(
        self,
        hidden: torch.Tensor,
        frequencies: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        projected = hidden * self.scale
        rotated = projected * frequencies[0]
        return rotated + frequencies[1] + self.offset


class _ManagedBlock(_Block):
    """CPU fake exposing the FSDP2 materialization lifecycle."""

    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.unshard_calls = 0
        self.reshard_calls = 0

    def unshard(self, *, async_op: bool = False) -> None:
        assert async_op is False
        self.unshard_calls += 1

    def reshard(self) -> None:
        self.reshard_calls += 1


class _FakeLayerwiseOffloadHook(ForwardHook):
    """CPU fake for FastVideo's parameter swapping hook."""

    class _State:
        def __init__(self, module: nn.Module) -> None:
            self.module = module
            self.full_parameters = {
                name: parameter.data.clone()
                for name, parameter in module.named_parameters()
            }
            self.gpu_named_parameters: dict[str, torch.Tensor] = {}
            self.release()

        def wait_and_replace_params(self) -> None:
            for name, parameter in self.module.named_parameters():
                materialized = self.full_parameters[name]
                parameter.data = materialized
                self.gpu_named_parameters[name] = materialized

        def release(self) -> None:
            for parameter in self.module.parameters():
                parameter.data = torch.empty(
                    (0, ) * parameter.ndim,
                    dtype=parameter.dtype,
                )
            self.gpu_named_parameters.clear()

    def __init__(self, module: nn.Module) -> None:
        self.state = self._State(module)
        self.pre_calls = 0
        self.post_calls = 0

    @classmethod
    def name(cls) -> str:
        return "LayerwiseOffloadHook"

    def pre_forward(self, module, *args, **kwargs):
        self.pre_calls += 1
        self.state.wait_and_replace_params()
        return args, kwargs

    def post_forward(self, module, output):
        self.post_calls += 1
        self.state.release()
        return output


class _Transformer(nn.Module):
    """A repeated block stack: the structure dispatch keys on."""

    def __init__(self, width: int = WIDTH, depth: int = 2, block_cls=_Block) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([block_cls(width) for _ in range(depth)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _ManagedTransformer(_Transformer):
    """CPU fake whose parameter lifecycle is owned above its blocks."""

    def __init__(self, width: int = WIDTH, depth: int = 2) -> None:
        super().__init__(width=width, depth=depth)
        self.unshard_calls = 0
        self.reshard_calls = 0

    def unshard(self, *, async_op: bool = False) -> None:
        assert async_op is False
        self.unshard_calls += 1

    def reshard(self) -> None:
        self.reshard_calls += 1


def _pipeline_modules(block_cls=_Block) -> dict[str, nn.Module]:
    torch.manual_seed(0)
    return {"transformer": _Transformer(block_cls=block_cls).eval()}


def _runtime(**overrides) -> RuntimeProfile:
    profile = {
        "model_id": "fake/model",
        "model_revision": "main",
        "gpu_architecture": "cpu",
        "torch_version": torch.__version__,
        "cuda_version": None,
        "triton_version": None,
        "execution_mode": "inference",
        "distributed_mode": "single",
    }
    profile.update(overrides)
    return RuntimeProfile(**profile)


def _observed_identity(
    module: nn.Module,
    hidden: torch.Tensor,
    *,
    scope: str = "transformer.blocks",
    tracer: str = "symbolic",
):
    """Trace one call the way dispatch does, to learn the artifact identity."""
    with torch.no_grad():
        output = module(hidden)
    region = graph_identity(module, (hidden, ), {}, output, scope=scope, tracer=tracer)
    return (
        region["fingerprint"],
        input_signatures((hidden, ), {}),
        output_signatures(output),
    )


def _sections(fingerprint, inputs, outputs, **overrides) -> dict:
    sections = {
        "schema_version": 1,
        "artifact_id": "fake-fused-block",
        "created_at": "2026-07-31T00:00:00+00:00",
        "producer": {
            "name": "motionkernel",
            "version": "1.0.0"
        },
        "operation": {
            "name": "generated_blocks_fused",
            "graph_fingerprint": fingerprint,
            "parent_module": "transformer.blocks",
            "operations": ["aten::linear", "aten::silu", "aten::add"],
        },
        "signature": {
            "inputs": copy.deepcopy(inputs),
            "outputs": copy.deepcopy(outputs),
        },
        "entry_point": {
            "file": "kernel.py",
            "symbol": "fused_block"
        },
        "compatibility": {
            "model_id": "fake/model",
            "model_revision": ANY,
            "gpu_architectures": ["cpu"],
            "torch": {},
            "cuda": {},
            "triton": {},
            "execution_modes": ["inference"],
            "distributed_modes": ["single"],
        },
        "evidence": {
            "benchmark": {
                "harness": "motionkernel-bench",
                "device": "cpu",
                "samples": 20,
                "baseline_us": 100.0,
                "candidate_us": 50.0,
                "speedup": 2.0,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 1e-5,
                "rtol": 1e-5,
                "passed": True,
                "result_ref": "",
            },
            "generation": {
                "workload_id": "fake-workload",
                "steps": 2,
                "metric": "max_abs_latent_diff",
                "value": 0.0,
                "threshold": 1e-3,
                "passed": True,
                "baseline_ref": "",
                "candidate_ref": "",
            },
        },
        "promotion": {
            "decision": "promoted",
            "reason": "2x with full-generation parity",
            "decided_at": "2026-07-31T00:00:00+00:00",
            "campaign": {
                "campaign_id": "campaign-1",
                "source": "cpu-fake",
                "target_name": "blocks_fused",
            },
        },
    }
    sections.update(overrides)
    return sections


def _subgraph_sections(module: nn.Module, hidden: torch.Tensor) -> dict:
    with torch.no_grad():
        output = module(hidden)
    region, _ = capture_export_invocation(
        module,
        (hidden, ),
        {},
        output,
        scope="transformer.blocks",
    )
    ir = region["attributes"]["executable_ir"]
    metadata = {
        item["id"]: item["meta"]
        for section in ("inputs", "nodes")
        for item in ir[section]
        if "meta" in item
    }

    def signatures(refs):
        return [
            {
                "name": f"boundary_{index}",
                **copy.deepcopy(metadata[ref]),
            }
            for index, ref in enumerate(refs)
        ]

    sections = _sections(
        region["fingerprint"],
        signatures(("n0", "p2")),
        signatures(("n2", )),
    )
    sections["operation"] = {
        "name": "generated_blocks_epilogue",
        "graph_fingerprint": region["fingerprint"],
        "parent_module": "transformer.blocks",
        "operations": ["aten::silu", "aten::add"],
        "target_kind": "subgraph",
        "capture_mode": "export",
        "selected_node_ids": ["n1", "n2"],
        "boundary_refs": ["n0", "p2"],
        "output_node_ids": ["n2"],
    }
    sections["entry_point"]["symbol"] = "fused_subgraph"
    return sections


def _lifted_constant_sections(module: nn.Module, hidden: torch.Tensor) -> dict:
    with torch.no_grad():
        output = module(hidden)
    region, _ = capture_export_invocation(
        module,
        (hidden, ),
        {},
        output,
        scope="transformer.blocks",
    )
    ir = region["attributes"]["executable_ir"]
    metadata = {
        item["id"]: item["meta"]
        for section in ("inputs", "nodes")
        for item in ir[section]
        if "meta" in item
    }
    final = ir["nodes"][-1]
    boundary_refs = [item["ref"] for item in final["args"] if "ref" in item]

    def signatures(refs):
        return [
            {"name": f"boundary_{index}", **copy.deepcopy(metadata[ref])}
            for index, ref in enumerate(refs)
        ]

    sections = _sections(
        region["fingerprint"],
        signatures(boundary_refs),
        signatures((final["id"], )),
    )
    sections["operation"] = {
        "name": "generated_lifted_constant_epilogue",
        "graph_fingerprint": region["fingerprint"],
        "parent_module": "transformer.blocks",
        "operations": [final["target"]],
        "target_kind": "subgraph",
        "capture_mode": "export",
        "selected_node_ids": [final["id"]],
        "boundary_refs": boundary_refs,
        "output_node_ids": [final["id"]],
    }
    sections["entry_point"]["symbol"] = "fused_subgraph"
    return sections


def _write_bundle(root: Path, sections: dict, *, kernel_source: str = FAKE_KERNEL, name: str = "fused") -> Path:
    """Write a bundle the way the producer would, hashes included."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    kernel = directory / "kernel.py"
    kernel.write_text(kernel_source, encoding="utf-8")
    document = dict(sections)
    document["files"] = [{
        "path": "kernel.py",
        "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "bytes": kernel.stat().st_size,
    }]
    (directory / MANIFEST_FILENAME).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return directory


@pytest.fixture()
def hidden() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, WIDTH)


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    return directory


def _session(
    store: Path,
    *,
    tracer: str = "symbolic",
    validation: bool = False,
    **overrides,
) -> GraphDispatchSession:
    return GraphDispatchSession(
        ArtifactRegistry(store),
        _runtime(**overrides),
        tracer=tracer,
        validation=validation,
    )


def _decisions(session: GraphDispatchSession) -> list[dict]:
    return session.diagnostics()["decisions"]


def _reasons(session: GraphDispatchSession) -> list[str]:
    return [item["reason"] for item in _decisions(session)]


# -- zero-effect baseline -----------------------------------------------------


def test_no_artifact_directory_behaves_exactly_like_native(monkeypatch, hidden):
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR", "")
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    before = [type(block).forward for block in transformer.blocks]
    with torch.no_grad():
        expected = transformer(hidden)

    session = attach_graph_dispatch(modules)

    assert session is None
    # No instance attribute shadows the class method: the module graph is
    # untouched, so behavior is identical to a build without this feature.
    assert all("forward" not in vars(block) for block in transformer.blocks)
    assert [type(block).forward for block in transformer.blocks] == before
    with torch.no_grad():
        assert torch.equal(transformer(hidden), expected)


def test_empty_artifact_directory_stays_native(store, monkeypatch, hidden):
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR", str(store))
    modules = _pipeline_modules()

    session = attach_graph_dispatch(modules)

    assert session is None


def test_detach_restores_the_native_forward(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    with torch.no_grad():
        expected = transformer(hidden)
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)

    assert session.attach_modules(modules) == 2
    assert all("forward" in vars(block) for block in transformer.blocks)

    session.detach()

    assert all("forward" not in vars(block) for block in transformer.blocks)
    with torch.no_grad():
        assert torch.equal(transformer(hidden), expected)


# -- selection ----------------------------------------------------------------


def test_compatible_artifact_is_selected(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        first = transformer(hidden)
        second = transformer(hidden)

    session.detach()
    # One decision covers the whole stack, so only the very first block call
    # runs natively; every later block -- including the second block of the
    # first pass -- goes through the candidate, which fills the tensor with the
    # marker it reads off the module it was handed.
    assert torch.allclose(first, torch.full_like(first, MARKER))
    assert torch.allclose(second, torch.full_like(second, MARKER))
    assert _reasons(session) == [REASON_SELECTED]
    decision = _decisions(session)[0]
    assert decision["artifact_id"] == "fake-fused-block"
    assert decision["candidate_calls"] == 3
    assert decision["runtime_fallbacks"] == 0


def test_quarantined_artifact_is_admitted_only_for_explicit_validation(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    sections = _sections(fingerprint, inputs, outputs)
    sections["promotion"]["decision"] = "quarantined"
    sections["evidence"]["generation"]["passed"] = False
    _write_bundle(store, sections)

    production = _session(store)
    production.attach_modules(modules)
    with torch.no_grad():
        transformer(hidden)
    production.detach()
    assert _reasons(production) == [REASON_NO_COMPATIBLE_ARTIFACT]

    validation = _session(store, validation=True)
    validation.attach_modules(modules)
    with torch.no_grad():
        transformer(hidden)
    validation.detach()
    assert _reasons(validation) == [REASON_SELECTED]


def test_export_identity_traces_native_forward_without_dispatch_recursion(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(
        transformer.blocks[0], hidden, tracer="export"
    )
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store, tracer="export")
    session.attach_modules(modules)

    with torch.no_grad():
        first = transformer(hidden)
        second = transformer(hidden)

    session.detach()
    assert torch.allclose(first, torch.full_like(first, MARKER))
    assert torch.allclose(second, torch.full_like(second, MARKER))
    assert _reasons(session) == [REASON_SELECTED]


def test_export_subgraph_artifact_rewrites_each_live_block_without_copying_weights(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    with torch.no_grad():
        expected = transformer(hidden)
    sections = _subgraph_sections(transformer.blocks[0], hidden)
    _write_bundle(store, sections, kernel_source=SUBGRAPH_KERNEL)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        first = transformer(hidden)
        second = transformer(hidden)

    session.detach()
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert [block.subgraph_calls for block in transformer.blocks] == [1, 2]
    assert _reasons(session) == [REASON_SELECTED]
    assert _decisions(session)[0]["candidate_calls"] == 3


def test_candidate_dispatch_runs_inside_managed_parameter_lifecycle(store, hidden):
    modules = _pipeline_modules(_ManagedBlock)
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(
        transformer.blocks[0], hidden
    )
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    assert [block.unshard_calls for block in transformer.blocks] == [1, 2]
    assert [block.reshard_calls for block in transformer.blocks] == [1, 2]
    assert _reasons(session) == [REASON_SELECTED]


def test_candidate_dispatch_uses_nearest_managed_ancestor(store, hidden):
    transformer = _ManagedTransformer().eval()
    modules = {"transformer": transformer}
    fingerprint, inputs, outputs = _observed_identity(
        transformer.blocks[0], hidden
    )
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    assert transformer.unshard_calls == 3
    assert transformer.reshard_calls == 3
    assert _reasons(session) == [REASON_SELECTED]


def test_candidate_dispatch_runs_through_module_hook_lifecycle(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(
        transformer.blocks[0], hidden
    )
    _write_bundle(
        store,
        _sections(fingerprint, inputs, outputs),
        kernel_source=OFFLOADED_KERNEL,
    )
    hooks = []
    for block in transformer.blocks:
        manager = ModuleHookManager.get_from_or_default(block)
        hook = _FakeLayerwiseOffloadHook(block)
        manager.append_forward_hook(hook)
        hooks.append(hook)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    assert [hook.pre_calls for hook in hooks] == [2, 2]
    assert [hook.post_calls for hook in hooks] == [2, 2]
    assert all(
        parameter.numel() == 0
        for block in transformer.blocks
        for parameter in block.parameters()
    )
    assert _reasons(session) == [REASON_SELECTED]
    assert _decisions(session)[0]["candidate_calls"] == 3


def test_export_subgraph_binds_opaque_lifted_constants_from_export(store, hidden):
    modules = _pipeline_modules(_LiftedConstantBlock)
    transformer = modules["transformer"]
    with torch.no_grad():
        expected = transformer(hidden)
    sections = _lifted_constant_sections(transformer.blocks[0], hidden)
    _write_bundle(store, sections, kernel_source=LIFTED_CONSTANT_KERNEL)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)  # resolve both blocks
        actual = transformer(hidden)

    session.detach()
    torch.testing.assert_close(actual, expected)
    assert [block.subgraph_calls for block in transformer.blocks] == [1, 2]
    assert _reasons(session) == [REASON_SELECTED]


def test_export_subgraph_rejects_recipe_without_one_insertion_point(tmp_path, hidden):
    module = _TopologicalGapBlock().eval()
    with torch.no_grad():
        output = module(hidden)
    region, exported = capture_export_invocation(
        module,
        (hidden, ),
        {},
        output,
        scope="transformer.blocks",
    )
    ir = region["attributes"]["executable_ir"]
    assert [node["id"] for node in ir["nodes"]] == ["n0", "n1", "n2"]
    metadata = {
        item["id"]: item["meta"]
        for section in ("inputs", "nodes")
        for item in ir[section]
        if "meta" in item
    }

    def signatures(refs):
        return [
            {"name": f"boundary_{index}", **copy.deepcopy(metadata[ref])}
            for index, ref in enumerate(refs)
        ]

    sections = _sections(
        region["fingerprint"],
        signatures(("p0", "n1")),
        signatures(("n0", "n2")),
    )
    sections["operation"] = {
        "name": "invalid_scattered_recipe",
        "graph_fingerprint": region["fingerprint"],
        "parent_module": "transformer.blocks",
        "operations": ["aten::neg", "aten::add"],
        "target_kind": "subgraph",
        "capture_mode": "export",
        "selected_node_ids": ["n0", "n2"],
        "boundary_refs": ["p0", "n1"],
        "output_node_ids": ["n0", "n2"],
    }
    sections["entry_point"]["symbol"] = "fused_subgraph"
    sections["files"] = [
        {"path": "kernel.py", "sha256": "0" * 64, "bytes": 0}
    ]
    manifest = ArtifactManifest.from_dict(sections, directory=tmp_path)
    dispatch = rewrite_exported_subgraph(exported, manifest, lambda *_: None)

    with pytest.raises(
        SubgraphRewriteError,
        match="no valid topological insertion point",
    ):
        dispatch(module, hidden)


def test_export_subgraph_preserves_structured_input_calling_convention(
    tmp_path, hidden
):
    module = _StructuredInputBlock(WIDTH).eval()
    frequencies = (torch.full_like(hidden, 2.0), torch.full_like(hidden, 3.0))
    with torch.no_grad():
        expected = module(hidden, frequencies)
    region, exported = capture_export_invocation(
        module,
        (hidden, frequencies),
        {},
        expected,
        scope="transformer.blocks",
    )
    ir = region["attributes"]["executable_ir"]
    final = ir["nodes"][-1]
    boundary_refs = tuple(
        item["ref"] for item in final["args"] if set(item) == {"ref"}
    )
    metadata = {
        item["id"]: item["meta"]
        for section in ("inputs", "nodes")
        for item in ir[section]
        if "meta" in item
    }

    def signatures(refs):
        return [
            {"name": f"boundary_{index}", **copy.deepcopy(metadata[ref])}
            for index, ref in enumerate(refs)
        ]

    sections = _sections(
        region["fingerprint"],
        signatures(boundary_refs),
        signatures((final["id"], )),
    )
    sections["operation"] = {
        "name": "structured_input_epilogue",
        "graph_fingerprint": region["fingerprint"],
        "parent_module": "transformer.blocks",
        "operations": [final["target"]],
        "target_kind": "subgraph",
        "capture_mode": "export",
        "selected_node_ids": [final["id"]],
        "boundary_refs": list(boundary_refs),
        "output_node_ids": [final["id"]],
    }
    sections["entry_point"]["symbol"] = "fused_subgraph"
    sections["files"] = [
        {"path": "kernel.py", "sha256": "0" * 64, "bytes": 0}
    ]
    manifest = ArtifactManifest.from_dict(sections, directory=tmp_path)
    candidate = lambda _module, left, right: left + right
    dispatch = rewrite_exported_subgraph(exported, manifest, candidate)

    with torch.no_grad():
        actual = dispatch(module, hidden, frequencies)

    torch.testing.assert_close(actual, expected)


def test_fastest_compatible_artifact_wins(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs), name="slow")
    fast = _sections(fingerprint, inputs, outputs)
    fast["artifact_id"] = "fake-fused-block-fast"
    fast["evidence"]["benchmark"]["speedup"] = 4.0
    _write_bundle(store, fast, name="fast")
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    decision = _decisions(session)[0]
    assert decision["artifact_id"] == "fake-fused-block-fast"
    assert decision["rejections"] == ["fake-fused-block:not_selected"]


# -- rejections ---------------------------------------------------------------


def test_fingerprint_mismatch_falls_back_to_native(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    _, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections("0" * 32, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        expected = _pipeline_modules()["transformer"](hidden)
        transformer(hidden)
        actual = transformer(hidden)

    session.detach()
    assert torch.equal(actual, expected)
    assert _reasons(session) == [REASON_NO_COMPATIBLE_ARTIFACT]
    assert _decisions(session)[0]["rejections"] == ["fake-fused-block:fingerprint_mismatch"]


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda item: item.update({"dtype": "bfloat16"}), "dtype"),
        (lambda item: item.update({"shape": [8, WIDTH]}), "shape"),
        (lambda item: item.update({"device_type": "cuda"}), "device_type"),
    ],
)
def test_signature_mismatches_never_reach_a_trace(store, hidden, mutate, field):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    mutated = copy.deepcopy(inputs)
    mutate(mutated[0])
    _write_bundle(store, _sections(fingerprint, mutated, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        expected = _pipeline_modules()["transformer"](hidden)
        transformer(hidden)
        actual = transformer(hidden)

    session.detach()
    assert torch.equal(actual, expected)
    # The input pre-filter rejects these before any graph is traced.
    assert _reasons(session) == [REASON_NO_SIGNATURE_MATCH]


def test_output_signature_mismatch_falls_back_to_native(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    mutated = copy.deepcopy(outputs)
    mutated[0]["shape"] = [2, WIDTH * 2]
    mutated[0]["stride"] = [WIDTH * 2, 1]
    _write_bundle(store, _sections(fingerprint, inputs, mutated))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    assert _decisions(session)[0]["rejections"] == ["fake-fused-block:output_signature_mismatch"]


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({
            "gpu_architectures": ["sm90"]
        }, "gpu_architecture_mismatch"),
        ({
            "torch": {
                "min": "99.0.0"
            }
        }, "torch_version_unsupported"),
        ({
            "cuda": {
                "min": "12.0"
            }
        }, "cuda_version_unsupported"),
        ({
            "triton": {
                "min": "3.0.0"
            }
        }, "triton_version_unsupported"),
        ({
            "model_id": "other/model"
        }, "model_mismatch"),
        ({
            "execution_modes": ["training"]
        }, "execution_mode_unsupported"),
        ({
            "distributed_modes": ["tensor_parallel"]
        }, "distributed_mode_unsupported"),
    ],
)
def test_environment_mismatches_fall_back_to_native(store, hidden, override, expected):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    sections = _sections(fingerprint, inputs, outputs)
    sections["compatibility"].update(override)
    _write_bundle(store, sections)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        native = _pipeline_modules()["transformer"](hidden)
        transformer(hidden)
        actual = transformer(hidden)

    session.detach()
    assert torch.equal(actual, native)
    assert _decisions(session)[0]["rejections"] == [f"fake-fused-block:{expected}"]


def test_unpromoted_artifact_is_never_dispatched(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    sections = _sections(fingerprint, inputs, outputs)
    sections["promotion"]["decision"] = "quarantined"
    _write_bundle(store, sections)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)

    session.detach()
    assert _decisions(session)[0]["rejections"] == ["fake-fused-block:not_promoted"]


# -- tampering and load failures ----------------------------------------------


def test_tampered_kernel_is_rejected_before_import(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    directory = _write_bundle(store, _sections(fingerprint, inputs, outputs))
    # Swap the kernel for one that would be obvious if it ever executed.
    (directory / "kernel.py").write_text('raise SystemExit("payload executed")\n', encoding="utf-8")

    registry = ArtifactRegistry(store)

    assert registry.manifests == []
    assert len(registry.errors) == 1
    assert "does not match manifest" in registry.errors[0] or "bytes" in registry.errors[0]

    session = GraphDispatchSession(registry, _runtime())
    assert session.attach_modules(modules) == 0
    with torch.no_grad():
        assert torch.equal(transformer(hidden), _pipeline_modules()["transformer"](hidden))


def test_undeclared_file_is_rejected(store, hidden):
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    directory = _write_bundle(store, _sections(fingerprint, inputs, outputs))
    (directory / "extra.py").write_text("SECRET = 1\n", encoding="utf-8")

    registry = ArtifactRegistry(store)

    assert registry.manifests == []
    assert "undeclared file" in registry.errors[0]


def test_unknown_schema_version_is_rejected(store, hidden):
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    sections = _sections(fingerprint, inputs, outputs)
    sections["schema_version"] = 99
    _write_bundle(store, sections)

    registry = ArtifactRegistry(store)

    assert registry.manifests == []
    assert "unsupported version" in registry.errors[0]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda document: document["signature"]["inputs"][0].update(
                requires_grad="false"
            ),
            "requires_grad",
        ),
        (
            lambda document: document["evidence"]["benchmark"].update(
                speedup=float("nan")
            ),
            "finite non-negative",
        ),
        (
            lambda document: document["compatibility"].update(
                torch={"min": "3.0", "max_exclusive": "2.0"}
            ),
            "min must be lower",
        ),
    ],
)
def test_malformed_acted_on_manifest_fields_fail_closed(
    store, hidden, mutate, match
):
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(
        modules["transformer"].blocks[0], hidden
    )
    document = _sections(fingerprint, inputs, outputs)
    mutate(document)
    _write_bundle(store, document)

    registry = ArtifactRegistry(store)

    assert registry.manifests == []
    assert match in registry.errors[0]


def test_artifact_module_names_do_not_collide_after_id_sanitizing(store, hidden):
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(
        modules["transformer"].blocks[0], hidden
    )
    dashed = _sections(fingerprint, inputs, outputs, artifact_id="my-kernel-v1")
    underscored = _sections(
        fingerprint, inputs, outputs, artifact_id="my_kernel_v1"
    )
    dashed_dir = _write_bundle(store, dashed, name="dashed")
    underscored_dir = _write_bundle(store, underscored, name="underscored")

    dashed_candidate = load_entry_point(
        verify_bundle(dashed_dir), trusted_root=store
    )
    underscored_candidate = load_entry_point(
        verify_bundle(underscored_dir), trusted_root=store
    )

    assert dashed_candidate.__module__ != underscored_candidate.__module__


def test_failing_import_falls_back_to_native(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs), kernel_source=IMPORT_FAILURE_KERNEL)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        native = _pipeline_modules()["transformer"](hidden)
        transformer(hidden)
        actual = transformer(hidden)

    session.detach()
    assert torch.equal(actual, native)
    assert _reasons(session) == ["artifact_load_failed:ArtifactError"]


def test_bundle_outside_the_trusted_root_is_never_loaded(store, tmp_path, hidden):
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    outside = _write_bundle(tmp_path / "elsewhere", _sections(fingerprint, inputs, outputs))

    # The registry only ever looks inside its own root.
    assert ArtifactRegistry(store).manifests == []
    assert verify_bundle(outside).artifact_id == "fake-fused-block"


# -- runtime failure ----------------------------------------------------------


def test_candidate_exception_falls_back_to_native_output(store, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs), kernel_source=RAISING_KERNEL)
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        native = _pipeline_modules()["transformer"](hidden)
        transformer(hidden)
        actual = transformer(hidden)
        again = transformer(hidden)

    session.detach()
    assert torch.equal(actual, native)
    assert torch.equal(again, native)
    decision = _decisions(session)[0]
    assert decision["reason"] == "candidate_runtime_error:RuntimeError"
    assert decision["active"] is False
    # Demoted after the first failure, not retried on every later call.
    assert decision["runtime_fallbacks"] == 1


# -- tracing failures ---------------------------------------------------------


def test_untraceable_module_stays_native(store, hidden):
    modules = _pipeline_modules(block_cls=_UntraceableBlock)
    transformer = modules["transformer"]
    with torch.no_grad():
        reference = transformer(hidden)
    # Build a bundle whose input signature matches, so the pre-filter passes
    # and the tracer is genuinely attempted.
    inputs = input_signatures((hidden, ), {})
    outputs = output_signatures(reference)
    _write_bundle(store, _sections("1" * 32, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    with torch.no_grad():
        transformer(hidden)
        actual = transformer(hidden)

    session.detach()
    assert torch.equal(actual, reference)
    assert _reasons(session)[0].startswith("graph_identity_unavailable")


# -- diagnostics --------------------------------------------------------------


def test_diagnostics_are_metadata_only(store, tmp_path, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)
    with torch.no_grad():
        transformer(hidden)
        transformer(hidden)
    session.detach()

    output = session.write_diagnostics(tmp_path / "dispatch.json")

    assert output is not None
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dispatch"]["dispatch_schema_version"] == 1
    assert report["dispatch"]["registry"]["artifact_ids"] == ["fake-fused-block"]
    assert report["dispatch"]["reason_counts"] == {REASON_SELECTED: 1}
    # Nothing tensor-shaped or prompt-shaped may appear anywhere in the report.
    serialized = json.dumps(report).lower()
    for forbidden in ("prompt", "tensor_values", "weights", "password", "secret"):
        assert forbidden not in serialized


def test_attach_graph_dispatch_uses_the_configured_directory(store, monkeypatch, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR", str(store))
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID", "fake/model")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_REVISION", ANY)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_TRACER", "symbolic")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SCOPES", 64)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SHAPES", 8)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DISTRIBUTED_MODE", "")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS", "")

    session = attach_graph_dispatch(modules)

    assert session is not None
    try:
        with torch.no_grad():
            transformer(hidden)
            actual = transformer(hidden)
        assert torch.allclose(actual, torch.full_like(actual, MARKER))
    finally:
        detach_graph_dispatch(session)
    assert all("forward" not in vars(block) for block in transformer.blocks)


def test_training_mode_is_detected_and_rejects_inference_only_artifacts(store, monkeypatch, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    transformer.train()
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR", str(store))
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID", "fake/model")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_REVISION", ANY)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_TRACER", "symbolic")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SCOPES", 64)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SHAPES", 8)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DISTRIBUTED_MODE", "")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS", "")

    session = attach_graph_dispatch(modules)

    assert session is not None
    try:
        with torch.no_grad():
            transformer(hidden)
            actual = transformer(hidden)
        assert not torch.allclose(actual, torch.full_like(actual, MARKER))
        assert _decisions(session)[0]["rejections"] == ["fake-fused-block:execution_mode_unsupported"]
    finally:
        detach_graph_dispatch(session)


def test_nested_training_module_is_detected_fail_closed(store, monkeypatch, hidden):
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    transformer.eval()
    transformer.blocks[0].train()
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR", str(store))
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID", "fake/model")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_REVISION", ANY)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_TRACER", "symbolic")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SCOPES", 64)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SHAPES", 8)
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DISTRIBUTED_MODE", "")
    monkeypatch.setattr(envs, "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS", "")

    session = attach_graph_dispatch(modules)

    assert session is not None
    try:
        with torch.no_grad():
            transformer(hidden)
            transformer(hidden)
        assert _decisions(session)[0]["rejections"] == [
            "fake-fused-block:execution_mode_unsupported"
        ]
    finally:
        detach_graph_dispatch(session)
# -- bundle immutability under trusted loading ---------------------------------


def _simple_bundle(store: Path, hidden: torch.Tensor, name: str = "fused") -> Path:
    modules = _pipeline_modules()
    block = modules["transformer"].blocks[0]
    fingerprint, inputs, outputs = _observed_identity(block, hidden)
    return _write_bundle(store, _sections(fingerprint, inputs, outputs), name=name)


def test_load_entry_point_writes_no_bytecode_into_bundle(store, hidden):
    bundle = _simple_bundle(store, hidden)
    registry = ArtifactRegistry(store)
    assert registry.errors == []

    load_entry_point(registry.manifests[0], trusted_root=store)

    assert not list(bundle.rglob("*.pyc"))
    # The producer-side validator ignores nothing: a bundle that acquired a
    # bytecode cache during loading would fail its next verification, which is
    # exactly the finalize-stage failure this guards against.
    verify_bundle(bundle)


def test_verify_bundle_rejects_undeclared_bytecode_cache(store, hidden):
    bundle = _simple_bundle(store, hidden)
    cache = bundle / "__pycache__"
    cache.mkdir()
    (cache / "kernel.cpython-311.pyc").write_bytes(b"forged")

    with pytest.raises(ArtifactError, match="undeclared"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_symlinked_directory(store, hidden, tmp_path):
    bundle = _simple_bundle(store, hidden)
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.py").write_text("secret", encoding="utf-8")
    os.symlink(hidden_dir, bundle / "hidden_link")

    with pytest.raises(ArtifactError, match="undeclared"):
        verify_bundle(bundle)


def test_load_entry_point_rejects_manifest_changed_since_validation(store, hidden):
    bundle = _simple_bundle(store, hidden)
    registry = ArtifactRegistry(store)
    assert registry.errors == []
    document = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    document["evidence"]["benchmark"]["speedup"] = 99.0
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactError, match="changed since validation"):
        load_entry_point(registry.manifests[0], trusted_root=store)


# ---------------------------------------------------------------------------
# Per-artifact isolation and hot-path cost
#
# Run ltx-v1-overnight-20260801-r4-sol dispatched four artifacts into
# vae.decoder.up_blocks.6.res_blocks simultaneously, made 56 candidate calls
# with zero runtime fallbacks, and regressed end-to-end from 3.2818s to
# 3.9410s. Nothing in the evidence could say which artifact was responsible for
# the parity change or the latency, because all four were enabled at once.
# ---------------------------------------------------------------------------


def test_enabled_ids_admits_only_the_named_artifact(store, hidden):
    """One artifact under test, same directory, no restaging."""
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    for index in range(3):
        sections = _sections(fingerprint, inputs, outputs)
        sections["artifact_id"] = f"mk-artifact-{index}"
        _write_bundle(store, sections, name=f"bundle{index}")

    everything = ArtifactRegistry(store)
    assert len(everything.manifests) == 3
    assert everything.enabled_ids == ()
    assert everything.excluded_ids == ()

    isolated = ArtifactRegistry(store, enabled_ids=["mk-artifact-1"])
    assert [item.artifact_id for item in isolated.manifests] == ["mk-artifact-1"]
    assert isolated.excluded_ids == ("mk-artifact-0", "mk-artifact-2")
    assert isolated.errors == []


def test_enabled_ids_appear_in_diagnostics(store, hidden):
    """A trial's own report records what was under test and what was held back."""
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    for index in range(2):
        sections = _sections(fingerprint, inputs, outputs)
        sections["artifact_id"] = f"mk-artifact-{index}"
        _write_bundle(store, sections, name=f"bundle{index}")

    summary = ArtifactRegistry(store, enabled_ids=["mk-artifact-0"]).summary()
    assert summary["artifact_ids"] == ["mk-artifact-0"]
    assert summary["enabled_filter"] == ["mk-artifact-0"]
    assert summary["excluded_ids"] == ["mk-artifact-1"]


def test_requesting_an_absent_artifact_is_an_error_not_a_silent_empty_run(store, hidden):
    """Otherwise a typo in a trial's ID reads as 'this artifact changes nothing'."""
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    sections = _sections(fingerprint, inputs, outputs)
    sections["artifact_id"] = "mk-artifact-real"
    _write_bundle(store, sections, name="bundle0")

    registry = ArtifactRegistry(store, enabled_ids=["mk-artifact-typo"])
    assert registry.manifests == []
    assert any("mk-artifact-typo" in error for error in registry.errors)


def test_enabled_ids_still_verify_every_bundle(store, hidden):
    """Selection narrows what is used; it never skips integrity checks."""
    modules = _pipeline_modules()
    fingerprint, inputs, outputs = _observed_identity(modules["transformer"].blocks[0], hidden)
    good = _sections(fingerprint, inputs, outputs)
    good["artifact_id"] = "mk-good"
    _write_bundle(store, good, name="good")

    tampered = _sections(fingerprint, inputs, outputs)
    tampered["artifact_id"] = "mk-tampered"
    directory = _write_bundle(store, tampered, name="tampered")
    (directory / "kernel.py").write_text("raise RuntimeError('altered')", encoding="utf-8")

    registry = ArtifactRegistry(store, enabled_ids=["mk-good"])
    assert [item.artifact_id for item in registry.manifests] == ["mk-good"]
    assert registry.errors, "the altered bundle is still reported"


def test_placeholder_contract_is_derived_once_not_per_call():
    """The per-call path must not walk the graph to rediscover placeholders."""
    import operator

    from torch import fx

    from fastvideo.optimization.subgraph import (
        _placeholder_contract,
        _validate_runtime_inputs,
    )

    graph = fx.Graph()
    first = graph.placeholder("x")
    first.meta["val"] = torch.zeros(2, 3)
    second = graph.placeholder("y")
    second.meta["val"] = torch.zeros(4)
    for index in range(200):
        graph.call_function(operator.add, args=(first, index))
    graph.output(first)

    contract = _placeholder_contract(graph)
    assert contract == (("x", ((2, 3), "torch.float32")), ("y", ((4,), "torch.float32")))

    _validate_runtime_inputs(contract, [torch.zeros(2, 3), torch.zeros(4)])

    with pytest.raises(SubgraphRewriteError, match="metadata changed"):
        _validate_runtime_inputs(contract, [torch.zeros(2, 5), torch.zeros(4)])
    with pytest.raises(SubgraphRewriteError, match="input count differs"):
        _validate_runtime_inputs(contract, [torch.zeros(2, 3)])


def test_dispatch_does_not_snapshot_parameters_on_the_success_path(store, hidden, monkeypatch):
    """Three FSDP-lifecycle snapshots per call were charged to every dispatch."""
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    counted = {"count": 0}
    original = GraphDispatchSession._parameter_snapshot

    def counting(wrapper):
        counted["count"] += 1
        return original(wrapper)

    monkeypatch.setattr(
        GraphDispatchSession, "_parameter_snapshot", staticmethod(counting)
    )

    with torch.no_grad():
        for _ in range(5):
            transformer(hidden)

    dispatched = sum(item["candidate_calls"] for item in _decisions(session))
    assert dispatched > 0, "the artifact must actually be running"
    assert counted["count"] == 0, "diagnostics belong on the failure path"


# ---------------------------------------------------------------------------
# Rewritten-graph execution cost
#
# Shadow timing on transformer.model.transformer_blocks measured the eager
# replay at 11.57ms against the module's own forward at 8.18ms on identical
# inputs. The op histogram showed 621 call_function nodes per call -- the graph
# is decomposed, so the penalty is per-op dispatch, not arithmetic. That 3.39ms
# swamped the artifact's 124us saving and is what fails gate 5.
# ---------------------------------------------------------------------------


def _assertion_graph(with_user: bool):
    """A graph holding two export-style metadata assertions."""
    import operator as _operator

    from torch import fx

    import fastvideo.optimization.subgraph as sg

    def _assert_tensor_metadata(*args):  # stands in for the aten overload
        return None

    graph = fx.Graph()
    x = graph.placeholder("x")
    kept = graph.call_function(_operator.add, args=(x, 1))
    first = graph.call_function(_assert_tensor_metadata, args=(x,))
    graph.call_function(_assert_tensor_metadata, args=(kept,))
    graph.output(graph.call_function(_operator.add, args=(first, 1)) if with_user else kept)
    return graph, sg, str(_assert_tensor_metadata), _operator


def test_export_runtime_assertions_are_stripped(monkeypatch):
    """68 of the transformer graph's 621 ops per call were metadata assertions."""
    graph, sg, target, _operator = _assertion_graph(with_user=False)
    monkeypatch.setattr(sg, "_ASSERTION_TARGETS", frozenset({target}))

    assert sg._strip_runtime_assertions(graph) == 2
    remaining = [str(n.target) for n in graph.nodes if n.op == "call_function"]
    assert remaining == [str(_operator.add)]


def test_an_assertion_with_users_is_never_stripped(monkeypatch):
    """Only dead assertions go; nothing the graph depends on is removed."""
    graph, sg, target, _ = _assertion_graph(with_user=True)
    monkeypatch.setattr(sg, "_ASSERTION_TARGETS", frozenset({target}))

    # One assertion feeds the output and must survive; the other is dead.
    assert sg._strip_runtime_assertions(graph) == 1


def test_stripping_leaves_an_unrelated_graph_untouched():
    from fastvideo.optimization.subgraph import _strip_runtime_assertions

    graph, _, _, _ = _assertion_graph(with_user=False)
    before = len([n for n in graph.nodes if n.op == "call_function"])
    assert _strip_runtime_assertions(graph) == 0, "no aten assertions present"
    assert len([n for n in graph.nodes if n.op == "call_function"]) == before


def test_cuda_graph_runner_warms_up_before_capturing():
    """Capturing the first call would record one-time allocator work."""
    from fastvideo.optimization.subgraph import (
        CudaGraphUnavailable,
        _CudaGraphRunner,
        _CudaGraphScope,
    )

    runner = _CudaGraphRunner(runnable=None, scope=_CudaGraphScope())
    for _ in range(_CudaGraphRunner.WARMUP_ITERATIONS):
        with pytest.raises(CudaGraphUnavailable, match="warming up"):
            runner([torch.zeros(2)])


def test_cuda_graph_runner_declines_non_cuda_inputs():
    """It must decline rather than guess: declining only costs speed."""
    from fastvideo.optimization.subgraph import (
        CudaGraphUnavailable,
        _CudaGraphRunner,
        _CudaGraphScope,
    )

    runner = _CudaGraphRunner(runnable=None, scope=_CudaGraphScope())
    runner._warmups = _CudaGraphRunner.WARMUP_ITERATIONS
    with pytest.raises(CudaGraphUnavailable):
        runner([torch.zeros(2)])


def test_cuda_graph_runner_declines_non_tensor_inputs():
    from fastvideo.optimization.subgraph import (
        CudaGraphUnavailable,
        _CudaGraphRunner,
        _CudaGraphScope,
    )

    runner = _CudaGraphRunner(runnable=None, scope=_CudaGraphScope())
    runner._warmups = _CudaGraphRunner.WARMUP_ITERATIONS
    with pytest.raises(CudaGraphUnavailable, match="not a tensor"):
        runner([object()])


def test_cuda_graph_scope_shares_one_buffer_per_position_across_blocks():
    """48 blocks x 27 placeholders of per-block buffers would add ~19GB."""
    from fastvideo.optimization.subgraph import _CudaGraphScope

    scope = _CudaGraphScope()
    first = scope.buffer_for(0, torch.zeros(4, 8))
    second = scope.buffer_for(0, torch.zeros(4, 8))
    assert first is second, "every block reads the same static address"
    assert scope.buffer_for(1, torch.ones(3)) is not first


def test_cuda_graph_scope_rejects_a_layout_change_between_blocks():
    from fastvideo.optimization.subgraph import CudaGraphUnavailable, _CudaGraphScope

    scope = _CudaGraphScope()
    scope.buffer_for(0, torch.zeros(4, 8))
    with pytest.raises(CudaGraphUnavailable, match="changed layout"):
        scope.buffer_for(0, torch.zeros(4, 9))


def test_a_pinned_input_that_moves_declines_rather_than_reading_stale_memory():
    """The capture reads pinned inputs in place; a moved one must not replay.

    Reading whatever now occupies that address would silently produce wrong
    output, which is the one outcome this path must never have.
    """
    from fastvideo.optimization.subgraph import (
        CudaGraphUnavailable,
        _CudaGraphRunner,
        _CudaGraphScope,
    )

    runner = _CudaGraphRunner(runnable=None, scope=_CudaGraphScope())
    runner._graph = object()
    runner._arity = 1
    runner._pinned = {0: 123456}
    runner._moving = {}

    with pytest.raises(CudaGraphUnavailable, match="moved after capture"):
        runner([torch.zeros(2)])


def test_warmup_records_input_addresses_for_the_stability_decision():
    from fastvideo.optimization.subgraph import (
        CudaGraphUnavailable,
        _CudaGraphRunner,
        _CudaGraphScope,
    )

    runner = _CudaGraphRunner(runnable=None, scope=_CudaGraphScope())
    tensor = torch.zeros(4)
    for _ in range(_CudaGraphRunner.WARMUP_ITERATIONS):
        with pytest.raises(CudaGraphUnavailable, match="warming up"):
            runner([tensor])
    assert len(runner._observed) == _CudaGraphRunner.WARMUP_ITERATIONS
    assert all(seen == [tensor.data_ptr()] for seen in runner._observed)


def test_dispatch_falls_back_to_eager_when_capture_is_unavailable(store, hidden):
    """On CPU there is no CUDA graph; the artifact must still run, eagerly.

    The fake kernel returns a marker value, so a marker-filled result is proof
    the candidate ran rather than the native forward.
    """
    modules = _pipeline_modules()
    transformer = modules["transformer"]
    fingerprint, inputs, outputs = _observed_identity(transformer.blocks[0], hidden)
    _write_bundle(store, _sections(fingerprint, inputs, outputs))
    session = _session(store)
    session.attach_modules(modules)

    from fastvideo.optimization.subgraph import _CudaGraphRunner

    with torch.no_grad():
        for _ in range(_CudaGraphRunner.WARMUP_ITERATIONS + 3):
            observed = transformer(hidden)

    assert torch.equal(observed, torch.full_like(observed, MARKER))
    decisions = _decisions(session)
    assert sum(item["candidate_calls"] for item in decisions) > 0
    assert sum(item["runtime_fallbacks"] for item in decisions) == 0
    session.detach()
