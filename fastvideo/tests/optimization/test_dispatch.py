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
from pathlib import Path

import pytest
import torch
from torch import nn

from fastvideo import envs
from fastvideo.optimization.artifact import (ANY, MANIFEST_FILENAME, ArtifactRegistry, RuntimeProfile, load_entry_point,
                                             verify_bundle)
from fastvideo.optimization.dispatch import (REASON_NO_COMPATIBLE_ARTIFACT, REASON_NO_SIGNATURE_MATCH, REASON_SELECTED,
                                             GraphDispatchSession, attach_graph_dispatch, detach_graph_dispatch)
from fastvideo.optimization.identity import (graph_identity, input_signatures, output_signatures)

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

IMPORT_FAILURE_KERNEL = '''"""CPU fake kernel that fails at import time."""
raise RuntimeError("import side effect")


def fused_block(module, hidden):
    return hidden
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


class _Transformer(nn.Module):
    """A repeated block stack: the structure dispatch keys on."""

    def __init__(self, width: int = WIDTH, depth: int = 2, block_cls=_Block) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([block_cls(width) for _ in range(depth)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


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


def _session(store: Path, *, tracer: str = "symbolic", **overrides) -> GraphDispatchSession:
    return GraphDispatchSession(
        ArtifactRegistry(store),
        _runtime(**overrides),
        tracer=tracer,
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
