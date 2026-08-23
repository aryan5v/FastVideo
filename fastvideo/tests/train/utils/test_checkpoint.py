# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.utils.checkpoint`.

Covers the pure-Python portions of the checkpoint manager: name
parsing, resume-path resolution, metadata round-trip, rolling-delete
cleanup, the ``_is_stateful`` predicate, and the ``maybe_save`` gating
logic. The inference staging path is covered with mocked DCP I/O; real
distributed collectives and CUDA RNG snapshots require a GPU runner.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import fastvideo.train.utils.inference_checkpoint as inference_checkpoint
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    _FullModelState,
    _find_latest_checkpoint,
    _is_stateful,
    _parse_step_from_dir,
    _resolve_resume_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint_dir(
    output_dir: Path,
    step: int,
    *,
    with_dcp: bool = True,
    with_metadata: bool = True,
) -> Path:
    """Create a fake ``checkpoint-<step>/dcp`` directory tree.

    ``dcp/.metadata`` is dcp.save's completion marker (written last);
    ``_find_latest_checkpoint`` requires it, so a complete fake checkpoint
    must include it. ``with_metadata=False`` fakes a crashed mid-write save.
    """
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if with_dcp:
        (ckpt_dir / "dcp").mkdir(exist_ok=True)
        if with_metadata:
            (ckpt_dir / "dcp" / ".metadata").touch()
    return ckpt_dir


def _make_manager(
    tmp_path: Path,
    *,
    save_steps: int = 0,
    save_inference_on_validation: bool = False,
    keep_last: int = 0,
    start_step: int = 0,
    raw_config: dict[str, Any] | None = None,
) -> CheckpointManager:
    """Build a minimal ``CheckpointManager`` for tests that don't touch DCP."""
    return CheckpointManager(
        method=None,
        dataloader=None,
        output_dir=str(tmp_path),
        config=CheckpointConfig(
            save_steps=save_steps,
            keep_last=keep_last,
            start_step=start_step,
            save_inference_on_validation=save_inference_on_validation,
        ),
        raw_config=raw_config,
    )


# ---------------------------------------------------------------------------
# A. _is_stateful predicate
# ---------------------------------------------------------------------------


class _Full:

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


class _MissingStateDict:

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


class _MissingLoad:

    def state_dict(self) -> dict[str, Any]:
        return {}


def test_is_stateful_true_for_full_object() -> None:
    assert _is_stateful(_Full()) is True


def test_is_stateful_false_when_missing_state_dict() -> None:
    assert _is_stateful(_MissingStateDict()) is False


def test_is_stateful_false_when_missing_load_state_dict() -> None:
    assert _is_stateful(_MissingLoad()) is False


def test_full_model_state_keeps_frozen_inference_parameters() -> None:
    module = torch.nn.Linear(3, 2)
    module.requires_grad_(False)

    state = _FullModelState(module).state_dict()

    assert set(state) == {"weight", "bias"}


def test_inference_checkpoint_role_is_explicit() -> None:
    student = SimpleNamespace(
        transformer=torch.nn.Linear(2, 2),
        _init_from="base/student",
    )
    fake_method = SimpleNamespace(_role_models={"student": student})

    modules = TrainingMethod.inference_checkpoint_modules(fake_method, "student")
    base_path = TrainingMethod.inference_checkpoint_base_model_path(fake_method, "student")

    assert modules == {"transformer": student.transformer}
    assert base_path == "base/student"


def test_unknown_inference_checkpoint_role_raises() -> None:
    fake_method = SimpleNamespace(_role_models={})

    with pytest.raises(ValueError, match="known roles"):
        TrainingMethod.inference_checkpoint_modules(fake_method, "ema")


# ---------------------------------------------------------------------------
# B. _parse_step_from_dir
# ---------------------------------------------------------------------------


def test_parse_step_valid(tmp_path: Path) -> None:
    assert _parse_step_from_dir(tmp_path / "checkpoint-100") == 100


def test_parse_step_zero(tmp_path: Path) -> None:
    assert _parse_step_from_dir(tmp_path / "checkpoint-0") == 0


def test_parse_step_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint directory"):
        _parse_step_from_dir(tmp_path / "not-a-checkpoint")


# ---------------------------------------------------------------------------
# C. _find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_returns_none_on_nonexistent_dir(tmp_path: Path) -> None:
    assert _find_latest_checkpoint(tmp_path / "missing") is None


def test_find_latest_returns_none_on_empty_dir(tmp_path: Path) -> None:
    assert _find_latest_checkpoint(tmp_path) is None


def test_find_latest_returns_largest_step(tmp_path: Path) -> None:
    _make_checkpoint_dir(tmp_path, 10)
    _make_checkpoint_dir(tmp_path, 200)
    _make_checkpoint_dir(tmp_path, 50)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-200"


def test_find_latest_skips_dirs_without_dcp_subdir(tmp_path: Path) -> None:
    # checkpoint-10 is "corrupted" — has no dcp/ subdir, must be skipped.
    _make_checkpoint_dir(tmp_path, 10, with_dcp=False)
    _make_checkpoint_dir(tmp_path, 5, with_dcp=True)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-5"


def test_find_latest_skips_incomplete_dcp_save(tmp_path: Path) -> None:
    # checkpoint-10 crashed mid-save — dcp/ exists but .metadata (written
    # last by dcp.save) does not; resuming from it would fail at boot.
    _make_checkpoint_dir(tmp_path, 10, with_metadata=False)
    _make_checkpoint_dir(tmp_path, 5)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-5"


def test_find_latest_skips_non_checkpoint_dirs(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "wandb").mkdir()
    (tmp_path / "some_file.txt").write_text("noise")
    _make_checkpoint_dir(tmp_path, 7)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-7"


# ---------------------------------------------------------------------------
# D. _resolve_resume_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_latest_with_no_checkpoints_returns_none(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    assert _resolve_resume_checkpoint("latest", output_dir=str(out)) is None


def test_resolve_latest_returns_latest_checkpoint(tmp_path: Path) -> None:
    _make_checkpoint_dir(tmp_path, 30)
    _make_checkpoint_dir(tmp_path, 10)
    resolved = _resolve_resume_checkpoint("latest", output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-30"


def test_resolve_explicit_checkpoint_dir(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 42)
    resolved = _resolve_resume_checkpoint(str(ckpt), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-42"


def test_resolve_dcp_subdir_returns_parent_checkpoint(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 42)
    dcp_path = ckpt / "dcp"
    resolved = _resolve_resume_checkpoint(str(dcp_path), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-42"


def test_resolve_output_dir_returns_latest(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    _make_checkpoint_dir(out, 100)
    _make_checkpoint_dir(out, 50)
    resolved = _resolve_resume_checkpoint(str(out), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-100"


def test_resolve_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_resume_checkpoint(str(tmp_path / "missing"), output_dir=str(tmp_path))


def test_resolve_checkpoint_without_dcp_raises(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 5, with_dcp=False)
    with pytest.raises(FileNotFoundError, match="dcp"):
        _resolve_resume_checkpoint(str(ckpt), output_dir=str(tmp_path))


def test_resolve_unknown_dir_raises(tmp_path: Path) -> None:
    """A dir that is neither a checkpoint nor an output_dir-with-checkpoints."""
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_resume_checkpoint(str(bogus), output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# E. metadata read/write
# ---------------------------------------------------------------------------


def test_write_metadata_roundtrip_with_step(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    mgr._write_metadata(ckpt_dir, step=7)
    loaded = CheckpointManager.load_metadata(ckpt_dir)
    assert loaded == {"step": 7}


def test_write_metadata_includes_raw_config(tmp_path: Path) -> None:
    raw = {
        "models": {
            "student": {
                "_target_": "X"
            }
        },
        "training": {
            "distributed": {
                "num_gpus": 4
            }
        },
    }
    mgr = _make_manager(tmp_path, raw_config=raw)
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    mgr._write_metadata(ckpt_dir, step=7)
    loaded = CheckpointManager.load_metadata(ckpt_dir)
    assert loaded["step"] == 7
    assert loaded["config"] == raw


def test_load_metadata_raises_on_missing_file(tmp_path: Path) -> None:
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    # No metadata.json written.
    with pytest.raises(FileNotFoundError, match="metadata"):
        CheckpointManager.load_metadata(ckpt_dir)


# ---------------------------------------------------------------------------
# F. _cleanup_old_checkpoints (rolling delete)
# ---------------------------------------------------------------------------


def test_cleanup_keep_last_zero_is_noop(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=0)
    for step in (1, 2, 3):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]


def test_cleanup_keeps_newest_when_over_limit(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=2)
    for step in (1, 5, 10, 50, 100):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-100", "checkpoint-50"]


def test_cleanup_no_op_when_under_limit(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=3)
    for step in (1, 2):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-1", "checkpoint-2"]


def test_cleanup_skips_non_checkpoint_dirs(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=1)
    for step in (1, 2, 3):
        _make_checkpoint_dir(tmp_path, step)
    (tmp_path / "logs").mkdir()
    (tmp_path / "wandb").mkdir()
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-3", "logs", "wandb"]


def test_cleanup_never_removes_inference_checkpoints(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=1)
    for step in (1, 2, 3):
        _make_checkpoint_dir(tmp_path, step)
        inference_dir = tmp_path / "inference" / f"checkpoint-{step}"
        inference_dir.mkdir(parents=True)
        (inference_dir / ".complete").touch()

    mgr._cleanup_old_checkpoints()

    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == ["checkpoint-3"]
    assert sorted(path.name for path in (tmp_path / "inference").iterdir()) == [
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
    ]


# ---------------------------------------------------------------------------
# G. maybe_save gating logic
# ---------------------------------------------------------------------------


def _record_save_calls(mgr: CheckpointManager) -> list[int]:
    """Replace ``mgr.save`` with a recorder that mimics the side effect
    of advancing ``_last_saved_step`` so dedup logic still works."""
    calls: list[int] = []

    def fake_save(step: int) -> None:
        calls.append(step)
        mgr._last_saved_step = step

    mgr.save = fake_save  # type: ignore[method-assign]
    return calls


def test_maybe_save_skipped_when_save_steps_is_zero(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=0)
    calls = _record_save_calls(mgr)
    mgr.maybe_save(step=10)
    mgr.maybe_save(step=100)
    assert calls == []


def test_maybe_save_skipped_when_step_not_on_interval(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    for step in (1, 5, 9, 11, 15):
        mgr.maybe_save(step=step)
    assert calls == []


def test_maybe_save_dedupes_on_repeated_call(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    mgr.maybe_save(step=20)
    mgr.maybe_save(step=20)
    mgr.maybe_save(step=20)
    assert calls == [20]


def test_maybe_save_triggers_on_each_interval(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    for step in range(1, 41):
        mgr.maybe_save(step=step)
    assert calls == [10, 20, 30, 40]


def _record_both_save_calls(
    mgr: CheckpointManager,
) -> tuple[list[int], list[int]]:
    training_calls: list[int] = []
    inference_calls: list[int] = []

    def fake_training_save(step: int) -> None:
        training_calls.append(step)
        mgr._last_saved_step = step

    def fake_inference_save(step: int) -> None:
        inference_calls.append(step)
        mgr._last_inference_saved_step = step

    mgr.save = fake_training_save  # type: ignore[method-assign]
    mgr.save_inference = fake_inference_save  # type: ignore[method-assign]
    return training_calls, inference_calls


def test_inference_save_tracks_validation_events_and_ignores_start_gate(tmp_path: Path) -> None:
    mgr = _make_manager(
        tmp_path,
        save_steps=10,
        save_inference_on_validation=True,
        start_step=12,
    )
    training_calls, inference_calls = _record_both_save_calls(mgr)

    mgr.maybe_save_inference(0, validation_scheduled=True)
    mgr.maybe_save_inference(4, validation_scheduled=False)
    mgr.maybe_save_inference(10, validation_scheduled=True)
    mgr.maybe_save_inference(10, validation_scheduled=True)
    for step in range(1, 21):
        mgr.maybe_save(step)

    assert training_calls == [20]
    assert inference_calls == [0, 10]


def test_disabled_validation_inference_checkpointing_is_no_op(tmp_path: Path) -> None:
    mgr = _make_manager(
        tmp_path,
        save_inference_on_validation=False,
    )
    _, inference_calls = _record_both_save_calls(mgr)
    mgr.maybe_save_inference(0, validation_scheduled=True)

    assert inference_calls == []


def test_save_final_dedupes_training_checkpoint(tmp_path: Path) -> None:
    mgr = _make_manager(
        tmp_path,
        save_steps=10,
        save_inference_on_validation=True,
    )
    training_calls, inference_calls = _record_both_save_calls(mgr)

    mgr.maybe_save_inference(20, validation_scheduled=True)
    mgr.maybe_save(20)
    mgr.save_final(20)

    assert training_calls == [20]
    assert inference_calls == [20]


def test_save_final_does_not_create_off_validation_inference_product(tmp_path: Path) -> None:
    mgr = _make_manager(
        tmp_path,
        save_steps=10,
        save_inference_on_validation=True,
    )
    training_calls, inference_calls = _record_both_save_calls(mgr)

    mgr.save_final(17)

    assert training_calls == [17]
    assert inference_calls == []


def test_save_inference_stages_full_state_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = torch.nn.Linear(2, 2)
    module.requires_grad_(False)
    base_model = tmp_path / "base"
    base_model.mkdir()

    class _Method:
        cuda_generator = None

        def inference_checkpoint_modules(self, role: str) -> dict[str, torch.nn.Module]:
            assert role == "student"
            return {"transformer": module}

        def inference_checkpoint_base_model_path(self, role: str) -> str:
            assert role == "student"
            return str(base_model)

    manager = CheckpointManager(
        method=_Method(),
        dataloader=None,
        output_dir=str(tmp_path / "run"),
        config=CheckpointConfig(
            save_steps=0,
            keep_last=0,
            save_inference_on_validation=True,
        ),
    )
    staged_states: list[dict[str, Any]] = []

    def fake_dcp_save(states: dict[str, Any], *, checkpoint_id: str) -> None:
        staged_states.append(states)
        dcp_dir = Path(checkpoint_id)
        dcp_dir.mkdir(parents=True, exist_ok=True)
        (dcp_dir / ".metadata").touch()

    def fake_export(**kwargs: Any) -> Path:
        target = Path(kwargs["output_dir"])
        target.mkdir(parents=True)
        (target / ".complete").touch()
        return target

    monkeypatch.setattr("fastvideo.train.utils.checkpoint.dcp.save", fake_dcp_save)
    monkeypatch.setattr(inference_checkpoint, "export_inference_checkpoint", fake_export)
    monkeypatch.setattr(
        inference_checkpoint,
        "validate_complete_inference_checkpoint",
        lambda path, *, step: path if (path / ".complete").is_file() else None,
    )

    manager.save_inference(10)
    manager.save_inference(10)

    assert len(staged_states) == 1
    state = staged_states[0]
    assert set(state) == {"roles.student.transformer"}
    assert isinstance(state["roles.student.transformer"], _FullModelState)
    assert not (tmp_path / "run" / ".inference-staging").exists()
    assert (tmp_path / "run" / "inference" / "checkpoint-10" / ".complete").is_file()
