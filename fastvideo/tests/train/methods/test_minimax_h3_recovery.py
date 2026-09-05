# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for MiniMax-H3 pruning recovery losses."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    _capture_hidden_summaries,
    _deployment_sigmas,
    _euler_update,
    _hidden_summary,
    _normalized_mse,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_normalized_mse_equalizes_target_scale() -> None:
    low_target = torch.full((2, 3), 0.5)
    high_target = torch.full((2, 3), 4.0)
    low_prediction = low_target * 2.0
    high_prediction = high_target * 2.0

    low_raw, low_energy, low_normalized = _normalized_mse(low_prediction, low_target, energy_floor=1.0e-3)
    high_raw, high_energy, high_normalized = _normalized_mse(high_prediction, high_target, energy_floor=1.0e-3)

    assert high_raw > low_raw
    assert high_energy > low_energy
    torch.testing.assert_close(low_normalized, torch.tensor(1.0))
    torch.testing.assert_close(high_normalized, torch.tensor(1.0))


def test_hidden_summary_preserves_gradient_with_bounded_shape() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_()

    summary = _hidden_summary(hidden)
    summary.sum().backward()

    assert summary.shape == (1, 12)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_hidden_capture_selects_only_requested_blocks() -> None:

    class TinyModel:

        def __init__(self) -> None:
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False) for _ in range(3)])

    model = TinyModel()
    value = torch.ones(1, 2, 4)
    with _capture_hidden_summaries(model, (0, 2)) as captures:  # type: ignore[arg-type]
        for block in model.transformer.transformer_blocks:
            value = block(value)

    assert set(captures) == {0, 2}
    assert all(summary.shape == (1, 8) for summary in captures.values())


def test_h3_four_call_rollout_reaches_constant_flow_endpoint() -> None:
    for shift in (12.0, 3.0):
        sigmas = _deployment_sigmas(5, shift, torch.device("cpu"))
        state = torch.tensor([2.0])
        flow = torch.tensor([1.0])
        for interval in range(4):
            state = _euler_update(state, flow, sigmas[interval], sigmas[interval + 1])

        torch.testing.assert_close(state, torch.tensor([1.0]))


def test_h3_four_call_schedule_rejects_non_deployment_grid() -> None:
    with pytest.raises(ValueError, match="five-point/four-call"):
        _deployment_sigmas(4, 12.0, torch.device("cpu"))


def test_four_call_rescue_config_locks_fp32_branch_matching_and_gate_retention() -> None:
    config = yaml.safe_load(
        (_REPO_ROOT / "examples/train/configs/fasth3_14b_four_call_rescue.yaml").read_text())

    assert config["method"]["_target_"].endswith("MiniMaxH3FourCallRecoveryMethod")
    assert config["method"]["match_teacher_backend"] is True
    assert config["method"]["require_fp32_master"] is True
    assert config["method"]["deployment_grid_points"] == 5
    assert config["training"]["dit_precision"] == "fp32"
    assert config["training"]["data"]["num_frames"] == 124
    assert config["training"]["data"]["num_latent_t"] == 37
    assert config["training"]["checkpoint"]["training_state_checkpointing_steps"] == 25
    assert config["training"]["checkpoint"]["preserve_every_steps"] == 25
    assert config["training"]["checkpoint"]["preserve_steps"] == [25, 50, 75, 100, 200]


def test_four_call_rescue_launcher_stops_at_quality_gates() -> None:
    launcher = (_REPO_ROOT / "scripts/fasth3_sprint/slurm_h3_four_call_rescue.sbatch").read_text()

    assert '25) RESUME=""' in launcher
    assert '50|75|100|200) RESUME="latest"' in launcher
    assert "TARGET_STEPS must stop at a 25-step rescue gate" in launcher
    assert "validated {len(records)} aligned rescue records" in launcher
    assert '--training.data.data_path "[${DATA_ROOT}, ${SYNTH_DATA_ROOT}]"' in launcher


def test_recovery_config_locks_teacher_losses_and_checkpoint_retention() -> None:
    config = yaml.safe_load((_REPO_ROOT / "examples/train/configs/fasth3_14b_recovery.yaml").read_text())

    assert config["models"]["teacher"]["attention_backend"] == "TORCH_SDPA"
    assert config["models"]["teacher"]["trainable"] is False
    assert config["method"]["teacher_velocity_weight"] == 1.0
    assert config["method"]["feature_weight"] == 0.01
    assert config["training"]["checkpoint"]["training_state_checkpointing_steps"] == 100
    assert config["training"]["checkpoint"]["preserve_every_steps"] == 100
    assert config["callbacks"]["ema"]["decay"] == 0.9999


def test_recovery_launcher_requires_method_gate_before_long_run() -> None:
    launcher = (_REPO_ROOT / "scripts/fasth3_sprint/slurm_h3_recovery.sbatch").read_text()

    assert 'RUN_MODE="${RUN_MODE:-long}"' in launcher
    assert 'finite recovery TARGET_STEPS must be 1 or 2' in launcher
    assert 'scale_gate recovery TARGET_STEPS must be 1 or 2' in launcher
    assert 'Finite FastH3 recovery gate requires exactly 4 GPUs' in launcher
    assert 'RECOVERY_GATE_ROOT="${SPRINT_ROOT}/runs/h18-recovery-gates/${SOURCE_KIND}-activation"' in launcher
    assert 'test -s "${RECOVERY_GATE_ROOT}/checkpoint-2/metadata.json"' in launcher
    assert 'Long FastH3 recovery requires 12 through 32 GPUs' in launcher
    assert '--training.checkpoint.training_state_checkpointing_steps 1' in launcher
