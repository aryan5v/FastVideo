# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for MiniMax-H3 pruning recovery losses."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    _capture_hidden_summaries,
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


def test_recovery_config_locks_teacher_losses_and_checkpoint_retention() -> None:
    config = yaml.safe_load((_REPO_ROOT / "examples/train/configs/fasth3_14b_recovery.yaml").read_text())

    assert config["models"]["teacher"]["attention_backend"] == "TORCH_SDPA"
    assert config["models"]["teacher"]["trainable"] is False
    assert config["method"]["teacher_velocity_weight"] == 1.0
    assert config["method"]["feature_weight"] == 0.01
    assert config["training"]["checkpoint"]["training_state_checkpointing_steps"] == 100
    assert config["training"]["checkpoint"]["preserve_every_steps"] == 100
    assert config["callbacks"]["ema"]["decay"] == 0.9999
