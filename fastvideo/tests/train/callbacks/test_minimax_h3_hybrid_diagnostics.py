# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniMax H3 hybrid health diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.models.dits.minimax_h3_hybrid.linear import BidirectionalLinearBranch
from fastvideo.train.callbacks.minimax_h3_hybrid_diagnostics import MiniMaxH3HybridDiagnosticsCallback


class _Tracker:

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int]] = []

    def log(self, metrics: dict[str, object], iteration: int) -> None:
        self.calls.append((metrics, iteration))


def _method():
    transformer = torch.nn.Module()
    transformer.linear_attention = BidirectionalLinearBranch(8, 2, 4, short_conv_targets=(), enable_text_state=False)
    transformer.requires_grad_(True)
    tracker = _Tracker()
    return SimpleNamespace(student=SimpleNamespace(transformer=transformer), tracker=tracker)


def test_hybrid_diagnostics_logs_forward_health_and_gradient_families() -> None:
    method = _method()
    branch = method.student.transformer.linear_attention
    branch.latest_diagnostics = {"gamma_median": torch.tensor(0.125)}
    branch.output_gate.latest_diagnostics = {"gate_mean": torch.tensor(0.5)}
    branch.write_log_scale.grad = torch.ones_like(branch.write_log_scale)
    branch.beta_proj.weight.grad = torch.ones_like(branch.beta_proj.weight)
    callback = MiniMaxH3HybridDiagnosticsCallback(every_n_steps=10)

    callback.on_before_optimizer_step(method, iteration=10)
    callback.on_training_step_end(method, {}, iteration=10)

    assert len(method.tracker.calls) == 2
    gradient_metrics, _ = method.tracker.calls[0]
    forward_metrics, _ = method.tracker.calls[1]
    assert gradient_metrics["hybrid_grad_norm/write_log_scale"] > 0
    assert gradient_metrics["hybrid_grad_norm/write_logits"] > 0
    assert "hybrid_health/linear_attention/gamma_median" in forward_metrics
    assert "hybrid_health/linear_attention.output_gate/gate_mean" in forward_metrics


def test_hybrid_diagnostics_skips_unselected_steps() -> None:
    method = _method()
    callback = MiniMaxH3HybridDiagnosticsCallback(every_n_steps=10)

    callback.on_before_optimizer_step(method, iteration=3)
    callback.on_training_step_end(method, {}, iteration=3)

    assert method.tracker.calls == []
