# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax H3 hybrid gradient-liveness guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.callbacks.minimax_h3_hybrid_liveness import (
    MiniMaxH3HybridGradientLivenessCallback,
)


class _Transformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_out_linear = torch.nn.Linear(2, 2, bias=False)
        self.linear_attention = torch.nn.Module()
        self.linear_attention.short_conv = torch.nn.Linear(2, 2, bias=False)


def _method(transformer: torch.nn.Module):
    return SimpleNamespace(student=SimpleNamespace(transformer=transformer), tracker=None)


def test_liveness_guard_accepts_readout_then_internal_gradients() -> None:
    transformer = _Transformer()
    callback = MiniMaxH3HybridGradientLivenessCallback()
    transformer.to_out_linear.weight.grad = torch.ones_like(transformer.to_out_linear.weight)
    callback.on_before_optimizer_step(_method(transformer), iteration=1)
    transformer.linear_attention.short_conv.weight.grad = torch.ones_like(
        transformer.linear_attention.short_conv.weight)
    callback.on_before_optimizer_step(_method(transformer), iteration=2)


def test_liveness_guard_rejects_dead_readout() -> None:
    transformer = _Transformer()
    callback = MiniMaxH3HybridGradientLivenessCallback()
    transformer.to_out_linear.weight.grad = torch.zeros_like(transformer.to_out_linear.weight)
    with pytest.raises(RuntimeError, match="all 1 readout tensors"):
        callback.on_before_optimizer_step(_method(transformer), iteration=1)


def test_liveness_guard_ignores_unchecked_steps() -> None:
    callback = MiniMaxH3HybridGradientLivenessCallback()
    callback.on_before_optimizer_step(_method(_Transformer()), iteration=3)
