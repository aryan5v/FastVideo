# SPDX-License-Identifier: Apache-2.0
"""Contracts for Wan's AutoKernel-derived normalization fast path."""

from __future__ import annotations

import torch

from fastvideo.layers.layernorm import ScaleResidualLayerNormScaleShift


def _layer(hidden: int) -> ScaleResidualLayerNormScaleShift:
    layer = ScaleResidualLayerNormScaleShift(
        hidden,
        norm_type="layer",
        eps=1e-6,
        elementwise_affine=True,
        dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    torch.manual_seed(3)
    layer.norm.weight.data.normal_()
    layer.norm.bias.data.normal_()
    return layer


def test_wan_native_fast_path_matches_existing_composition(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "0")
    layer = _layer(hidden=7)
    residual = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(2, 1, 7, dtype=torch.float32)
    zero = torch.tensor([0])

    expected = layer(residual, x, gate, zero, zero)
    actual = layer.forward_wan_self_attention(residual, x, gate)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=0, atol=0
        )


def test_wan_fusion_eligibility_is_inference_cuda_only(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "1")
    layer = _layer(hidden=7)
    residual = torch.randn(1, 2, 7, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 1, 7, dtype=torch.float32)

    assert not layer._can_use_wan_fusion(residual, x, gate)


def test_wan_framewise_gate_retains_native_path(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "1")
    layer = _layer(hidden=7)
    residual = torch.randn(1, 6, 7, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 7, dtype=torch.float32)

    assert not layer._can_use_wan_fusion(residual, x, gate)
    normalized, updated = layer.forward_wan_self_attention(
        residual, x, gate
    )
    expected_updated = residual + (
        x.unflatten(1, (2, 3)) * gate
    ).flatten(1, 2)
    torch.testing.assert_close(updated, expected_updated, rtol=0, atol=0)
    torch.testing.assert_close(
        normalized, layer.norm(expected_updated), rtol=0, atol=0
    )
