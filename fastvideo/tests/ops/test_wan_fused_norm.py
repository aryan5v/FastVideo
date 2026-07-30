# SPDX-License-Identifier: Apache-2.0
"""Contracts for Wan's AutoKernel-derived normalization fast path."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA"
)
def test_wan_triton_fusion_matches_module_boundary(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "1")
    layer = _layer(hidden=1537).cuda()
    residual = torch.randn(
        2, 17, 1537, device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn_like(residual)
    gate = torch.randn(2, 1, 1537, device="cuda", dtype=torch.float32)

    expected_updated_fp32 = residual.float() + x.float() * gate
    expected_normalized = F.layer_norm(
        expected_updated_fp32,
        (1537,),
        layer.norm.weight,
        layer.norm.bias,
        layer.norm.eps,
    ).to(residual.dtype)

    with torch.inference_mode():
        normalized, updated = layer.forward_wan_self_attention(
            residual, x, gate
        )

    torch.testing.assert_close(
        normalized, expected_normalized, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        updated, expected_updated_fp32.to(residual.dtype), atol=0, rtol=0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA"
)
def test_wan_triton_fusion_rejects_noncontiguous_inputs():
    from fastvideo.layers.wan_fusions import (
        wan_gated_residual_layer_norm, )

    residual = torch.randn(1, 2, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 16, device="cuda")[:, ::2]
    weight = torch.randn(8, device="cuda")
    bias = torch.randn(8, device="cuda")

    with pytest.raises(ValueError, match="contiguous"):
        wan_gated_residual_layer_norm(
            residual, x, gate, weight, bias, 1e-6
        )
