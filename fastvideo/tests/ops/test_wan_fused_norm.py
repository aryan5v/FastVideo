# SPDX-License-Identifier: Apache-2.0
"""Contracts for Wan's AutoKernel-derived normalization fast path."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from fastvideo.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.optimization import optimization_workload


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

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


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
    normalized, updated = layer.forward_wan_self_attention(residual, x, gate)
    expected_updated = residual + (x.unflatten(1, (2, 3)) * gate).flatten(1, 2)
    torch.testing.assert_close(updated, expected_updated, rtol=0, atol=0)
    torch.testing.assert_close(normalized, layer.norm(expected_updated), rtol=0, atol=0)


def test_wan_modulated_norm_native_path_matches_composition(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "0")
    norm = FP32LayerNorm(7, elementwise_affine=False, eps=1e-6)
    inputs = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 7, dtype=torch.float32)
    shift = torch.randn_like(scale)

    expected = (norm(inputs.float()) * (1.0 + scale) + shift).to(inputs.dtype)
    actual = norm.forward_wan_modulated(inputs, scale, shift)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wan_mlp_residual_native_path_matches_composition(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "0")
    layer = ScaleResidual()
    residual = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(2, 1, 7, dtype=torch.float32)
    expected = residual + x * gate
    actual = layer.forward_wan_mlp(residual, x, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wan_capture_emits_all_three_targets(monkeypatch, tmp_path):
    output = tmp_path / "wan-campaign.json"
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_CAPTURE", str(output))
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "0")
    hidden = 7
    residual = torch.randn(1, 3, hidden, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 1, hidden, dtype=torch.float32)
    scale = torch.randn_like(gate)
    shift = torch.randn_like(gate)
    affine = _layer(hidden)
    norm = FP32LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
    mlp_residual = ScaleResidual()

    with optimization_workload(workload_id="wan", model_id="Wan2.1"):
        norm.forward_wan_modulated(residual, scale, shift)
        affine.forward_wan_self_attention(residual, x, gate)
        mlp_residual.forward_wan_mlp(residual, x, gate)

    campaign = json.loads(output.read_text(encoding="utf-8"))
    assert {target["operation"]
            for target in campaign["targets"]} == {
                "wan_modulated_layer_norm",
                "wan_gated_residual_norm",
                "wan_gated_residual",
            }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA")
def test_wan_triton_fusion_matches_module_boundary(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSED_NORM", "1")
    layer = _layer(hidden=1537).cuda()
    residual = torch.randn(2, 17, 1537, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(2, 1, 1537, device="cuda", dtype=torch.float32)

    expected_updated_fp32 = residual.float() + x.float() * gate
    expected_normalized = F.layer_norm(
        expected_updated_fp32,
        (1537, ),
        layer.norm.weight,
        layer.norm.bias,
        layer.norm.eps,
    ).to(residual.dtype)

    with torch.inference_mode():
        normalized, updated = layer.forward_wan_self_attention(residual, x, gate)

    torch.testing.assert_close(normalized, expected_normalized, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(updated, expected_updated_fp32.to(residual.dtype), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA")
def test_wan_modulated_norm_triton_matches_module_boundary(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSIONS", "1")
    norm = FP32LayerNorm(1537, elementwise_affine=False, eps=1e-6).cuda()
    inputs = torch.randn(2, 17, 1537, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 1537, device="cuda", dtype=torch.float32)
    shift = torch.randn_like(scale)
    expected = (norm(inputs.float()) * (1.0 + scale) + shift).to(inputs.dtype)

    with torch.inference_mode():
        actual = norm.forward_wan_modulated(inputs, scale, shift)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA")
def test_wan_mlp_residual_triton_matches_module_boundary(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_WAN_FUSIONS", "1")
    layer = ScaleResidual()
    residual = torch.randn(2, 17, 1537, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(2, 1, 1537, device="cuda", dtype=torch.float32)
    expected = (residual.float() + x.float() * gate).to(residual.dtype)

    with torch.inference_mode():
        actual = layer.forward_wan_mlp(residual, x, gate)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan Triton fusion requires CUDA")
def test_wan_triton_fusion_rejects_noncontiguous_inputs():
    from fastvideo.layers.wan_fusions import (
        wan_gated_residual_layer_norm, )

    residual = torch.randn(1, 2, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 16, device="cuda")[:, ::2]
    weight = torch.randn(8, device="cuda")
    bias = torch.randn(8, device="cuda")

    with pytest.raises(ValueError, match="contiguous"):
        wan_gated_residual_layer_norm(residual, x, gate, weight, bias, 1e-6)
