# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MiniMax-H3 decoder linears: conversion contract, rotation exactness, train/deploy parity."""
import pytest
import torch
import torch.nn as nn

from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (
    NVFP4DecoderLinear,
    convert_decoder_to_nvfp4,
    nvfp4_decoder_linear_names,
)
from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoViTDecoder3d
from fastvideo.models.vaes.minimax_h3_int8_convrot import rotate_activation


def _tiny_decoder(num_layers: int = 3) -> MiniMaxH3VideoViTDecoder3d:
    torch.manual_seed(0)
    return MiniMaxH3VideoViTDecoder3d(in_channels=24,
                                      out_channels=3,
                                      patch_size=2,
                                      patch_size_t=1,
                                      num_layers=num_layers,
                                      num_attention_heads=4,
                                      attention_head_dim=64,
                                      num_register_tokens=4,
                                      ffn_mult=4,
                                      rope_theta=100.0,
                                      rope_dim_ratio=0.75,
                                      norm_eps=1e-5)


def test_selects_only_block_linears():
    decoder = _tiny_decoder()
    names = nvfp4_decoder_linear_names(decoder)
    assert len(names) == 3 * 6
    assert all(name.startswith("transformer_blocks.") for name in names)
    assert "proj_in" not in names and "proj_out" not in names


def test_skip_blocks_keeps_them_dense():
    decoder = _tiny_decoder()
    names = convert_decoder_to_nvfp4(decoder, skip_blocks=(0, 2))
    assert {name.split(".")[1] for name in names} == {"1"}
    assert type(decoder.transformer_blocks[0].attn.to_q) is nn.Linear
    assert isinstance(decoder.transformer_blocks[1].attn.to_q, NVFP4DecoderLinear)
    assert isinstance(decoder.transformer_blocks[1].ff.net[2], NVFP4DecoderLinear)
    assert isinstance(decoder.transformer_blocks[1].attn.to_out[0], NVFP4DecoderLinear)


@pytest.mark.parametrize("group", [16, 64, 256])
def test_rotation_is_exact_in_full_precision(group):
    torch.manual_seed(0)
    linear = nn.Linear(256, 96)
    rotated = NVFP4DecoderLinear.from_linear(linear, rotation_group=group, compute_dtype=torch.float32)
    x = torch.randn(5, 7, 256, dtype=torch.float64)
    expected = x @ linear.weight.double().T + linear.bias.double()
    actual = rotate_activation(x, group) @ rotated.weight.double().T + rotated.bias.double()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rejects_non_blockable_input():
    with pytest.raises(ValueError, match="divisible"):
        NVFP4DecoderLinear.from_linear(nn.Linear(24, 32), rotation_group=None, compute_dtype=torch.bfloat16)


def _fp4_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def test_rejects_unknown_activation_scale():
    with pytest.raises(ValueError, match="act_scale"):
        NVFP4DecoderLinear.from_linear(nn.Linear(32, 32), rotation_group=None, compute_dtype=torch.bfloat16,
                                       act_scale="per-token")


@pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")
@pytest.mark.parametrize("act_scale", ["dynamic", "unit", "static"])
@pytest.mark.parametrize("group", [None, 256])
def test_inference_matches_training_forward_bitwise(group, act_scale):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import calibrate_static_scales

    torch.manual_seed(0)
    linear = nn.Linear(2048, 768).cuda()
    layer = NVFP4DecoderLinear.from_linear(linear, rotation_group=group, compute_dtype=torch.bfloat16,
                                           act_scale=act_scale).cuda()
    x = torch.randn(3, 300, 2048, device="cuda", dtype=torch.bfloat16)
    if act_scale == "static":
        assert calibrate_static_scales(layer, lambda: layer(x)) == 1
        assert layer.input_amax.item() > 0
    with torch.enable_grad():
        train_out = layer(x)
    with torch.no_grad():
        deploy_out = layer(x)
    assert torch.equal(train_out.detach(), deploy_out)
    relative = (deploy_out.float() - linear(x.float())).norm() / linear(x.float()).norm()
    assert relative < 0.15


@pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")
def test_training_forward_has_gradients():
    torch.manual_seed(0)
    layer = NVFP4DecoderLinear.from_linear(nn.Linear(256, 128), rotation_group=64,
                                           compute_dtype=torch.bfloat16).cuda()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_global_scale_matches_reference_reduction(dtype):
    from fastvideo.layers.fp4linear import _global_sf

    torch.manual_seed(0)
    x = (torch.randn(1797, 256) * 7).to(dtype)
    expected = (448.0 * 6.0) / x.float().abs().nan_to_num().max().clamp(min=1e-12)
    assert torch.equal(_global_sf(x), expected)
    with_inf = x.clone()
    with_inf[3, 5] = float("inf")
    expected_inf = (448.0 * 6.0) / with_inf.float().abs().nan_to_num().max().clamp(min=1e-12)
    assert torch.equal(_global_sf(with_inf), expected_inf)


# Fused inference kernels: every one must match the eager op sequence bit for bit.
_needs_fp4 = pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")


def _flashinfer_quantize(x, global_sf):
    import flashinfer

    from fastvideo.layers.quantization.nvfp4_config import nvfp4_quantize_fenced

    return nvfp4_quantize_fenced(x, global_sf, flashinfer.SfLayout.layout_128x4.value)


def _assert_quantized_equal(actual, expected):
    assert actual[0].shape == expected[0].shape and actual[1].shape == expected[1].shape
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))


@_needs_fp4
@pytest.mark.parametrize("global_sf", [1.0, 2688.0 / 40.0])
@pytest.mark.parametrize("magnitude", [3.0, 3000.0])
def test_fused_quantize_matches_flashinfer(global_sf, magnitude):
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import quantize

    torch.manual_seed(0)
    x = (torch.randn(1797 * 2 + 5, 2048, device="cuda") * magnitude).bfloat16()
    x[7, :16] = 0  # an all-zero block
    sf = torch.tensor(global_sf, device="cuda")
    _assert_quantized_equal(quantize(x, sf), _flashinfer_quantize(x, sf))


@_needs_fp4
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("norm", [False, True])
def test_fused_residual_norm_quantize_matches_eager(residual, norm):
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import residual_norm_quantize

    if not (residual or norm):
        pytest.skip("nothing to fuse")
    torch.manual_seed(0)
    rows, dim, eps = 1797 * 2 + 5, 2048, 1e-5
    h = torch.randn(rows, dim, device="cuda") * 5
    o = (torch.randn(rows, dim, device="cuda") * 2).bfloat16()
    bias = torch.randn(dim, device="cuda") * 0.1
    scale = torch.randn(dim, device="cuda") * 0.3
    gamma = torch.rand(dim, device="cuda") + 0.5
    sf = torch.ones((), device="cuda")
    expected_h = h
    if residual:
        biased = o.clone()
        biased.add_(bias.to(torch.bfloat16))
        expected_h = h + biased * scale
    actual_h = h.clone()
    quantized = residual_norm_quantize(actual_h, (o, bias.to(torch.bfloat16), scale) if residual else None,
                                       (gamma, eps, sf) if norm else None)
    assert torch.equal(actual_h, expected_h)
    if norm:
        normed = F.rms_norm(expected_h, (dim, ), gamma, eps).to(torch.bfloat16)
        _assert_quantized_equal(quantized, _flashinfer_quantize(normed, sf))


@_needs_fp4
def test_fused_swiglu_quantize_matches_eager():
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import swiglu_quantize

    torch.manual_seed(0)
    rows, ffn = 1797 + 5, 8192
    o = (torch.randn(rows, 2 * ffn, device="cuda") * 2).bfloat16()
    bias = torch.randn(2 * ffn, device="cuda").bfloat16()
    sf = torch.ones((), device="cuda")
    biased = o.clone()
    biased.add_(bias)
    value, gate = biased.chunk(2, dim=-1)
    expected = _flashinfer_quantize((value * F.silu(gate)).contiguous(), sf)
    _assert_quantized_equal(swiglu_quantize(o, bias, sf), expected)


@_needs_fp4
def test_fused_qkv_epilogue_matches_eager_attention_prologue():
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import qkv_epilogue

    torch.manual_seed(0)
    batch, seq, heads, head_dim, rotary, eps = 2, 1797, 32, 64, 48, 1e-5
    rows = batch * seq
    projections = [(torch.randn(rows, heads * head_dim, device="cuda") * 2).bfloat16() for _ in range(3)]
    biases = tuple((torch.randn(heads * head_dim, device="cuda") * 0.2).bfloat16() for _ in range(3))
    cos = torch.randn(batch, seq, 1, rotary, device="cuda").cos().to(torch.bfloat16)
    sin = torch.randn(batch, seq, 1, rotary, device="cuda").sin().to(torch.bfloat16)
    expected = []
    for projection, bias in zip(projections, biases, strict=True):
        biased = projection.clone()
        biased.add_(bias)
        expected.append(biased.view(batch, seq, heads, head_dim))
    for index in range(2):  # the attention module's q/k RMSNorm and partial RoPE, op by op
        x = F.rms_norm(expected[index].float(), (head_dim, ), None, eps).to(torch.bfloat16)
        x_rotary, x_pass = x[..., :rotary], x[..., rotary:]
        first, second = x_rotary.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        expected[index] = torch.cat([x_rotary * cos + rotated * sin, x_pass], dim=-1)
    actual = qkv_epilogue(*projections, biases, cos.reshape(rows, rotary), sin.reshape(rows, rotary), heads, eps)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got.view(batch, seq, heads, head_dim), want)


def _fused_test_decoder(act_scale: str, num_layers: int = 2) -> MiniMaxH3VideoViTDecoder3d:
    torch.manual_seed(0)
    decoder = MiniMaxH3VideoViTDecoder3d(in_channels=24,
                                         out_channels=3,
                                         patch_size=2,
                                         patch_size_t=1,
                                         num_layers=num_layers,
                                         num_attention_heads=8,
                                         attention_head_dim=64,
                                         num_register_tokens=4,
                                         ffn_mult=4,
                                         rope_theta=100.0,
                                         rope_dim_ratio=0.75,
                                         norm_eps=1e-5).cuda()
    with torch.no_grad():
        for block in decoder.transformer_blocks:
            block.scale1.normal_(0, 0.5)
            block.scale2.normal_(0, 0.5)
        decoder.register_tokens.normal_()
    convert_decoder_to_nvfp4(decoder, compute_dtype=torch.bfloat16, act_scale=act_scale)
    return decoder.requires_grad_(False).eval()


@_needs_fp4
@pytest.mark.parametrize("act_scale", ["unit", "static"])
@pytest.mark.parametrize("autocast", [True, False])
def test_fused_decoder_matches_eager_decoder_bitwise(act_scale, autocast):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import calibrate_static_scales

    decoder = _fused_test_decoder(act_scale)
    latents = torch.randn(3, 24, 3, 8, 8, device="cuda")
    fused = decoder.fused_blocks_forward
    assert fused is not None

    def run():
        context = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
        with torch.no_grad(), context:
            return decoder(latents)

    if act_scale == "static":
        decoder.fused_blocks_forward = None
        calibrate_static_scales(decoder, run)
        decoder.fused_blocks_forward = fused
    fused_out = run()
    assert all(block._nvfp4_fused_plan[1] is not None for block in decoder.transformer_blocks)
    decoder.fused_blocks_forward = None
    eager_out = run()
    assert torch.equal(fused_out, eager_out)


@_needs_fp4
def test_fused_decoder_falls_back_when_inapplicable():
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import fused_nvfp4_blocks_forward

    hidden = torch.randn(1, 64, 512, device="cuda")
    rotary = (torch.ones(1, 64, 1, 48, device="cuda"), torch.zeros(1, 64, 1, 48, device="cuda"))
    dynamic = _fused_test_decoder("dynamic", num_layers=1)
    with torch.no_grad():
        assert fused_nvfp4_blocks_forward(dynamic.transformer_blocks, hidden, rotary) is None
    unit = _fused_test_decoder("unit", num_layers=1)
    with torch.no_grad():
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden, rotary) is not None
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden.bfloat16(), rotary) is None
    with torch.enable_grad():
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden, rotary) is None
