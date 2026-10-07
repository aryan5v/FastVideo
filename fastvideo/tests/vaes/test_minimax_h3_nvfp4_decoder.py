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


@pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")
@pytest.mark.parametrize("group", [None, 256])
def test_inference_matches_training_forward_bitwise(group):
    torch.manual_seed(0)
    linear = nn.Linear(2048, 768).cuda()
    layer = NVFP4DecoderLinear.from_linear(linear, rotation_group=group, compute_dtype=torch.bfloat16).cuda()
    x = torch.randn(3, 300, 2048, device="cuda", dtype=torch.bfloat16)
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
