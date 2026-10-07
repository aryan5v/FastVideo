# SPDX-License-Identifier: Apache-2.0
"""NVFP4 linears for the MiniMax-H3 ViT video decoder.

Training and inference share one numeric path, so a decoder distilled with
quantization-aware training runs at deployment exactly as it was trained:

* activations: dynamic per-tensor global scale + per-16 E4M3 block scales
  (``_global_sf`` / ``nvfp4_quantize_fenced``), FP4 E2M1 values;
* weights: the same scheme, quantized once at inference and cached;
* GEMM: ``flashinfer.mm_fp4`` (cutlass), bias added in the activation dtype.

Training runs through ``_LinearFWD4BWD16Fn`` (FP4 forward, full-precision
straight-through backward). An optional block Hadamard rotation (the regular
Hadamard used by the INT8 ConvRot overlay) is folded into the weight along the
input dimension and applied to the activation at runtime; it is exact in full
precision and spreads activation outliers before FP4 rounding.

Only the transformer-block linears are replaced. ``proj_in`` (K = 24, not
FP4-blockable), ``proj_out``, norms, register tokens and attention stay as they
are.
"""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn

from fastvideo.models.vaes.minimax_h3_int8_convrot import rotate_activation

NVFP4_BLOCK_SIZE = 16
# Block linears of MiniMaxH3VideoTransformerBlock, relative to ``decoder.``.
DECODER_BLOCK_LINEAR = re.compile(r"^transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0"
                                  r"|ff\.net\.0\.proj|ff\.net\.2)$")


def _quantized_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight the way ``_LinearFWD4BWD16Fn`` does on every training step."""
    from fastvideo.layers.fp4linear import _global_sf, _require_flashinfer
    from fastvideo.layers.quantization.nvfp4_config import nvfp4_quantize_fenced

    flashinfer_mod = _require_flashinfer()
    global_sf = _global_sf(weight)
    packed, inv_scale = nvfp4_quantize_fenced(weight, global_sf, flashinfer_mod.SfLayout.layout_128x4.value)
    return packed, inv_scale, global_sf


def nvfp4_linear_inference(x: torch.Tensor, packed: torch.Tensor, inv_scale: torch.Tensor, global_sf_w: torch.Tensor,
                           bias: torch.Tensor | None, out_features: int) -> torch.Tensor:
    """FP4 GEMM with the training forward's activation quantization."""
    from fastvideo.layers.fp4linear import _global_sf, _require_flashinfer
    from fastvideo.layers.quantization.nvfp4_config import nvfp4_quantize_fenced

    flashinfer_mod = _require_flashinfer()
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    global_sf_x = _global_sf(x2d)
    x_fp4, x_inv_scale = nvfp4_quantize_fenced(x2d, global_sf_x, flashinfer_mod.SfLayout.layout_128x4.value)
    out = torch.empty((x2d.shape[0], out_features), device=x.device, dtype=x.dtype)
    flashinfer_mod.mm_fp4(
        x_fp4,
        packed.T,
        x_inv_scale,
        inv_scale.T,
        1.0 / (global_sf_x * global_sf_w),
        x.dtype,
        out,
        block_size=NVFP4_BLOCK_SIZE,
        use_8x4_sf_layout=False,
        backend="cutlass",
    )
    if bias is not None:
        out.add_(bias.to(x.dtype))
    return out.reshape(*orig_shape[:-1], out_features)


class NVFP4DecoderLinear(nn.Module):
    """Drop-in for an ``nn.Linear`` in the H3 decoder blocks.

    ``weight`` is the full-precision master (already rotated when
    ``rotation_group`` is set). In training mode with gradients enabled the
    forward is the FP4 straight-through function; otherwise the cached packed
    weight is used. ``invalidate()`` must follow any in-place weight update
    made outside the optimizer step hooks (the trainer calls it after each step).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, rotation_group: int | None,
                 compute_dtype: torch.dtype) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        if in_features % NVFP4_BLOCK_SIZE:
            raise ValueError(f"NVFP4 needs in_features divisible by {NVFP4_BLOCK_SIZE}, got {in_features}")
        if rotation_group is not None and in_features % rotation_group:
            raise ValueError(f"rotation group {rotation_group} does not divide in_features {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.rotation_group = rotation_group
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self._packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rotation_group: int | None,
                    compute_dtype: torch.dtype) -> NVFP4DecoderLinear:
        weight = linear.weight.detach().float()
        if rotation_group is not None:
            # y = x W^T = (x H)(W H)^T for orthonormal symmetric H: rotate W along its input dim.
            weight = rotate_activation(weight, rotation_group)
        bias = linear.bias.detach().float() if linear.bias is not None else None
        return cls(weight, bias, rotation_group, compute_dtype)

    def invalidate(self) -> None:
        self._packed = None

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rotation_group={self.rotation_group}, compute_dtype={self.compute_dtype}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.compute_dtype)
        if self.rotation_group is not None:
            x = rotate_activation(x, self.rotation_group)
        if torch.is_grad_enabled() and self.weight.requires_grad:
            from fastvideo.layers.fp4linear import _LinearFWD4BWD16Fn
            self._packed = None
            return _LinearFWD4BWD16Fn.apply(x, self.weight, self.bias, "cutlass", NVFP4_BLOCK_SIZE, True)
        if self._packed is None:
            self._packed = _quantized_weight(self.weight.detach().to(self.compute_dtype))
        return nvfp4_linear_inference(x, *self._packed, self.bias, self.out_features)


def nvfp4_decoder_linear_names(decoder: nn.Module, skip_blocks: tuple[int, ...] = ()) -> list[str]:
    """Names (relative to the decoder) of the block linears to replace."""
    names = []
    for name, module in decoder.named_modules():
        match = DECODER_BLOCK_LINEAR.match(name)
        if match and type(module) is nn.Linear and int(match.group(1)) not in skip_blocks:
            names.append(name)
    return names


def convert_decoder_to_nvfp4(decoder: nn.Module,
                             *,
                             rotation_group: int | None = None,
                             skip_blocks: tuple[int, ...] = (),
                             compute_dtype: torch.dtype = torch.bfloat16) -> list[str]:
    """Replace the decoder's block linears with ``NVFP4DecoderLinear``; return the replaced names."""
    names = nvfp4_decoder_linear_names(decoder, skip_blocks)
    if not names:
        raise ValueError("No MiniMax-H3 decoder block linears found to convert")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = decoder.get_submodule(parent_name)
        original = getattr(parent, child) if not child.isdigit() else parent[int(child)]
        replacement = NVFP4DecoderLinear.from_linear(original, rotation_group=rotation_group,
                                                     compute_dtype=compute_dtype)
        if child.isdigit():
            parent[int(child)] = replacement
        else:
            setattr(parent, child, replacement)
    return names


def nvfp4_decoder_metadata(names: list[str], rotation_group: int | None, skip_blocks: tuple[int, ...]) -> dict[str, Any]:
    return {
        "format": "fastvideo_h3_decoder_nvfp4_qad",
        "block_size": NVFP4_BLOCK_SIZE,
        "rotation_group": rotation_group,
        "skip_blocks": list(skip_blocks),
        "num_linears": len(names),
    }
