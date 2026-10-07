# SPDX-License-Identifier: Apache-2.0
"""NVFP4 linears for the MiniMax-H3 ViT video decoder.

Training and inference share one numeric path, so a decoder distilled with
quantization-aware training runs at deployment exactly as it was trained:

* activations: per-16 E4M3 block scales under a per-tensor global scale that is
  ``dynamic`` (abs-max of every call), ``unit`` (1.0, the H3 DiT scheme), or
  ``static`` (a per-layer abs-max calibrated once and stored with the weights);
* weights: the same scheme, quantized once at inference and cached;
* GEMM: ``flashinfer.mm_fp4`` (cutlass), bias added in the activation dtype.

Training runs through ``_NVFP4DecoderSTE``, whose forward *is* the inference
function and whose backward is full precision (straight-through). An optional block Hadamard rotation (the regular
Hadamard used by the INT8 ConvRot overlay) is folded into the weight along the
input dimension and applied to the activation at runtime; it is exact in full
precision and spreads activation outliers before FP4 rounding.

Only the transformer-block linears are replaced. ``proj_in`` (K = 24, not
FP4-blockable), ``proj_out``, norms, register tokens and attention stay as they
are.
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


ACT_SCALES = ("dynamic", "unit", "static")
FP4_E4M3_RANGE = 448.0 * 6.0


def nvfp4_linear_inference(x: torch.Tensor, packed: torch.Tensor, inv_scale: torch.Tensor, global_sf_w: torch.Tensor,
                           bias: torch.Tensor | None, out_features: int,
                           global_sf_x: torch.Tensor | None) -> torch.Tensor:
    """FP4 GEMM. ``global_sf_x=None`` computes the activation scale dynamically."""
    from fastvideo.layers.fp4linear import _global_sf, _require_flashinfer
    from fastvideo.layers.quantization.nvfp4_config import nvfp4_quantize_fenced

    flashinfer_mod = _require_flashinfer()
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    if global_sf_x is None:
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


class NVFP4FusedState(NamedTuple):
    """Everything the fused inference kernels need from one ``NVFP4DecoderLinear``.

    Each tensor is computed with the same expression as ``nvfp4_linear_inference``, so a GEMM
    fed from here matches the eager linear bit for bit.
    """
    packed: torch.Tensor
    inv_scale: torch.Tensor
    global_sf_x: torch.Tensor
    alpha: torch.Tensor
    bias: torch.Tensor
    out_features: int


class _NVFP4DecoderSTE(torch.autograd.Function):
    """Forward = ``nvfp4_linear_inference`` on a freshly quantized weight; backward = full precision."""

    @staticmethod
    def forward(ctx, x, weight, bias, global_sf_x):  # type: ignore[override]
        packed = _quantized_weight(weight.detach().to(x.dtype))
        out = nvfp4_linear_inference(x, *packed, bias, weight.shape[0], global_sf_x)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = (grad_2d @ weight.to(grad_2d.dtype)).reshape(x.shape)
        grad_w = grad_2d.t() @ x.reshape(-1, x.shape[-1]).to(grad_2d.dtype)
        grad_b = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w.to(weight.dtype), grad_b, None


class NVFP4DecoderLinear(nn.Module):
    """Drop-in for an ``nn.Linear`` in the H3 decoder blocks.

    ``weight`` is the full-precision master (already rotated when
    ``rotation_group`` is set). With gradients enabled the forward is the STE;
    otherwise the cached packed weight is used, so call ``invalidate()`` after
    any weight update (the trainer does after each optimizer step).

    ``act_scale="static"`` uses the persistent ``input_amax`` buffer. Set
    ``calibrating = True`` to run dynamically while folding each call's
    abs-max into it.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, rotation_group: int | None,
                 compute_dtype: torch.dtype, act_scale: str = "dynamic") -> None:
        super().__init__()
        out_features, in_features = weight.shape
        if in_features % NVFP4_BLOCK_SIZE:
            raise ValueError(f"NVFP4 needs in_features divisible by {NVFP4_BLOCK_SIZE}, got {in_features}")
        if rotation_group is not None and in_features % rotation_group:
            raise ValueError(f"rotation group {rotation_group} does not divide in_features {in_features}")
        if act_scale not in ACT_SCALES:
            raise ValueError(f"act_scale must be one of {ACT_SCALES}, got {act_scale!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.rotation_group = rotation_group
        self.compute_dtype = compute_dtype
        self.act_scale = act_scale
        self.calibrating = False
        # Run the master weight densely (bf16) instead of NVFP4; for sensitivity analysis and mixed precision.
        self.dense_bypass = False
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.register_buffer("input_amax", torch.zeros((), dtype=torch.float32, device=weight.device))
        self._packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._fused: tuple[int, NVFP4FusedState] | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rotation_group: int | None, compute_dtype: torch.dtype,
                    act_scale: str = "dynamic") -> NVFP4DecoderLinear:
        weight = linear.weight.detach().float()
        if rotation_group is not None:
            # y = x W^T = (x H)(W H)^T for orthonormal symmetric H: rotate W along its input dim.
            weight = rotate_activation(weight, rotation_group)
        bias = linear.bias.detach().float() if linear.bias is not None else None
        return cls(weight, bias, rotation_group, compute_dtype, act_scale)

    def invalidate(self) -> None:
        self._packed = None
        self._fused = None

    def fused_state(self) -> NVFP4FusedState | None:
        """Cached inference state for the fused decoder kernels, or None when they cannot reproduce this layer.

        The fused kernels quantize with a precomputed global scale, so a dynamic (per-call abs-max)
        scale, calibration, and the activation rotation keep the eager path.
        """
        if (self.dense_bypass or self.rotation_group is not None or self.calibrating or self.act_scale == "dynamic"
                or self.compute_dtype != torch.bfloat16 or self.bias is None):
            return None
        version = self.input_amax._version
        if self._fused is None or self._fused[0] != version:
            if self._packed is None:
                self._packed = _quantized_weight(self.weight.detach().to(self.compute_dtype))
            packed, inv_scale, global_sf_w = self._packed
            global_sf_x = self._activation_global_sf(self.weight)
            alpha = 1.0 / (global_sf_x * global_sf_w)
            state = NVFP4FusedState(packed, inv_scale, global_sf_x, alpha, self.bias.detach().to(self.compute_dtype),
                                    self.out_features)
            self._fused = (version, state)
        return self._fused[1]

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rotation_group={self.rotation_group}, act_scale={self.act_scale}")

    def _activation_global_sf(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.calibrating:
            self.input_amax.copy_(torch.maximum(self.input_amax, x.detach().abs().amax().float()))
            return None
        if self.act_scale == "dynamic":
            return None
        if self.act_scale == "unit":
            return torch.ones((), dtype=torch.float32, device=x.device)
        return FP4_E4M3_RANGE / self.input_amax.clamp(min=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.compute_dtype)
        if self.rotation_group is not None:
            x = rotate_activation(x, self.rotation_group)
        if self.dense_bypass:
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(x.dtype), bias)
        global_sf_x = self._activation_global_sf(x)
        if torch.is_grad_enabled() and self.weight.requires_grad:
            self.invalidate()
            return _NVFP4DecoderSTE.apply(x, self.weight, self.bias, global_sf_x)
        if self._packed is None:
            self._packed = _quantized_weight(self.weight.detach().to(self.compute_dtype))
        return nvfp4_linear_inference(x, *self._packed, self.bias, self.out_features, global_sf_x)


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
                             compute_dtype: torch.dtype = torch.bfloat16,
                             act_scale: str = "dynamic") -> list[str]:
    """Replace the decoder's block linears with ``NVFP4DecoderLinear``; return the replaced names."""
    names = nvfp4_decoder_linear_names(decoder, skip_blocks)
    if not names:
        raise ValueError("No MiniMax-H3 decoder block linears found to convert")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = decoder.get_submodule(parent_name)
        original = getattr(parent, child) if not child.isdigit() else parent[int(child)]
        replacement = NVFP4DecoderLinear.from_linear(original,
                                                     rotation_group=rotation_group,
                                                     compute_dtype=compute_dtype,
                                                     act_scale=act_scale)
        if child.isdigit():
            parent[int(child)] = replacement
        else:
            setattr(parent, child, replacement)
    try:
        from fastvideo.models.vaes.minimax_h3_nvfp4_fused import fused_nvfp4_blocks_forward
    except ImportError:
        # Triton-less hosts (CPU, macOS) keep the eager NVFP4 path; the fused kernels are an exact speedup only.
        return names
    # Inference runs eligible block stacks through the bit-exact fused kernels (see that module).
    decoder.fused_blocks_forward = fused_nvfp4_blocks_forward
    return names


def nvfp4_linears(module: nn.Module) -> list[NVFP4DecoderLinear]:
    return [sub for sub in module.modules() if isinstance(sub, NVFP4DecoderLinear)]


@torch.no_grad()
def calibrate_static_scales(module: nn.Module, run_batches, margin: float = 1.0) -> int:
    """Record each NVFP4 linear's activation abs-max over ``run_batches()`` (dynamic numerics while recording).

    ``margin`` > 1 widens the stored range. Returns the number of calibrated linears.
    """
    layers = nvfp4_linears(module)
    for layer in layers:
        layer.input_amax.zero_()
        layer.calibrating = True
    try:
        run_batches()
    finally:
        for layer in layers:
            layer.calibrating = False
    for layer in layers:
        if layer.input_amax.item() <= 0:
            raise RuntimeError("an NVFP4 decoder linear saw no activations during calibration")
        layer.input_amax.mul_(margin)
    return len(layers)


def nvfp4_decoder_metadata(names: list[str], rotation_group: int | None, skip_blocks: tuple[int, ...],
                           act_scale: str = "dynamic") -> dict[str, Any]:
    return {
        "format": "fastvideo_h3_decoder_nvfp4_qad",
        "block_size": NVFP4_BLOCK_SIZE,
        "rotation_group": rotation_group,
        "skip_blocks": list(skip_blocks),
        "act_scale": act_scale,
        "num_linears": len(names),
    }
