# SPDX-License-Identifier: Apache-2.0
"""PyTorch twin of MLX's MXFP4 quantizer, for Mac-targeted QAT.

The Apple Silicon MXFP4 deployment path quantizes DiT linear weights with
``mx.quantize(w, mode="mxfp4")`` and dequantizes with ``mx.dequantize``. For
quantization-aware distillation to transfer to that runtime, train-time
fake-quantization must reproduce the exact MLX deploy grid: 32-element groups,
4-bit E2M1 element codes, and one power-of-two E8M0 scale per group. MXFP4 has
no affine bias and no configurable ``bits`` argument.

This module transcribes the MLX 0.31.2 Metal behavior empirically verified in
``fastvideo/tests/mlx/test_mlx_mxfp4_qat_parity.py``. The details the twin
reproduces are:

- groups are formed along the last axis with fixed group size 32,
- the shared E8M0 scale is the nearest power of two to
  ``max(abs(group)) / 6`` (because E2M1's largest finite magnitude is 6),
- E2M1 levels are ``0, 0.5, 1, 1.5, 2, 3, 4, 6`` with a sign bit,
- midpoint ties choose the even code (for example 0.25 -> 0, 0.75 -> 1,
  1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5.0 -> 4),
- ``mx.dequantize(..., mode="mxfp4")`` returns bf16 values, regardless of
  whether the input weight was fp16 or fp32.

The public fake-quant function is torch-only and uses a straight-through
estimator: the forward value is the MLX MXFP4 dequantized tensor, while
gradients pass to the original weight unchanged.
"""

from __future__ import annotations

import torch

DEFAULT_GROUP_SIZE = 32
_MAX_E2M1 = 6.0

_CODE_TO_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _group(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} is not divisible by group_size {group_size}; "
                         "MLX MXFP4 quantization groups along the last axis.")
    return w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)


def mlx_mxfp4_quantize_reference(
    w: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize like ``mx.quantize(..., mode="mxfp4")`` on MLX Metal.

    Returns ``(codes, scales)`` with unpacked integer ``codes`` (one 4-bit code
    value per element, not MLX's packed uint32 words) and uint8 E8M0 scale
    bytes. ``group_size`` is fixed to 32 by MLX for MXFP4; the argument exists
    for tests and error messages, not as a tunable deploy parameter.
    """
    if group_size != DEFAULT_GROUP_SIZE:
        raise ValueError("MLX MXFP4 uses fixed group_size 32")

    grouped = _group(w, group_size).float()
    max_abs = grouped.abs().amax(dim=-1)

    exponent = torch.round(torch.log2(max_abs / _MAX_E2M1))
    exponent = torch.where(max_abs == 0, torch.full_like(exponent, -127.0), exponent)
    exponent = exponent.clamp(min=-127.0, max=127.0)
    scale = torch.pow(2.0, exponent)

    normalized_abs = (grouped / scale.unsqueeze(-1)).abs()
    code_mag = torch.zeros_like(normalized_abs, dtype=torch.int32)
    code_mag = torch.where(normalized_abs > 0.25, torch.ones_like(code_mag), code_mag)
    code_mag = torch.where(normalized_abs >= 0.75, torch.full_like(code_mag, 2), code_mag)
    code_mag = torch.where(normalized_abs > 1.25, torch.full_like(code_mag, 3), code_mag)
    code_mag = torch.where(normalized_abs >= 1.75, torch.full_like(code_mag, 4), code_mag)
    code_mag = torch.where(normalized_abs > 2.5, torch.full_like(code_mag, 5), code_mag)
    code_mag = torch.where(normalized_abs >= 3.5, torch.full_like(code_mag, 6), code_mag)
    code_mag = torch.where(normalized_abs > 5.0, torch.full_like(code_mag, 7), code_mag)

    sign = torch.signbit(grouped)
    codes = torch.where(sign, code_mag + 8, code_mag)
    scales = (exponent + 127.0).to(torch.uint8)
    return codes, scales


def mlx_mxfp4_dequantize_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_shape: torch.Size | None = None,
) -> torch.Tensor:
    """Dequantize unpacked MXFP4 codes like MLX, returning fp32 bf16 values."""
    scale = torch.pow(2.0, scales.float() - 127.0)
    codebook = _CODE_TO_E2M1.to(device=codes.device)
    deq = codebook[codes.long()] * scale.unsqueeze(-1)
    deq = deq.to(torch.bfloat16).float()
    if out_shape is not None:
        deq = deq.reshape(out_shape)
    return deq


def fake_quantize_mlx_mxfp4(
    w: torch.Tensor,
    *,
    simulate_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Straight-through-estimator fake quantization for MXFP4 QAD forwards.

    Simulates the deploy pipeline: cast the master weight to ``simulate_dtype``
    before quantization, apply MLX's fixed MXFP4 grid, and dequantize to bf16.
    The returned fp32 tensor represents those bf16 deploy-time values exactly.
    Gradients pass through to ``w`` unchanged (STE).
    """
    w_sim = w.detach().to(simulate_dtype)
    codes, scales = mlx_mxfp4_quantize_reference(w_sim)
    deq = mlx_mxfp4_dequantize_reference(codes, scales, out_shape=w.shape)
    w32 = w.float()
    return w32 + (deq - w32).detach()
