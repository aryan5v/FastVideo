# SPDX-License-Identifier: Apache-2.0
"""PyTorch twin of MLX's affine quantizer, for Mac-targeted QAT.

The Apple Silicon deployment path quantizes DiT linear weights with
``mx.quantize(w, group_size=64, bits=8, mode="affine")``. For
quantization-aware training to transfer to that runtime, the train-time
fake-quantization must reproduce MLX's quantizer exactly — a different
rounding rule or zero-point convention silently erases the QAT gains at
deploy time.

This module transcribes the affine algorithm from MLX's CPU kernel
(``mlx/backend/cpu/quantized.cpp::quantize`` at v0.31.2), whose non-obvious
details are:

- per-group min/max is computed in fp32 regardless of the input dtype,
- the scale is *negative* when ``|w_max| >= |w_min|`` (the quantizer anchors
  at the endpoint with the larger magnitude),
- the anchor endpoint is re-expressed as an exact integer multiple of the
  scale (``q0 = rint(edge / scale); scale = edge / q0; bias = edge``), and
  MLX uses that adjusted scale to produce the integer codes,
- rounding is ``rint`` (round-half-to-even), matching ``torch.round``,
- codes are clamped to ``[0, 2^bits - 1]`` and scales/biases are cast to the
  input dtype at the end.

``fastvideo/tests/mlx/test_mlx_affine_qat_parity.py`` pins this against the
real ``mx.quantize``/``mx.dequantize`` — that test is the gate the roadmap
requires before any GPU is spent on a QAT run.
"""

from __future__ import annotations

import torch

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 8
_EPS = 1e-7


def _group(w: torch.Tensor, group_size: int) -> torch.Tensor:
    if w.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} is not divisible by group_size {group_size}; "
                         "MLX affine quantization groups along the last axis.")
    return w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)


def mlx_affine_quantize_reference(
    w: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize exactly like ``mx.quantize(..., mode="affine")``.

    Returns ``(codes, scales, biases)`` with unpacked integer ``codes`` (one
    uint8-range value per element, not MLX's packed uint32 words); ``scales``
    and ``biases`` are cast to ``w.dtype`` like MLX casts to its input dtype.
    """
    n_bins = float((1 << bits) - 1)
    grouped = _group(w, group_size).float()

    w_min = grouped.min(dim=-1).values
    w_max = grouped.max(dim=-1).values
    mask = w_min.abs() > w_max.abs()
    quant_scale = ((w_max - w_min) / n_bins).clamp_min(_EPS)
    quant_scale = torch.where(mask, quant_scale, -quant_scale)
    edge = torch.where(mask, w_min, w_max)

    q0 = torch.round(edge / quant_scale)
    nonzero_q0 = q0 != 0
    scale = torch.where(
        nonzero_q0,
        edge / torch.where(nonzero_q0, q0, torch.ones_like(q0)),
        quant_scale,
    )
    bias = torch.where(nonzero_q0, edge, torch.zeros_like(edge))
    codes = torch.round((grouped - bias.unsqueeze(-1)) / scale.unsqueeze(-1))
    codes = codes.clamp(min=0.0, max=n_bins)
    return codes.to(torch.int32), scale.to(w.dtype), bias.to(w.dtype)


def mlx_affine_dequantize_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    out_shape: torch.Size | None = None,
) -> torch.Tensor:
    """Dequantize exactly like MLX's kernels: ``code * scale + bias`` in the
    scales' dtype (elementwise, no fp32 upcast of the fused expression)."""
    dtype = scales.dtype
    deq = codes.to(dtype) * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    if out_shape is not None:
        deq = deq.reshape(out_shape)
    return deq


def fake_quantize_mlx_affine(
    w: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    bits: int = DEFAULT_BITS,
    simulate_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Straight-through-estimator fake quantization for QAT forwards.

    Simulates the full deployment pipeline: cast the (bf16/fp32 master)
    weight to ``simulate_dtype`` — the MLX loader casts checkpoints to fp16
    before quantizing — then quantize/dequantize with MLX's affine rules.

    Returns the dequantized weight in fp32, which represents every fp16
    deploy-time value exactly; casting back to a bf16 master dtype would
    re-round and break the bit-for-bit correspondence with ``mx.dequantize``.
    Gradients pass through to ``w`` unchanged (STE).
    """
    w_sim = w.detach().to(simulate_dtype)
    codes, scales, biases = mlx_affine_quantize_reference(w_sim, group_size=group_size, bits=bits)
    deq = mlx_affine_dequantize_reference(codes, scales, biases, out_shape=w.shape).float()
    w32 = w.float()
    return w32 + (deq - w32).detach()
