# SPDX-License-Identifier: Apache-2.0
"""Inference-only Triton fusions promoted from the AutoKernel Wan corpus."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from fastvideo.optimization.artifacts import get_optimized_kernel

logger = logging.getLogger(__name__)


def _run_promoted(
    operation: str,
    *args: torch.Tensor,
) -> tuple[bool, object]:
    kernel = get_optimized_kernel(operation)
    if kernel is None:
        return False, None
    try:
        return True, kernel(*args)
    except Exception:
        logger.exception(
            "Promoted %s kernel failed; using bundled fallback",
            operation,
        )
        return False, None


@triton.jit
def _gated_residual_layer_norm_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    updated_ptr,
    tokens,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    row_offsets = row * hidden + offsets
    gate_offsets = batch * hidden + offsets

    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    updated = residual + x * gate

    mean = tl.sum(updated, axis=0) / hidden
    centered = tl.where(mask, updated - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    normalized = centered * tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = normalized * weight + bias

    tl.store(normalized_ptr + row_offsets, normalized, mask=mask)
    tl.store(updated_ptr + row_offsets, updated, mask=mask)


@triton.jit
def _modulated_layer_norm_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    tokens,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    row_offsets = row * hidden + offsets
    modulation_offsets = batch * hidden + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    normalized = centered * tl.rsqrt(variance + eps)
    scale = tl.load(scale_ptr + modulation_offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + modulation_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row_offsets,
        normalized * (1.0 + scale) + shift,
        mask=mask,
    )


@triton.jit
def _gated_residual_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
    output_ptr,
    tokens,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    row_offsets = row * hidden + offsets
    gate_offsets = batch * hidden + offsets
    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + row_offsets, residual + x * gate, mask=mask)


def _validate_channelwise_inputs(
    primary: torch.Tensor,
    vectors: tuple[torch.Tensor, ...],
    *,
    primary_name: str,
) -> tuple[int, int, int]:
    if not primary.is_cuda:
        raise ValueError("Wan fusion requires CUDA tensors")
    if primary.ndim != 3:
        raise ValueError(f"{primary_name} must have shape [B, S, D]")
    if any(tensor.device != primary.device for tensor in vectors):
        raise ValueError(f"all inputs must be on the {primary_name} device")
    if not all(tensor.is_contiguous() for tensor in (primary, *vectors)):
        raise ValueError("all inputs must be contiguous")
    batch, tokens, hidden = primary.shape
    if hidden > 65536:
        raise ValueError("hidden dimension exceeds the Triton baseline limit")
    return batch, tokens, hidden


def wan_gated_residual_layer_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the promoted Wan self-attention transition in one launch."""
    promoted, result = _run_promoted(
        "wan_gated_residual_norm",
        residual,
        x,
        gate,
        weight,
        bias,
    )
    if promoted:
        assert isinstance(result, tuple)
        return result
    if not residual.is_cuda:
        raise ValueError("Wan fused normalization requires CUDA tensors")
    if residual.ndim != 3 or x.shape != residual.shape:
        raise ValueError("residual and x must have matching [B, S, D] shapes")
    if any(tensor.device != residual.device for tensor in (x, gate, weight, bias)):
        raise ValueError("all inputs must be on the residual device")
    if not all(tensor.is_contiguous() for tensor in (residual, x, gate, weight, bias)):
        raise ValueError("all inputs must be contiguous")

    batch, tokens, hidden = residual.shape
    if gate.shape != (batch, hidden):
        raise ValueError(f"gate must have shape {(batch, hidden)}")
    if weight.shape != (hidden, ) or bias.shape != (hidden, ):
        raise ValueError(f"weight and bias must have shape {(hidden,)}")
    if hidden > 65536:
        raise ValueError("hidden dimension exceeds the Triton baseline limit")
    normalized = torch.empty_like(residual)
    updated = torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _gated_residual_layer_norm_kernel[(batch * tokens, )](
        residual,
        x,
        gate,
        weight,
        bias,
        normalized,
        updated,
        tokens=tokens,
        hidden=hidden,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return normalized, updated


def wan_modulated_layer_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run Wan's pre-attention LayerNorm and modulation in one launch."""
    promoted, result = _run_promoted(
        "wan_modulated_layer_norm",
        x,
        scale,
        shift,
    )
    if promoted:
        assert isinstance(result, torch.Tensor)
        return result
    batch, tokens, hidden = _validate_channelwise_inputs(x, (scale, shift), primary_name="x")
    if scale.shape != (batch, hidden) or shift.shape != (batch, hidden):
        raise ValueError(f"scale and shift must have shape {(batch, hidden)}")
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _modulated_layer_norm_kernel[(batch * tokens, )](
        x,
        scale,
        shift,
        output,
        tokens=tokens,
        hidden=hidden,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def wan_gated_residual(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Run Wan's post-MLP gated residual update in one launch."""
    promoted, result = _run_promoted(
        "wan_gated_residual",
        residual,
        x,
        gate,
    )
    if promoted:
        assert isinstance(result, torch.Tensor)
        return result
    batch, tokens, hidden = _validate_channelwise_inputs(residual, (x, gate), primary_name="residual")
    if x.shape != residual.shape:
        raise ValueError("residual and x must have matching shapes")
    if gate.shape != (batch, hidden):
        raise ValueError(f"gate must have shape {(batch, hidden)}")
    output = torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if block_size <= 2048 else 8
    _gated_residual_kernel[(batch * tokens, )](
        residual,
        x,
        gate,
        output,
        tokens=tokens,
        hidden=hidden,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
