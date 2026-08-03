# SPDX-License-Identifier: Apache-2.0
"""PyTorch twin of MLX's microscaling (MX) quantizers, for Mac-targeted QAT.

Sibling of :mod:`fastvideo.layers.quantization.mlx_affine_qat`. Where that
module targets ``mx.quantize(..., mode="affine")``, this one targets
``mode="mxfp4"`` and ``mode="mxfp8"`` — the OCP Microscaling formats that get a
hardware path on M5 and newer via the GPU neural accelerators.

.. warning::
   **This implementation is derived from the OCP Microscaling Formats spec,
   not transcribed from MLX's kernel.** ``mlx_affine_qat`` was written against
   ``mlx/backend/cpu/quantized.cpp`` directly, so its agreement with MLX is
   structural. This module's agreement is *asserted by test only*:
   ``fastvideo/tests/mlx/test_mlx_mx_qat_parity.py``. Do not spend GPU time on
   an MX QAT run until that test passes. A train/deploy grid mismatch silently
   erases the QAT gains and is indistinguishable from "the format doesn't
   work" — which is exactly how ~31 GPU-days were previously spent.

Format details this module encodes:

- **Block size 32** along the last axis, not 64. MX blocks are fixed by the
  spec; ``group_size`` is not a free parameter the way it is for affine.
- **Shared scale is E8M0** — a power of two, no mantissa, no zero point. This
  is the structural reason MX reconstructs worse than affine int8 at a
  comparable bit budget: affine carries a per-group scale *and* bias and can
  recenter an asymmetric group, while MX can only rescale by a power of two.
- **Elements are E2M1 (mxfp4) or E4M3 (mxfp8)**, with the finite magnitude
  grids enumerated below. Rounding is round-half-to-even *on the grid*.

Scale selection follows the spec's shared-exponent rule::

    shared_exp = floor(log2(max_abs_in_block)) - emax_element
    X = 2 ** shared_exp

Note this deliberately does *not* search for an MSE-optimal exponent. The
default rule can waste close to a full bit when a block's maximum sits just
above a power of two, and biasing the exponent down by one is a cheap and
often large post-training win — see the recovery ladder in
``docs/design/apple_silicon_minimax_h3.md``. That search belongs in a separate
calibration pass, not here: this module's whole job is to reproduce the deploy
grid exactly.
"""

from __future__ import annotations

import torch

# MX blocks are fixed at 32 elements by the OCP spec.
MX_BLOCK_SIZE = 32

# E8M0 shared-scale exponent range (bias 127; 255 is reserved for NaN).
_E8M0_MIN_EXP = -127
_E8M0_MAX_EXP = 127

# Finite magnitude grids, ascending. Index parity matches mantissa-bit parity,
# which is what makes "ties to even index" equivalent to IEEE ties-to-even.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EMAX = 2

_E4M3_EMAX = 8

_MODE_SPECS = {
    "mxfp4": {"emax": _E2M1_EMAX, "bits": 4},
    "mxfp8": {"emax": _E4M3_EMAX, "bits": 8},
}


def _e4m3_magnitudes() -> tuple[float, ...]:
    """OCP FP8 E4M3 finite magnitudes, ascending.

    Bias 7, 3 mantissa bits, no infinities; ``S.1111.111`` is NaN, so the
    largest finite magnitude is 448 rather than 480.
    """
    mags = [0.0]
    # Subnormals: exponent field 0, value = (m / 8) * 2 ** (1 - 7).
    for m in range(1, 8):
        mags.append(m / 8.0 * 2.0**-6)
    # Normals: exponent field 1..15, value = (1 + m / 8) * 2 ** (e - 7),
    # excluding the NaN encoding at e=15, m=7.
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            mags.append((1.0 + m / 8.0) * 2.0**(e - 7))
    return tuple(sorted(mags))


_MAGNITUDE_GRIDS: dict[str, tuple[float, ...]] = {
    "mxfp4": _E2M1_MAGNITUDES,
    "mxfp8": _e4m3_magnitudes(),
}


def _block(w: torch.Tensor, block_size: int = MX_BLOCK_SIZE) -> torch.Tensor:
    if w.shape[-1] % block_size != 0:
        raise ValueError(f"Last dim {w.shape[-1]} is not divisible by MX block size {block_size}; "
                         "MX quantization blocks along the last axis.")
    return w.reshape(*w.shape[:-1], w.shape[-1] // block_size, block_size)


def _round_to_grid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to the nearest grid level, ties to even index.

    ``grid`` is ascending and non-uniform, so nearest-neighbour is found by
    bucketing against midpoints rather than by scaling. Exact ties land on the
    even-indexed neighbour, which for both E2M1 and E4M3 is the level whose
    mantissa bit is zero.
    """
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    # right=False resolves an exact midpoint downward, to grid[idx].
    idx = torch.bucketize(values, midpoints, right=False)

    # A value sitting exactly on midpoints[idx] is tied between grid[idx] and
    # grid[idx + 1]; bucketize already took the lower one, so only an odd idx
    # needs bumping to reach the even neighbour.
    n_mid = midpoints.numel()
    safe = idx.clamp(max=n_mid - 1)
    on_midpoint = (idx < n_mid) & (values == midpoints[safe])
    idx = torch.where(on_midpoint & (idx % 2 == 1), idx + 1, idx)

    return grid[idx.clamp(max=grid.numel() - 1)]


def mlx_mx_quantize_reference(
    w: torch.Tensor,
    *,
    mode: str = "mxfp4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize like ``mx.quantize(..., mode=mode)`` for an MX format.

    Returns ``(elements, scale_exponents)``. ``elements`` holds the
    *dequantized element magnitudes with sign*, one per input value, before
    the shared scale is reapplied; ``scale_exponents`` holds the integer E8M0
    exponents, one per 32-element block. Returning the decoded element values
    rather than bit patterns keeps this comparable against
    ``mx.dequantize`` without reimplementing MLX's packing.
    """
    if mode not in _MODE_SPECS:
        raise ValueError(f"Unsupported MX mode {mode!r}; expected one of {sorted(_MODE_SPECS)}")

    emax = _MODE_SPECS[mode]["emax"]
    grid = torch.tensor(_MAGNITUDE_GRIDS[mode], dtype=torch.float32, device=w.device)

    blocked = _block(w).float()
    max_abs = blocked.abs().amax(dim=-1)

    # Spec shared-exponent rule. An all-zero block gets exponent 0 (scale 1).
    nonzero = max_abs > 0
    shared_exp = torch.where(
        nonzero,
        torch.floor(torch.log2(torch.where(nonzero, max_abs, torch.ones_like(max_abs)))) - emax,
        torch.zeros_like(max_abs),
    )
    shared_exp = shared_exp.clamp(_E8M0_MIN_EXP, _E8M0_MAX_EXP)
    scale = torch.pow(torch.tensor(2.0, device=w.device), shared_exp)

    scaled = blocked / scale.unsqueeze(-1)
    elements = torch.sign(scaled) * _round_to_grid(scaled.abs(), grid)
    return elements, shared_exp.to(torch.int32)


def mlx_mx_dequantize_reference(
    elements: torch.Tensor,
    scale_exponents: torch.Tensor,
    *,
    out_shape: torch.Size | None = None,
) -> torch.Tensor:
    """Reapply the E8M0 shared scale to decoded element values."""
    scale = torch.pow(torch.tensor(2.0, device=elements.device), scale_exponents.float())
    deq = elements * scale.unsqueeze(-1)
    if out_shape is not None:
        deq = deq.reshape(out_shape)
    return deq


def fake_quantize_mlx_mx(
    w: torch.Tensor,
    *,
    mode: str = "mxfp4",
    simulate_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Straight-through-estimator fake quantization for MX QAT forwards.

    Mirrors :func:`fastvideo.layers.quantization.mlx_affine_qat.fake_quantize_mlx_affine`:
    cast the master weight to the dtype the MLX loader quantizes from, run the
    deploy quantizer, and return the dequantized value with an **unclipped**
    straight-through gradient.

    The STE is deliberately unclipped. A clipped STE zeroes gradients for
    weights that saturate the grid, and mxfp4's usable range is narrow enough
    that a large fraction of weights sit at saturation — which starves the
    student of gradient and looks exactly like "4-bit does not work". Keep the
    passthrough total unless a measurement says otherwise.
    """
    w_sim = w.detach().to(simulate_dtype)
    elements, exps = mlx_mx_quantize_reference(w_sim, mode=mode)
    deq = mlx_mx_dequantize_reference(elements, exps, out_shape=w.shape).float()
    w32 = w.float()
    return w32 + (deq - w32).detach()
