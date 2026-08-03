# SPDX-License-Identifier: Apache-2.0
"""Pin the PyTorch MX quantizer against MLX's ``mx.quantize(mode="mxfp4"/"mxfp8")``.

This is the gate. Two mxfp4 QAD runs (1.3B and 14B, ~31 GPU-days on B200s)
produced bad output using a fake-quantizer that was never checked against the
deploy grid, and the code was not even committed. A train/deploy grid mismatch
is indistinguishable from "the format doesn't work", so **no GPU time goes to an
MX QAT run until this file passes**.

Sibling of ``test_mlx_affine_qat_parity.py``. Note the asymmetry: the affine
reference was transcribed from MLX's CPU kernel, so it agrees structurally.
``mlx_mx_qat`` is derived from the OCP Microscaling spec, so it agrees only if
MLX also follows the spec. If these tests fail, the likely divergence points,
in rough order of probability:

1. **Shared-exponent rule.** The spec uses
   ``floor(log2(max_abs)) - emax_element``. An implementation may bias this by
   one, round rather than floor, or clamp differently.
2. **Rounding mode on the element grid.** Ties-to-even vs ties-away vs
   round-toward-zero.
3. **Subnormal handling.** Whether E2M1's 0.5 and E4M3's subnormal range are
   emitted at all.
4. **Block size or axis.** MX blocks are 32 along the last axis; a transposed
   or 64-wide grouping would show up as a total mismatch rather than a
   near-miss.

Requires ``mlx`` — skipped where it is unavailable, which includes CI on Linux.
Run it on the Mac before launching training on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is only available on Apple Silicon")

from fastvideo.layers.quantization.mlx_mx_qat import (  # noqa: E402
    MX_BLOCK_SIZE,
    fake_quantize_mlx_mx,
    mlx_mx_dequantize_reference,
    mlx_mx_quantize_reference,
)

MODES = ["mxfp4", "mxfp8"]


def _mlx_dequantized(w_np, mode: str):
    """Round-trip through MLX and return the dequantized weight as numpy."""
    arr = mx.array(w_np)
    packed, scales, biases = mx.quantize(arr, group_size=MX_BLOCK_SIZE, mode=mode)
    deq = mx.dequantize(packed, scales, biases, group_size=MX_BLOCK_SIZE, mode=mode)
    return np.asarray(deq)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(8, MX_BLOCK_SIZE), (16, MX_BLOCK_SIZE * 4), (64, 1536)])
def test_dequantized_weights_bitmatch_mlx(mode: str, shape: tuple[int, int]) -> None:
    """The full quantize/dequantize round-trip must match MLX exactly.

    This is the assertion that actually matters: QAT only transfers if the
    value the student trains against is the value the runtime computes with.
    """
    torch.manual_seed(0)
    w = torch.randn(*shape, dtype=torch.float16)

    elements, exps = mlx_mx_quantize_reference(w, mode=mode)
    ours = mlx_mx_dequantize_reference(elements, exps, out_shape=w.shape).numpy()
    theirs = _mlx_dequantized(w.numpy(), mode)

    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("mode", MODES)
def test_saturating_and_tiny_blocks_match(mode: str) -> None:
    """Cover the block extremes the random test is unlikely to generate.

    Blocks that are all-zero, that straddle a power of two, and that contain a
    single large outlier are where the shared-exponent rule is most likely to
    diverge.
    """
    rows = [
        torch.zeros(MX_BLOCK_SIZE),
        torch.full((MX_BLOCK_SIZE, ), 1.0),
        torch.full((MX_BLOCK_SIZE, ), 2.0 - 1e-3),
        torch.full((MX_BLOCK_SIZE, ), 2.0 + 1e-3),
        torch.cat([torch.full((MX_BLOCK_SIZE - 1, ), 1e-4), torch.tensor([100.0])]),
        torch.linspace(-6.0, 6.0, MX_BLOCK_SIZE),
    ]
    w = torch.stack(rows).to(torch.float16)

    elements, exps = mlx_mx_quantize_reference(w, mode=mode)
    ours = mlx_mx_dequantize_reference(elements, exps, out_shape=w.shape).numpy()
    theirs = _mlx_dequantized(w.numpy(), mode)

    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("mode", MODES)
def test_fake_quantize_matches_deploy_pipeline(mode: str) -> None:
    """``fake_quantize_mlx_mx`` must reproduce the deploy round-trip."""
    torch.manual_seed(1)
    w = torch.randn(32, 512, dtype=torch.bfloat16, requires_grad=True)

    fq = fake_quantize_mlx_mx(w, mode=mode, simulate_dtype=torch.float16)
    theirs = _mlx_dequantized(w.detach().to(torch.float16).numpy(), mode)

    np.testing.assert_array_equal(fq.detach().numpy(), theirs)


@pytest.mark.parametrize("mode", MODES)
def test_ste_passes_gradients_unclipped(mode: str) -> None:
    """Gradients must pass through untouched, including at grid saturation.

    A clipped STE zeroes gradients for saturated weights. mxfp4's range is
    narrow enough that this starves the student — the failure signature looks
    identical to the format being unusable, so it is asserted here rather than
    left to inspection.
    """
    w = torch.randn(8, MX_BLOCK_SIZE, dtype=torch.float32, requires_grad=True)
    # Force saturation: scale well past the top of the element grid.
    w_big = w * 1000.0

    fake_quantize_mlx_mx(w_big, mode=mode).sum().backward()

    assert w.grad is not None
    assert torch.count_nonzero(w.grad) == w.numel(), (
        "STE zeroed gradients on saturated weights; this starves QAT and is the "
        "prime suspect for the earlier mxfp4 run failures")
    torch.testing.assert_close(w.grad, torch.full_like(w.grad, 1000.0))


def test_block_size_is_enforced() -> None:
    """MX blocks are fixed at 32; an indivisible last dim must raise."""
    with pytest.raises(ValueError, match="block size"):
        mlx_mx_quantize_reference(torch.randn(4, 100), mode="mxfp4")
