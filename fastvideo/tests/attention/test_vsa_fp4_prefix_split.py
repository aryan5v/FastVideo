# SPDX-License-Identifier: Apache-2.0
"""The FP4 prefix split's row bookkeeping and dense prefix attention.

The split takes the dense prefix query tiles (text, reference, audio rows) off
the FP4 sparse kernel and attends them densely at 8-bit Q/K. These tests cover
the parts that do not need a Blackwell GPU: the FP8 rounding, the valid-key
vector the kernel's half counts expand to, and the dense attention itself.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.models.dits.minimax_h3_vsa_fp4 import _fp8_per_token, _half_row_valid, _prefix_dense_attention

HALF = 64


def _reference_dense_attention(q, k, v, valid, scale):
    """fp32 softmax attention over ``[B, L, H, D]`` with the invalid keys masked out."""
    scores = torch.einsum("blhd,bkhd->bhlk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~valid[None, None, None, :], float("-inf"))
    return torch.einsum("bhlk,bkhd->blhd", torch.softmax(scores, dim=-1), v.float()).to(v.dtype)


def test_fp8_per_token_rounds_within_one_e4m3_step() -> None:
    """FP8 E4M3 has 3 mantissa bits, so a per-row scale bounds the error at ~6%."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 5, 128, dtype=torch.bfloat16)
    y = _fp8_per_token(x)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    assert torch.allclose(y, x, rtol=0.1, atol=1e-2)
    # Per (row, head) scale: a row whose magnitude is 1000x the others keeps its own precision.
    x[0, 0, 0] *= 1000.0
    assert ((_fp8_per_token(x)[0, 0, 0] - x[0, 0, 0]).norm() / x[0, 0, 0].norm()).item() < 0.1


def test_fp8_per_token_keeps_zero_rows_zero() -> None:
    """Pad rows stay exactly zero, which the kernel's zero-padded k/v relies on."""
    x = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(_fp8_per_token(x), x)


def test_half_row_valid_expands_counts_to_rows() -> None:
    kv_valid = torch.tensor([HALF, 3, 0, HALF], dtype=torch.int32)
    valid = _half_row_valid(kv_valid, HALF)
    assert valid.shape == (4 * HALF, )
    assert valid.sum().item() == 2 * HALF + 3
    assert valid[:HALF].all() and not valid[HALF + 3:HALF + HALF].any()
    assert not valid[2 * HALF:3 * HALF].any()


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_prefix_dense_attention_matches_a_masked_reference(precision: str) -> None:
    torch.manual_seed(0)
    rows, prefix, kv_len, heads, dim = 1, 8, 16, 2, 128
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, kv_len, heads, dim, dtype=torch.bfloat16)
    valid = torch.ones(kv_len, dtype=torch.bool)
    valid[-4:] = False  # a partial last tile

    out = _prefix_dense_attention(q, k, v, prefix, kv_len, valid, precision)

    assert out.shape == (rows, heads, prefix, dim) and out.dtype == torch.bfloat16
    q_ref = _fp8_per_token(q) if precision == "fp8" else q
    k_ref = _fp8_per_token(k) if precision == "fp8" else k
    # K smoothing shifts every score row by the same constant, so the reference
    # needs no extra term; only the valid keys set the mean.
    k_ref = k_ref - k_ref.sum(dim=1, keepdim=True) / valid.sum()
    # The kernel-shaped output is [B, H, L, D]; the reference builds [B, L, H, D].
    expected = _reference_dense_attention(q_ref, k_ref, v, valid, dim**-0.5).transpose(1, 2)

    assert (out.float() - expected.float()).norm() / expected.float().norm() < 2e-2


def test_prefix_dense_attention_ignores_invalid_keys() -> None:
    """Masked-out keys must not reach the output.

    Truncating the keys to the valid rows is the same computation: the mask
    covers exactly those rows and K smoothing averages exactly them, so the two
    must agree however extreme the masked-out rows are.
    """
    torch.manual_seed(0)
    rows, prefix, valid_len, heads, dim = 1, 4, 6, 2, 128
    q = torch.randn(rows, prefix, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(rows, valid_len + 2, heads, dim, dtype=torch.bfloat16)
    # Comparable scale: a huge value would push the smoothing constant so far
    # that bf16 cannot resolve the valid keys after subtracting the mean.
    k[:, valid_len:] = 3.0
    v[:, valid_len:] = -3.0
    valid = torch.ones(valid_len + 2, dtype=torch.bool)
    valid[valid_len:] = False

    masked = _prefix_dense_attention(q, k, v, prefix, valid_len + 2, valid, "bf16")
    truncated = _prefix_dense_attention(q, k[:, :valid_len], v[:, :valid_len], prefix, valid_len,
                                        torch.ones(valid_len, dtype=torch.bool), "bf16")

    assert (masked.float() - truncated.float()).norm() / truncated.float().norm() < 5e-3
