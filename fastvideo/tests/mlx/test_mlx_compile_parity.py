# SPDX-License-Identifier: Apache-2.0
"""Guard the ``mx.compile`` path of the MLX FastWan forward.

`mx.compile` on the DiT forward was silently broken: a NumPy scalar multiplying
a traced array (``np.sqrt(2/pi) * x`` in ``gelu_tanh``) dispatched through NumPy,
which evals the traced array — illegal during compile. It raised "Attempting to
eval an array during function transformations" (caught → eager fallback, no
speedup) or segfaulted the process. These tests keep the compiled forward tracing
cleanly and numerically identical to eager, so the ~1.4x compile speedup cannot
silently regress. See ``fastvideo/mlx_runtime/fastwan.py`` and
``docs/design/apple_silicon_benchmark_baseline.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the compile-parity tests")

from fastvideo.mlx_runtime.fastwan import (  # noqa: E402
    _GELU_TANH_COEF,
    MLXWanTransformerBlock,
    gelu_tanh,
    timestep_embedding,
)


def _rand(*shape: int, scale: float = 1.0) -> "mx.array":
    return mx.array((np.random.default_rng(0).standard_normal(shape) * scale).astype(np.float32))


def test_gelu_tanh_compiles_and_matches_eager() -> None:
    """The tanh-GELU (the op that broke compile) traces and matches eager."""
    x = _rand(1, 120, 64)
    eager = gelu_tanh(x)
    mx.eval(eager)
    compiled = mx.compile(gelu_tanh)(x)
    mx.eval(compiled)
    np.testing.assert_array_equal(np.array(eager), np.array(compiled))


def test_gelu_tanh_coefficient_is_a_python_float() -> None:
    """Regression guard: the tanh-GELU coefficient must be a plain Python
    ``float``, not a NumPy scalar — a ``np.float64`` here is exactly what
    dispatched through NumPy's ``__mul__``, evaluated the traced array, and
    broke ``mx.compile`` (see the lesson this test file guards against)."""
    assert type(_GELU_TANH_COEF) is float
    assert _GELU_TANH_COEF == math.sqrt(2.0 / math.pi)


def test_gelu_tanh_matches_elementwise_reference() -> None:
    """The tanh-GELU output matches a plain NumPy/math reimplementation,
    independent of the compile machinery."""
    x_np = np.linspace(-6.0, 6.0, 97, dtype=np.float32)
    expected = 0.5 * x_np * (1.0 + np.tanh(_GELU_TANH_COEF * (x_np + 0.044715 * x_np**3)))

    out = gelu_tanh(mx.array(x_np))
    mx.eval(out)

    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-6)


def test_gelu_tanh_zero_is_zero() -> None:
    """``gelu_tanh(0) == 0`` exactly, regardless of the coefficient's dtype."""
    out = gelu_tanh(mx.array([0.0], dtype=mx.float32))
    mx.eval(out)
    assert np.array(out)[0] == 0.0


def test_timestep_embedding_compiles_and_matches_eager() -> None:
    """``timestep_embedding`` (also touched by the np-scalar -> math fix)
    traces under ``mx.compile`` and matches eager exactly."""
    t = mx.array([0.0, 1.0, 17.5, 999.0], dtype=mx.float32)
    dim = 64

    eager = timestep_embedding(t, dim)
    mx.eval(eager)
    compiled = mx.compile(lambda tt: timestep_embedding(tt, dim))(t)
    mx.eval(compiled)

    np.testing.assert_array_equal(np.array(eager), np.array(compiled))


def test_timestep_embedding_matches_reference() -> None:
    """The embedding matches a plain NumPy/math reimplementation of the
    standard sinusoidal timestep embedding."""
    t_np = np.array([0.0, 3.0, 42.0], dtype=np.float32)
    dim, max_period = 32, 10000

    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = t_np[:, None] * freqs[None]
    expected = np.concatenate([np.cos(args), np.sin(args)], axis=-1)

    out = timestep_embedding(mx.array(t_np), dim, max_period)
    mx.eval(out)

    assert out.shape == (3, dim)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-6)


def test_timestep_embedding_odd_dim_pads_zero_column() -> None:
    """When ``dim`` is odd, a trailing zero column is appended so the output
    still has ``dim`` columns."""
    t = mx.array([1.0, 2.0], dtype=mx.float32)
    dim = 33

    out = timestep_embedding(t, dim)
    mx.eval(out)
    out_np = np.array(out)

    assert out_np.shape == (2, dim)
    np.testing.assert_array_equal(out_np[:, -1], np.zeros(2, dtype=out_np.dtype))


def _tiny_block_weights(dim: int, ffn: int) -> dict:
    square = ["to_q", "to_k", "to_v", "to_out", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]
    weights = {f"{k}.weight": _rand(dim, dim, scale=0.05) for k in square}
    weights.update({f"{k}.bias": _rand(dim, scale=0.05) for k in ["to_q", "to_k", "to_v", "to_out"]})
    weights.update({
        "scale_shift_table": _rand(1, 6, dim, scale=0.05),
        "norm_q.weight": _rand(dim, scale=0.05),
        "norm_k.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.weight": _rand(dim, scale=0.05),
        "self_attn_residual_norm.norm.bias": _rand(dim, scale=0.05),
        "attn2.norm_q.weight": _rand(dim, scale=0.05),
        "attn2.norm_k.weight": _rand(dim, scale=0.05),
        "ffn.fc_in.weight": _rand(ffn, dim, scale=0.05),
        "ffn.fc_in.bias": _rand(ffn, scale=0.05),
        "ffn.fc_out.weight": _rand(dim, ffn, scale=0.05),
        "ffn.fc_out.bias": _rand(dim, scale=0.05),
    })
    return weights


def test_transformer_block_compiles_and_matches_eager() -> None:
    """The full dense block (the mx.compile target's body) traces and matches."""
    dim, num_heads, head_dim, ffn, seq, ctx = 64, 4, 16, 128, 120, 32
    block = MLXWanTransformerBlock(_tiny_block_weights(dim, ffn), dim=dim, ffn_dim=ffn, num_heads=num_heads, eps=1e-6)
    hidden = _rand(1, seq, dim, scale=0.05)
    context = _rand(1, ctx, dim, scale=0.05)
    temb = _rand(1, 6, dim, scale=0.05)
    cos = _rand(seq, head_dim)
    sin = _rand(seq, head_dim)

    eager = block(hidden, context, temb, (cos, sin))
    mx.eval(eager)
    compiled = mx.compile(lambda h, e, t, c, s: block(h, e, t, (c, s)))(hidden, context, temb, cos, sin)
    mx.eval(compiled)

    np.testing.assert_array_equal(np.array(eager), np.array(compiled))
