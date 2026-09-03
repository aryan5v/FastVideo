# SPDX-License-Identifier: Apache-2.0
"""MLX hybrid MiniMax H3 vs the torch hybrid module on a tiny random checkpoint."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 hybrid parity tests")
torch = pytest.importorskip("torch", reason="PyTorch supplies the independent H3 reference")

from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402
    build_hf_config,
    build_inputs,
    build_tiny_hybrid_config,
    build_torch_model,
    mlx_dit_from_torch_model,
    mlx_output,
    torch_hybrid_layout,
    torch_reference_output,
)
from fastvideo.models.dits.minimax_h3_hybrid.linear import factor_delta, frame_statistics  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_hybrid import _factor_delta  # noqa: E402

FP32_ATOL = 2e-4
FP32_RTOL = 2e-4


def test_mlx_hybrid_full_cover_matches_torch(distributed_setup) -> None:
    config = build_tiny_hybrid_config(hybrid_enable_softmax_gate=False)
    model = build_torch_model(config)
    layout, torch_inputs, mlx_inputs = build_inputs()
    torch_inputs["hybrid_layout"] = torch_hybrid_layout(layout)
    torch_video, torch_audio = torch_reference_output(model, torch_inputs)
    dit = mlx_dit_from_torch_model(model, build_hf_config(config))
    mlx_video, mlx_audio = mlx_output(dit, layout, mlx_inputs)
    np.testing.assert_allclose(mlx_video, torch_video, atol=FP32_ATOL, rtol=FP32_RTOL)
    np.testing.assert_allclose(mlx_audio, torch_audio, atol=FP32_ATOL, rtol=FP32_RTOL)


def test_mlx_factor_delta_cpu_inv_matches_torch_cholesky() -> None:
    import mlx.core as mx

    torch.manual_seed(0)
    frames, heads, tokens, dim = 3, 2, 4, 8
    keys = torch.randn(frames, heads, tokens, dim)
    values = torch.randn(frames, heads, tokens, dim)
    beta = torch.rand(frames, heads, tokens)
    matrix_a, matrix_b = frame_statistics(keys, values, beta, a_fp32=True)
    alpha = torch.rand(frames, heads, dim).clamp(min=0.1, max=0.9)
    torch_transitions, torch_injections = factor_delta("vdn_solve", alpha, matrix_a, matrix_b, tokens)
    mlx_transitions, mlx_injections = _factor_delta(
        "vdn_solve",
        mx.array(alpha.numpy()),
        mx.array(matrix_a.numpy()),
        mx.array(matrix_b.numpy()),
        tokens,
    )
    mx.eval(mlx_transitions, mlx_injections)
    np.testing.assert_allclose(np.asarray(mlx_transitions), torch_transitions.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(mlx_injections), torch_injections.numpy(), atol=1e-4, rtol=1e-4)
