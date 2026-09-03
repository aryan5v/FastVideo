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
