# SPDX-License-Identifier: Apache-2.0
"""Experimental Apple MLX runtime helpers.

This package is intentionally small for now. It exists to grow the Apple-native
FastWan path in measurable steps: shape planning, primitive benchmarks, then
Wan block parity, then full DiT/runtime support.
"""

from fastvideo.mlx_runtime.fastwan import (
    FastWanShape,
    MLXWanDiT,
    MLXWanTransformerBlock,
    fastwan_shape,
    fastwan_shape_from_config,
    mlx_dit_from_diffusers_safetensors,
    mlx_block_weights_from_torch,
    mlx_block_weights_from_diffusers_safetensors,
)

__all__ = [
    "FastWanShape",
    "MLXWanDiT",
    "MLXWanTransformerBlock",
    "fastwan_shape",
    "fastwan_shape_from_config",
    "mlx_dit_from_diffusers_safetensors",
    "mlx_block_weights_from_diffusers_safetensors",
    "mlx_block_weights_from_torch",
]
