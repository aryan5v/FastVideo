# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native MiniMax H3 hybrid attention (window softmax + linear scan).

Eager imports stay limited to symbols that do not pull PyTorch. ``layout.py`` is
torch-free at module level so MLX can reuse window geometry without loading
CUDA. Attention, linear, window, and checkpoint helpers are resolved lazily.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.models.dits.minimax_h3_hybrid.attention import HybridAttention
    from fastvideo.models.dits.minimax_h3_hybrid.checkpoint import (
        hybrid_arch_fields_from_spec,
        merge_lora_pairs,
        remap_vdn_key,
    )
    from fastvideo.models.dits.minimax_h3_hybrid.layout import (
        HybridSequenceLayout,
        hybrid_layout_from_packed,
        window_bounds,
        windows_cover_all_frames,
    )
    from fastvideo.models.dits.minimax_h3_hybrid.linear import (
        DELTA_RULES,
        BidirectionalLinearBranch,
        factor_delta,
        frame_statistics,
        gather_linear_state,
        run_scans,
        scaled_exponential_write_strength,
    )
    from fastvideo.models.dits.minimax_h3_hybrid.window import window_softmax

__all__ = [
    "DELTA_RULES",
    "BidirectionalLinearBranch",
    "HybridAttention",
    "HybridSequenceLayout",
    "factor_delta",
    "frame_statistics",
    "gather_linear_state",
    "hybrid_arch_fields_from_spec",
    "hybrid_layout_from_packed",
    "merge_lora_pairs",
    "remap_vdn_key",
    "run_scans",
    "scaled_exponential_write_strength",
    "window_bounds",
    "window_softmax",
    "windows_cover_all_frames",
]

_LAZY_ATTRS = {
    "HybridSequenceLayout": "fastvideo.models.dits.minimax_h3_hybrid.layout",
    "hybrid_layout_from_packed": "fastvideo.models.dits.minimax_h3_hybrid.layout",
    "window_bounds": "fastvideo.models.dits.minimax_h3_hybrid.layout",
    "windows_cover_all_frames": "fastvideo.models.dits.minimax_h3_hybrid.layout",
    "HybridAttention": "fastvideo.models.dits.minimax_h3_hybrid.attention",
    "DELTA_RULES": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "BidirectionalLinearBranch": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "factor_delta": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "frame_statistics": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "gather_linear_state": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "run_scans": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "scaled_exponential_write_strength": "fastvideo.models.dits.minimax_h3_hybrid.linear",
    "hybrid_arch_fields_from_spec": "fastvideo.models.dits.minimax_h3_hybrid.checkpoint",
    "merge_lora_pairs": "fastvideo.models.dits.minimax_h3_hybrid.checkpoint",
    "remap_vdn_key": "fastvideo.models.dits.minimax_h3_hybrid.checkpoint",
    "window_softmax": "fastvideo.models.dits.minimax_h3_hybrid.window",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
