# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native MiniMax H3 hybrid attention (window softmax + linear scan)."""

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
    BidirectionalLinearBranch,
    DELTA_RULES,
    factor_delta,
    frame_statistics,
    gather_linear_state,
    run_scans,
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
    "window_bounds",
    "window_softmax",
    "windows_cover_all_frames",
]
