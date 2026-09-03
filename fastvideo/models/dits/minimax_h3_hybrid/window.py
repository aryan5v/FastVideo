# SPDX-License-Identifier: Apache-2.0
"""Chunk-aligned window softmax as a union of dense attentions.

The c1 mask is a handful of dense rectangles (global rows vs the full sequence,
plus per-chunk video windows). Running those as SDPA/Flash is the FastVideo
path; FlexAttention is intentionally not used. Sequence-parallel ranks must
all-gather QKV before calling into this module (see HybridAttention).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3_hybrid.layout import HybridSequenceLayout

_ANCHOR_MODES = ("none", "columns", "rows", "both")


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """query/key/value: [rows, H, d] -> [rows, H, d]."""
    attended = F.scaled_dot_product_attention(
        query.permute(1, 0, 2).unsqueeze(0),
        key.permute(1, 0, 2).unsqueeze(0),
        value.permute(1, 0, 2).unsqueeze(0),
        scale=scale,
        dropout_p=0.0,
        is_causal=False,
    )
    return attended.squeeze(0).permute(1, 0, 2)


def _clamp_bounds(bounds: list[tuple[int, int]], num_frames: int) -> list[tuple[int, int]]:
    last = num_frames - 1
    return [(max(lo, 0), min(hi, last)) for lo, hi in bounds]


def window_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: HybridSequenceLayout,
    bounds: list[tuple[int, int]],
    scale: float,
    anchor_frames: str = "both",
) -> torch.Tensor:
    """Windowed softmax over one packed document.

    Global (text/audio/condition) queries attend to the full sequence. Video
    queries attend to their chunk window plus every global key. Anchor frames
    0 and F-1 can be dense as columns, rows, both, or neither.
    """
    if anchor_frames not in _ANCHOR_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {_ANCHOR_MODES}.")
    heads, head_dim = query.shape[1], query.shape[2]
    out = torch.empty_like(query)
    global_idx = layout.global_index(query.device)
    clamped = _clamp_bounds(bounds, layout.num_frames)

    if global_idx.numel():
        out[global_idx] = _sdpa(query[global_idx], key, value, scale)

    video_start, video_end = layout.video_start, layout.video_end
    frame_shape = (layout.num_frames, layout.tokens_per_frame, heads, head_dim)
    video_query = query[video_start:video_end].reshape(frame_shape)
    video_key = key[video_start:video_end].reshape(frame_shape)
    video_value = value[video_start:video_end].reshape(frame_shape)
    global_key = key[global_idx] if global_idx.numel() else key.new_empty(0, heads, head_dim)
    global_value = value[global_idx] if global_idx.numel() else value.new_empty(0, heads, head_dim)

    dense_row_frames = {0, layout.num_frames - 1} if anchor_frames in ("rows", "both") else set()
    dense_col_frames = {0, layout.num_frames - 1} if anchor_frames in ("columns", "both") else set()

    groups: dict[tuple[int, ...], list[int]] = {}
    for frame, (lo, hi) in enumerate(clamped):
        if frame in dense_row_frames:
            key_frames = tuple(range(layout.num_frames))
        else:
            kept = set(range(lo, hi + 1)) | dense_col_frames
            key_frames = tuple(sorted(kept))
        groups.setdefault(key_frames, []).append(frame)

    for key_frames, query_frames in groups.items():
        q_rows = video_query[query_frames].reshape(-1, heads, head_dim)
        k_parts = [global_key] if global_key.numel() else []
        v_parts = [global_value] if global_value.numel() else []
        if key_frames:
            k_parts.append(video_key[list(key_frames)].reshape(-1, heads, head_dim))
            v_parts.append(video_value[list(key_frames)].reshape(-1, heads, head_dim))
        k_rows = torch.cat(k_parts, dim=0) if k_parts else global_key
        v_rows = torch.cat(v_parts, dim=0) if v_parts else global_value
        attended = _sdpa(q_rows, k_rows, v_rows, scale)
        out_rows = attended.reshape(len(query_frames), layout.tokens_per_frame, heads, head_dim)
        for local, frame in enumerate(query_frames):
            start = video_start + frame * layout.tokens_per_frame
            out[start:start + layout.tokens_per_frame] = out_rows[local]
    return out
