# SPDX-License-Identifier: Apache-2.0
"""Geometry and decomposed window softmax for MiniMax H3 hybrid attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3_hybrid.layout import (
    HybridSequenceLayout,
    window_bounds,
    windows_cover_all_frames,
)
from fastvideo.models.dits.minimax_h3_hybrid.window import window_softmax


def test_window_bounds_centered_frame_window() -> None:
    assert window_bounds(5, radius=1, chunk=0) == [(-1, 1), (0, 2), (1, 3), (2, 4), (3, 5)]


def test_window_bounds_chunk_aligned_c1() -> None:
    # radius 1, chunk 5: each query sees its VAE chunk plus one neighbour.
    bounds = window_bounds(12, radius=1, chunk=5)
    assert bounds[0] == (-5, 9)
    assert bounds[4] == (-5, 9)
    assert bounds[5] == (0, 14)
    assert windows_cover_all_frames(bounds, 12) is False
    assert windows_cover_all_frames(window_bounds(4, radius=1, chunk=5), 4) is True


def _layout(num_frames: int = 4, tokens_per_frame: int = 2, text: int = 3, audio: int = 2) -> HybridSequenceLayout:
    video = num_frames * tokens_per_frame
    seq = text + audio + video
    return HybridSequenceLayout(
        seq_len=seq,
        video_start=text + audio,
        video_end=seq,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=1,
        frame_width=tokens_per_frame,
        text_start=0,
        text_end=text,
    )


def test_full_cover_window_softmax_matches_dense_sdpa() -> None:
    torch.manual_seed(0)
    layout = _layout(num_frames=3, tokens_per_frame=2)
    heads, dim = 2, 4
    query = torch.randn(layout.seq_len, heads, dim)
    key = torch.randn(layout.seq_len, heads, dim)
    value = torch.randn(layout.seq_len, heads, dim)
    scale = dim**-0.5
    bounds = window_bounds(layout.num_frames, radius=layout.num_frames, chunk=0)
    assert windows_cover_all_frames(bounds, layout.num_frames)
    # Anchors "none": every video query already sees every frame, plus globals.
    actual = window_softmax(query, key, value, layout, bounds, scale, anchor_frames="none")
    expected = F.scaled_dot_product_attention(
        query.permute(1, 0, 2).unsqueeze(0),
        key.permute(1, 0, 2).unsqueeze(0),
        value.permute(1, 0, 2).unsqueeze(0),
        scale=scale,
        dropout_p=0.0,
        is_causal=False,
    ).squeeze(0).permute(1, 0, 2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_radius_zero_window_keeps_globals_and_own_frame() -> None:
    torch.manual_seed(1)
    layout = _layout(num_frames=4, tokens_per_frame=1)
    heads, dim = 1, 4
    query = torch.randn(layout.seq_len, heads, dim)
    key = torch.randn(layout.seq_len, heads, dim)
    value = torch.randn(layout.seq_len, heads, dim)
    scale = dim**-0.5
    bounds = window_bounds(layout.num_frames, radius=0, chunk=0)
    out = window_softmax(query, key, value, layout, bounds, scale, anchor_frames="none")
    # Frame 1 (second generated token) may only see globals + its own frame.
    frame = 1
    token = layout.video_start + frame
    global_idx = list(range(layout.video_start)) + list(range(layout.video_end, layout.seq_len))
    allowed = global_idx + [token]
    q = query[token:token + 1].permute(1, 0, 2).unsqueeze(0)
    k = key[allowed].permute(1, 0, 2).unsqueeze(0)
    v = value[allowed].permute(1, 0, 2).unsqueeze(0)
    expected = F.scaled_dot_product_attention(q, k, v, scale=scale, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(out[token], expected.squeeze(0).permute(1, 0, 2)[0], atol=1e-5, rtol=1e-5)
