# SPDX-License-Identifier: Apache-2.0
"""Chunk-aligned window softmax as a union of dense attentions.

The c1 mask is a handful of dense rectangles (global rows vs the full sequence,
plus per-chunk video windows). Running those as SDPA/Flash is the FastVideo
path; FlexAttention is intentionally not used. Sequence-parallel ranks must
all-gather QKV before calling into this module (see HybridAttention).

The rectangle decomposition is a property of the *request* -- frame count, chunk
size, radius, anchor mode -- and of nothing that changes between layers.  It used
to be rebuilt on every call, per layer, as Python dictionaries plus one slice
assignment per query frame.  :func:`build_window_plan` computes the rectangle
plan, its index tensors and its output rows once per layout and caches it; the
per-layer call then gathers and scatters with precomputed indices instead of
rebuilding them.  The arithmetic -- which keys each query row sees, in what
order -- is unchanged, so results are identical to the previous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


@dataclass(frozen=True)
class WindowGroup:
    """One dense rectangle: which query rows, which keys, and where they land.

    Indices are into the packed key/value buffer ``cat([global_rows, video_rows])``
    and into the ``[num_frames * tokens_per_frame, H, d]`` video query buffer.
    """

    key_index: torch.Tensor    # keys this rectangle attends to
    query_index: torch.Tensor  # query rows in the rectangle
    out_index: torch.Tensor    # rows of the video output buffer these land in


@dataclass(frozen=True)
class WindowPlan:
    """Request-static window metadata, reusable by every layer of a request."""

    global_index: torch.Tensor
    groups: tuple[WindowGroup, ...]
    video_start: int
    video_end: int
    tokens_per_frame: int
    num_frames: int
    heads: int
    head_dim: int

    def retain(self) -> "WindowPlan":
        """Keep the plan alive across layers without rebuilding it."""
        return self


def _device_free_key(num_frames: int, radius: int, chunk: int, anchor_frames: str) -> tuple:
    return (num_frames, radius, chunk, anchor_frames)


def build_window_plan(
    layout: HybridSequenceLayout,
    bounds: list[tuple[int, int]],
    anchor_frames: str,
    device: torch.device,
    heads: int,
    head_dim: int,
) -> WindowPlan:
    """Rectangle decomposition + index tensors for one request shape.

    Global (text/audio/condition) queries attend the whole sequence once as a
    single dense rectangle.  Video queries are grouped by the *set* of frames
    they attend; because the window is chunk-aligned, frames inside one chunk
    share a key set, so the group count is the chunk count rather than the frame
    count.
    """
    if anchor_frames not in _ANCHOR_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {_ANCHOR_MODES}.")

    num_frames = layout.num_frames
    per_frame = layout.tokens_per_frame
    global_index = layout.global_index(device)
    num_global = int(global_index.numel())
    clamped = _clamp_bounds(bounds, num_frames)

    dense_row_frames = {0, num_frames - 1} if anchor_frames in ("rows", "both") else set()
    dense_col_frames = {0, num_frames - 1} if anchor_frames in ("columns", "both") else set()

    # Key buffer layout: [global rows][video rows], so video frame f starts at
    # num_global + f * per_frame.
    frame_span = torch.arange(per_frame, device=device)

    groups_by_key: dict[tuple[int, ...], list[int]] = {}
    for frame, (lo, hi) in enumerate(clamped):
        if frame in dense_row_frames:
            key_frames = tuple(range(num_frames))
        else:
            kept = set(range(lo, hi + 1)) | dense_col_frames
            key_frames = tuple(sorted(kept))
        groups_by_key.setdefault(key_frames, []).append(frame)

    groups: list[WindowGroup] = []
    for key_frames, query_frames in groups_by_key.items():
        parts = []
        if num_global:
            parts.append(torch.arange(num_global, device=device))
        for frame in key_frames:
            parts.append(num_global + frame * per_frame + frame_span)
        key_index = torch.cat(parts) if parts else torch.arange(0, device=device)

        q_parts = [frame * per_frame + frame_span for frame in query_frames]
        query_index = torch.cat(q_parts)
        # Video outputs land at layout.video_start + frame * per_frame + token.
        out_index = torch.cat([layout.video_start + part for part in q_parts])
        groups.append(WindowGroup(key_index=key_index, query_index=query_index, out_index=out_index))

    return WindowPlan(
        global_index=global_index,
        groups=tuple(groups),
        video_start=layout.video_start,
        video_end=layout.video_end,
        tokens_per_frame=per_frame,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
    )


# The plan depends only on the request shape, so one entry per shape is enough
# and it survives across all 50 layers of a denoise.  Bounded so a long-lived
# process cannot accumulate plans for unbounded shapes.
@lru_cache(maxsize=32)
def _cached_plan(
    num_frames: int,
    seq_len: int,
    video_start: int,
    video_end: int,
    tokens_per_frame: int,
    frame_height: int,
    frame_width: int,
    text_start: int,
    text_end: int,
    radius: int,
    chunk: int,
    anchor_frames: str,
    device_index: int,
    heads: int,
    head_dim: int,
) -> WindowPlan:
    layout = HybridSequenceLayout(
        seq_len=seq_len,
        video_start=video_start,
        video_end=video_end,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=frame_height,
        frame_width=frame_width,
        text_start=text_start,
        text_end=text_end,
    )
    from fastvideo.models.dits.minimax_h3_hybrid.layout import window_bounds

    device = torch.device("cuda", device_index) if device_index >= 0 else torch.device("cpu")
    bounds = window_bounds(num_frames, radius, chunk)
    return build_window_plan(layout, bounds, anchor_frames, device, heads, head_dim)


def window_plan_for(
    layout: HybridSequenceLayout,
    radius: int,
    chunk: int,
    anchor_frames: str,
    query: torch.Tensor,
) -> WindowPlan:
    """Fetch (or build) the cached plan for this request shape.

    Keyed on the packed geometry plus the window parameters, so every layer of a
    request hits the same entry and the decomposition is computed once.
    """
    device_index = query.device.index if query.device.type == "cuda" else -1
    return _cached_plan(
        layout.num_frames,
        layout.seq_len,
        layout.video_start,
        layout.video_end,
        layout.tokens_per_frame,
        layout.frame_height,
        layout.frame_width,
        layout.text_start,
        layout.text_end,
        radius,
        chunk,
        anchor_frames,
        device_index,
        query.shape[1],
        query.shape[2],
    )


def window_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: HybridSequenceLayout,
    bounds: list[tuple[int, int]],
    scale: float,
    anchor_frames: str = "both",
    plan: WindowPlan | None = None,
) -> torch.Tensor:
    """Windowed softmax over one packed document.

    Global (text/audio/condition) queries attend to the full sequence. Video
    queries attend to their chunk window plus every global key. Anchor frames
    0 and F-1 can be dense as columns, rows, both, or neither.

    ``plan`` carries the precomputed rectangle decomposition; when omitted the
    eager decomposition is rebuilt so the function stays usable stand-alone.
    """
    if anchor_frames not in _ANCHOR_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {_ANCHOR_MODES}.")
    heads, head_dim = query.shape[1], query.shape[2]
    if plan is None:
        plan = build_window_plan(layout, bounds, anchor_frames, query.device, heads, head_dim)

    out = torch.empty_like(query)
    global_index = plan.global_index

    if global_index.numel():
        # CUDA autocast may return BF16 SDPA output for FP32 residual-stream
        # inputs. Preserve the caller-visible query dtype while satisfying
        # index_put's exact source/destination dtype requirement.
        gathered = query.index_select(0, global_index)
        attended = _sdpa(gathered, key, value, scale).to(out.dtype)
        out.index_copy_(0, global_index, attended)

    per_frame = plan.tokens_per_frame
    video_start, video_end = plan.video_start, plan.video_end
    video_query = query[video_start:video_end].reshape(-1, heads, head_dim)
    video_key = key[video_start:video_end].reshape(-1, heads, head_dim)
    video_value = value[video_start:video_end].reshape(-1, heads, head_dim)

    # One packed K/V buffer per call: [global rows][video rows]. Building it once
    # lets every rectangle gather with a single index_select.
    if global_index.numel():
        k_buffer = torch.cat([key.index_select(0, global_index), video_key], dim=0)
        v_buffer = torch.cat([value.index_select(0, global_index), video_value], dim=0)
    else:
        k_buffer, v_buffer = video_key, video_value

    video_out = torch.empty_like(video_query)
    for group in plan.groups:
        q_rows = video_query.index_select(0, group.query_index)
        k_rows = k_buffer.index_select(0, group.key_index)
        v_rows = v_buffer.index_select(0, group.key_index)
        attended = _sdpa(q_rows, k_rows, v_rows, scale).to(video_out.dtype)
        video_out.index_copy_(0, group.query_index, attended)

    # ``out`` is [rows, H, d]; the video block is the same rank-3 shape, so it
    # assigns directly.  (Assigning a rank-4 view here silently broadcasts the
    # trailing head_dim instead of erroring, which is how this was caught.)
    out[video_start:video_end] = video_out
    return out
