# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence geometry for MiniMax H3 hybrid window + linear attention.

The softmax branch attends inside a VAE-chunk-aligned frame window. The linear
branch summarises everything outside that window. Layout is derived from the
existing packed indices the H3 pipeline already builds — not a second packing
scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout


@dataclass(frozen=True)
class HybridSequenceLayout:
    """Frame geometry for one packed H3 document.

    Video tokens that the DiT denoises are a contiguous generated tail after any
    condition-video rows. Globals (text + audio + condition video) stay dense in
    the softmax branch in both directions.
    """

    seq_len: int
    video_start: int
    video_end: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int
    text_end: int

    def __post_init__(self) -> None:
        video_tokens = self.video_end - self.video_start
        if self.num_frames <= 0 or self.tokens_per_frame <= 0:
            raise ValueError("hybrid layout needs a positive frame grid.")
        if video_tokens != self.num_frames * self.tokens_per_frame:
            raise ValueError(
                f"generated video rows {video_tokens} != {self.num_frames} frames "
                f"* {self.tokens_per_frame} tokens/frame.")
        if self.frame_height * self.frame_width != self.tokens_per_frame:
            raise ValueError("frame_height * frame_width must equal tokens_per_frame.")

    @property
    def text_range(self) -> tuple[int, int]:
        return self.text_start, self.text_end

    def global_index(self, device: torch.device | str) -> torch.Tensor:
        """Every packed row that is not a generated-video token."""
        import torch

        rows = torch.arange(self.seq_len, device=device)
        return rows[(rows < self.video_start) | (rows >= self.video_end)]


def window_bounds(num_frames: int, radius: int, chunk: int = 0) -> list[tuple[int, int]]:
    """Inclusive per-frame softmax window ``[lo, hi]``, unclamped.

    ``chunk == 0`` is a centered frame window ``|t_q - t_k| <= radius``.
    ``chunk == K`` aligns the window to whole VAE latent chunks of length K so
    every query sees complete neighbouring chunks (H3 encodes 5 latent frames
    per 17-pixel chunk).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    if chunk < 0:
        raise ValueError(f"chunk must be >= 0, got {chunk}.")
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1) for t in range(num_frames)]


def windows_cover_all_frames(bounds: list[tuple[int, int]], num_frames: int) -> bool:
    """True when every query frame's window already spans the whole clip."""
    last = num_frames - 1
    return all(lo <= 0 and hi >= last for lo, hi in bounds)


def hybrid_layout_from_packed(
    packed: MiniMaxH3PackedLayout,
    patch_size: tuple[int, int, int],
) -> HybridSequenceLayout:
    """Build hybrid geometry from the pipeline's packed layout."""
    patch_t, patch_h, patch_w = (int(v) for v in patch_size)
    if packed.num_video_latent_frames % patch_t:
        raise ValueError("num_video_latent_frames must be divisible by patch_t.")
    if packed.latent_height % patch_h or packed.latent_width % patch_w:
        raise ValueError("latent spatial size must be divisible by the spatial patch.")
    num_frames = packed.num_video_latent_frames // patch_t
    frame_height = packed.latent_height // patch_h
    frame_width = packed.latent_width // patch_w
    tokens_per_frame = frame_height * frame_width
    generated_video = tokens_per_frame * num_frames
    video_end = packed.sequence_length
    video_start = video_end - generated_video
    if video_start < 0:
        raise ValueError("packed sequence is shorter than the generated video grid.")
    text_start = int(packed.text_indices.min().item()) if packed.text_indices.numel() else 0
    text_end = int(packed.text_indices.max().item()) + 1 if packed.text_indices.numel() else 0
    return HybridSequenceLayout(
        seq_len=packed.sequence_length,
        video_start=video_start,
        video_end=video_end,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=frame_height,
        frame_width=frame_width,
        text_start=text_start,
        text_end=text_end,
    )
