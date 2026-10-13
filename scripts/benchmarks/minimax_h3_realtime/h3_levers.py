# SPDX-License-Identifier: Apache-2.0
"""Training-free speed levers for a loaded MiniMax-H3 DiT (opt-in, in-process).

``install_levers(transformer, ...)`` swaps the transformer's block list for thin
wrappers and returns a handle whose ``remove()`` restores the original blocks.
Nothing changes unless this is called. Levers:

* ``skip``: block indices replaced by the identity.
* ``compress``: run blocks ``[start, end)`` on fewer video tokens. ``mode``
  ``"subsample"`` keeps every second token along height and width
  (LynnReal-style stride 2); ``"pair_x"`` averages horizontally adjacent token
  pairs (h3.c-style pairing). Text and audio rows are untouched, kept tokens keep
  their original RoPE coordinates, and on exit every video token receives the
  update of its nearest kept token: ``H_out = H + upsample(F(P H) - P H)``.
  Under VIDEO_SPARSE_ATTN_H3 the compressed blocks get their own block-sparse
  metadata for the smaller video grid.
* ``score``: accumulate ``1 - cos(block input, block output)`` over video rows
  per block (ShortGPT-style residual-change score).

The video token grid ``(T, H, W)`` (latent frames, latent height / 2, latent
width / 2) must be set on the handle with ``set_grid`` before a forward when
``compress`` is used. Video rows must be the tail of the packed sequence
(T2VA/FL2VA/Ref2VA packing all place the target video last).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from fastvideo.forward_context import get_forward_context

VIDEO_TAG = 0
MODALITY_NUM = 3


@dataclass
class CompressSpec:
    start: int
    end: int
    mode: str = "subsample"  # "subsample" | "pair_x"


@dataclass
class _State:
    skip: frozenset[int]
    compress: CompressSpec | None
    score: bool
    grid: tuple[int, int, int] | None = None
    scores: dict[int, list[float]] = field(default_factory=dict)
    # Per-forward compression context.
    full: torch.Tensor | None = None
    compressed_input: torch.Tensor | None = None
    adaln: torch.Tensor | None = None
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None
    seq_len: int = 0
    num_prefix: int = 0
    sub_grid: tuple[int, int, int] | None = None
    saved_metadata: Any = None
    builder: Any = None


def _video_grid_ops(mode: str, grid: tuple[int, int, int]) -> tuple[tuple[int, int, int], Any, Any]:
    """Compressed grid, compress(video [B, N, C]) and expand(delta [B, n, C]) for one mode."""
    frames, height, width = grid

    def as_grid(rows: torch.Tensor) -> torch.Tensor:
        return rows.reshape(rows.shape[0], frames, height, width, rows.shape[-1])

    if mode == "subsample":
        sub = (frames, (height + 1) // 2, (width + 1) // 2)

        def compress(rows: torch.Tensor) -> torch.Tensor:
            return as_grid(rows)[:, :, ::2, ::2].reshape(rows.shape[0], -1, rows.shape[-1])

        def expand(delta: torch.Tensor) -> torch.Tensor:
            d = delta.reshape(delta.shape[0], *sub, delta.shape[-1])
            d = d.repeat_interleave(2, dim=2)[:, :, :height].repeat_interleave(2, dim=3)[:, :, :, :width]
            return d.reshape(delta.shape[0], -1, delta.shape[-1])

        return sub, compress, expand
    if mode == "pair_x":
        sub = (frames, height, (width + 1) // 2)

        def compress(rows: torch.Tensor) -> torch.Tensor:
            g = as_grid(rows)
            left = g[:, :, :, ::2]
            right = g[:, :, :, 1::2]
            if right.shape[3] < left.shape[3]:
                right = torch.cat((right, left[:, :, :, -1:]), dim=3)
            return ((left.float() + right.float()) * 0.5).to(rows.dtype).reshape(rows.shape[0], -1, rows.shape[-1])

        def expand(delta: torch.Tensor) -> torch.Tensor:
            d = delta.reshape(delta.shape[0], *sub, delta.shape[-1]).repeat_interleave(2, dim=3)[:, :, :, :width]
            return d.reshape(delta.shape[0], -1, delta.shape[-1])

        return sub, compress, expand
    raise ValueError(f"unknown compression mode {mode!r}")


def _keep_rows(mode: str, grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Row index (within the video rows) whose RoPE coordinates and AdaLN rows a compressed token keeps."""
    frames, height, width = grid
    index = torch.arange(frames * height * width, device=device).reshape(frames, height, width)
    if mode == "subsample":
        return index[:, ::2, ::2].reshape(-1)
    return index[:, :, ::2].reshape(-1)


class _LeverBlock(nn.Module):

    def __init__(self, inner: nn.Module, index: int, state: _State, last_index: int) -> None:
        super().__init__()
        self.inner = inner
        self.index = index
        self.state = state
        self.last_index = last_index

    def _enter(self, hidden: torch.Tensor, adaln: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        state = self.state
        spec = state.compress
        assert spec is not None
        if state.grid is None:
            raise RuntimeError("set_grid() must be called before a compressed forward")
        num_video = state.grid[0] * state.grid[1] * state.grid[2]
        num_prefix = hidden.shape[1] - num_video
        if num_prefix < 0 or bool((adaln[num_prefix:] % MODALITY_NUM != VIDEO_TAG).any()):
            raise ValueError("video rows must be the tail of the packed sequence")
        sub_grid, compress, _ = _video_grid_ops(spec.mode, state.grid)
        keep = _keep_rows(spec.mode, state.grid, hidden.device) + num_prefix
        rows = torch.cat((torch.arange(num_prefix, device=hidden.device), keep))
        state.full = hidden
        state.compressed_input = torch.cat((hidden[:, :num_prefix], compress(hidden[:, num_prefix:])), dim=1)
        state.adaln = adaln.index_select(0, rows)
        state.rotary = (rotary[0].index_select(0, rows), rotary[1].index_select(0, rows))
        state.seq_len = int(rows.numel())
        state.num_prefix = num_prefix
        state.sub_grid = sub_grid
        self._swap_metadata(num_prefix, adaln)

    def _swap_metadata(self, num_prefix: int, adaln: torch.Tensor) -> None:
        state = self.state
        context = get_forward_context()
        metadata = context.attn_metadata
        state.saved_metadata = metadata
        if metadata is None or type(metadata).__name__ != "MiniMaxH3VSAMetadata":
            return
        from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder

        if state.builder is None:
            state.builder = MiniMaxH3VSAMetadataBuilder()
        prefix_tags = adaln[:num_prefix] % MODALITY_NUM
        num_text = int((prefix_tags != 2).sum())
        num_audio = num_prefix - num_text
        frames, height, width = state.sub_grid
        context.attn_metadata = state.builder.build(
            current_timestep=metadata.current_timestep,
            patch_size=(1, 2, 2),
            VSA_sparsity=metadata.VSA_sparsity,
            packed_segments=(num_text, num_audio, (frames, height * 2, width * 2)),
            device=adaln.device,
            exempt=metadata.exempt,
            dense_layers=tuple(metadata.dense_layers),
            tile_size=metadata.tile_elems,
        )

    def _exit(self, output: torch.Tensor) -> torch.Tensor:
        state = self.state
        spec = state.compress
        assert spec is not None and state.full is not None and state.compressed_input is not None
        _, _, expand = _video_grid_ops(spec.mode, state.grid)
        n = state.num_prefix
        delta = expand(output[:, n:] - state.compressed_input[:, n:])
        restored = torch.cat((output[:, :n], state.full[:, n:] + delta.to(state.full.dtype)), dim=1)
        get_forward_context().attn_metadata = state.saved_metadata
        state.full = state.compressed_input = None
        return restored

    def _score(self, before: torch.Tensor, after: torch.Tensor, adaln: torch.Tensor) -> None:
        video = (adaln % MODALITY_NUM) == VIDEO_TAG
        a = before[0, video].float()
        b = after[0, video].float()
        change = 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
        self.state.scores.setdefault(self.index, []).append(float(change))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, adaln_indices: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor], original_seq_len: int) -> torch.Tensor:
        state = self.state
        spec = state.compress
        compressed = spec is not None and spec.start <= self.index < spec.end
        if compressed and self.index == spec.start:
            self._enter(hidden_states, adaln_indices, rotary_emb)
            hidden_states = state.compressed_input
        if compressed:
            adaln, rotary, seq_len = state.adaln, state.rotary, state.seq_len
        else:
            adaln, rotary, seq_len = adaln_indices, rotary_emb, original_seq_len
        if self.index in state.skip:
            output = hidden_states
        else:
            output = self.inner(hidden_states, temb, adaln, rotary, seq_len)
            if state.score and not compressed:
                self._score(hidden_states, output, adaln)
        if compressed and self.index == min(spec.end, self.last_index + 1) - 1:
            output = self._exit(output)
        return output


class LeverHandle:

    def __init__(self, transformer: nn.Module, original: nn.ModuleList, state: _State) -> None:
        self.transformer = transformer
        self.original = original
        self.state = state

    def set_grid(self, frames: int, height: int, width: int) -> None:
        self.state.grid = (frames, height, width)

    @property
    def scores(self) -> dict[int, list[float]]:
        return self.state.scores

    def remove(self) -> None:
        self.transformer.transformer_blocks = self.original


def install_levers(transformer: nn.Module,
                   *,
                   skip: tuple[int, ...] = (),
                   compress: CompressSpec | None = None,
                   score: bool = False) -> LeverHandle:
    original = transformer.transformer_blocks
    if isinstance(next(iter(original)), _LeverBlock):
        raise RuntimeError("levers are already installed")
    state = _State(skip=frozenset(skip), compress=compress, score=score)
    last = len(original) - 1
    transformer.transformer_blocks = nn.ModuleList(
        [_LeverBlock(block, index, state, last) for index, block in enumerate(original)])
    return LeverHandle(transformer, original, state)
