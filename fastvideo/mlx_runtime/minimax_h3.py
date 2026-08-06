# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 joint audio-video DiT for the Apple Silicon MLX runtime.

A faithful MLX port of the upstream CUDA reference:

- DiT: ``fastvideo/models/dits/minimax_h3.py`` (merged upstream in #1674).
  Single-stream packed transformer, per-head qk RMSNorm, 3-axis MM-RoPE
  (96 of 128 head dims rotated), row-indexed AdaLN keyed by
  ``(timestep, modality)``, dual video/audio output heads, 2-block text
  token refiner.
- Scheduler: ``fastvideo/models/schedulers/scheduling_minimax_h3.py``.
  Rectified-flow Euler with H3's clean-time convention: ``t = 1 - sigma``,
  data-ward velocity (``x0 = x_t + sigma * v``), exponential-shift sigma
  grid (video shift 12.0, audio shift 3.0), fp32 Euler blend.
- Packing: ``fastvideo/pipelines/basic/minimax_h3/packing.py``.
  ``[text | condition | audio | video]`` rows, float64 position grids.

Mac-specific levers (per docs/design/fasth3_roadmap.md):

- **AdaLN precompute cache.** ~40% of H3's parameters live in per-block
  AdaLN projections whose output depends only on ``(timestep, modality)``.
  For a fixed step schedule the full set of timesteps is known at load time,
  so :meth:`MLXMiniMaxH3DiT.precompute_adaln` evaluates every modulation
  table once and (optionally) drops the projection weights — that is what
  makes a 14B student workable on a 24 GB Mac, and it removes the per-step
  260M-param matmul per block as well.
- **Affine int8 quantization** of attention/FFN matrices only (group 64),
  matching the parity-tested QAT deploy grid. Modulation, embeddings, norms,
  and the input/output projections stay in higher precision. mxfp4/nvfp4 are
  deliberately not offered here: measured bad quality and zero speed on Metal
  (docs/design/m5_survey_results.md).

Checkpoint layout: the released H3 checkpoint uses the diffusers reference
module names 1:1 (``transformer_blocks.{i}.attn.to_out.0.weight``,
``ff.net.0.proj.weight``, ``time_embedder.linear_1.weight``, ...). This
module keeps those names as the MLX weight keys — no renaming contract.
``proj_in`` / ``audio_proj_in`` / ``time_embedder`` / ``proj_out`` /
``audio_proj_out`` are fp32 in the release and are kept fp32 here.

Nothing in this file requires the CUDA stack; it imports ``fastvideo.logger``
only. Parity tests against the torch reference live in
``fastvideo/tests/mlx/test_mlx_minimax_h3_parity.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.fastwan import (
    MLXQuantizationSpec,
    QuantizedMatrix,
    ensure_quantization_supported,
    linear,
    quantize_matrix,
    rms_norm,
    silu,
    timestep_embedding,
    weight_dtype,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrors fastvideo/pipelines/basic/minimax_h3/packing.py)
# ---------------------------------------------------------------------------

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_MODALITY_NUM = 3

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999

MINIMAX_H3_VIDEO_SHIFT = 12.0
MINIMAX_H3_AUDIO_SHIFT = 3.0

MINIMAX_H3_ROPE_FRAME_RESCALE = 5.0 / 3.0
MINIMAX_H3_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0

# Modules the release checkpoint stores in fp32 (and this runtime keeps fp32).
FP32_MODULE_PREFIXES = (
    "proj_in",
    "audio_proj_in",
    "time_embedder",
    "proj_out",
    "audio_proj_out",
)

# Linear weights that go on the int8 deploy grid. Everything else — norms,
# AdaLN/modulation, embeddings, input/output projections — stays unquantized.
_QUANTIZABLE_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
)


def _is_quantizable(key: str) -> bool:
    return key.endswith(_QUANTIZABLE_SUFFIXES)


# ---------------------------------------------------------------------------
# Geometry (numpy, float64 — the reference RoPE coordinate arithmetic)
# ---------------------------------------------------------------------------


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}.")
    if ratio >= 1:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio
    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area)**0.5
        width, height = width * scale, height * scale
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
            ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # NumPy endpoint=False float64 arithmetic is the reference contract.
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array(
        [
            MINIMAX_H3_ROPE_FRAME_RESCALE *
            MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), spans[:-1].cumsum()])


def _temporal_position_span(num_latent_frames: int) -> float:
    # Preserve NumPy's pairwise summation order (reference parity note).
    spans = np.ones(num_latent_frames, dtype=np.float64) * MINIMAX_H3_ROPE_FRAME_RESCALE
    for index, frames in enumerate(MINIMAX_H3_ROPE_FRAMES_PER_LATENT):
        spans[index::len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)] *= frames
    return float(spans.sum())


def patchify_video_latents(latents: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    """(B, C, T, H, W) -> (B*T'*H'*W', C*pt*ph*pw), channel-major patch features."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {latents.shape} are not divisible by the patch {patch_size}.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return np.ascontiguousarray(latents.reshape(-1, channels * patch_t * patch_h * patch_w))


def unpatchify_video_tokens(
    rows: np.ndarray,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return np.ascontiguousarray(rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width))


def unpack_audio_tokens(rows: np.ndarray, num_audio_latents: int) -> np.ndarray:
    """(2*num_audio_latents, feat) channel-major rows -> (2, feat, num_audio_latents)."""
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return np.ascontiguousarray(rows.transpose(0, 2, 1))


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    """One packed joint sequence and the geometry needed to interpret it.

    Arrays are NumPy (position_ids in float64, indices int64) and converted
    to MLX at the model boundary.
    """

    sequence_length: int
    position_ids: np.ndarray
    token_tags: np.ndarray
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_video_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int


def build_packed_layout(
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: tuple[str, ...] = (),
    text_token_tags: np.ndarray | None = None,
) -> MiniMaxH3PackedLayout:
    """Build the ``[text | condition | audio | video]`` layout (T2VA / FL2VA)."""
    _, patch_h, patch_w = patch_size
    if text_token_tags is None:
        text_token_tags = np.full(num_text_tokens, MINIMAX_H3_TEXT_TAG, dtype=np.int64)
    if text_token_tags.shape != (num_text_tokens, ):
        raise ValueError(f"text_token_tags must have shape ({num_text_tokens},), got {text_token_tags.shape}.")
    if not np.isin(text_token_tags, (MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)).all():
        raise ValueError("text_token_tags may contain only text and vision tags.")

    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text_tokens, 0] = np.arange(num_text_tokens, dtype=np.float64)

    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
    mesh_h, mesh_w = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = np.stack([mesh_h.reshape(-1), mesh_w.reshape(-1)], axis=-1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (float(num_text_tokens) + _temporal_position_span(num_latent_frames) -
                           MINIMAX_H3_ROPE_FRAME_RESCALE)
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[audio_start:video_start, 0] = np.tile(audio_time, MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = np.concatenate([
        np.full(num_audio_latents, float(width_grid[0]), dtype=np.float64),
        np.full(num_audio_rows - num_audio_latents, float(width_grid[-1]), dtype=np.float64),
    ])

    frame_time = temporal_position_grid(num_latent_frames, float(num_text_tokens))
    position_ids[video_start:, 0] = np.repeat(frame_time, rows_per_frame)
    position_ids[video_start:, 1:] = np.tile(frame_grid, (num_latent_frames, 1))

    video_indices = np.concatenate([
        np.arange(condition_start, audio_start),
        np.arange(video_start, sequence_length),
    ])
    audio_indices = np.arange(audio_start, video_start)
    text_indices = np.arange(num_text_tokens)
    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedLayout(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
        num_video_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float = 1.0,
    condition_audio_timestep: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row timesteps -> (unique timesteps sorted ascending, per-row indices).

    Matches ``torch.unique(..., sorted=True, return_inverse=True)``.
    """
    row_timesteps = np.full(layout.sequence_length, video_timestep, dtype=np.float64)
    if layout.num_condition_video_rows:
        row_timesteps[layout.video_indices[:layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows:]] = audio_timestep
    if layout.num_condition_audio_rows:
        row_timesteps[layout.audio_indices[:layout.num_condition_audio_rows]] = condition_audio_timestep
    unique, inverse = np.unique(row_timesteps, return_inverse=True)
    return unique.astype(np.float32), inverse.astype(np.int64)


# ---------------------------------------------------------------------------
# Scheduler (rectified flow, clean-time convention, data-ward velocity)
# ---------------------------------------------------------------------------


def _unique_consecutive(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    keep = np.concatenate(([True], values[1:] != values[:-1]))
    return values[keep]


def minimax_h3_sigmas(shift: float, num_denoise_steps: int) -> np.ndarray:
    """Sigma grid of length ``num_denoise_steps + 1`` (descending, ending at 0).

    The reference ``set_timesteps(num_inference_steps=N)`` builds N sigmas and
    N-1 timesteps; this wrapper takes the *denoise step count* directly, which
    is what a sampler actually schedules.
    """
    if num_denoise_steps < 1:
        raise ValueError(f"num_denoise_steps must be >= 1, got {num_denoise_steps}.")
    base = np.linspace(1.0, 0.0, num_denoise_steps + 1, dtype=np.float64)
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    return _unique_consecutive(sigmas).astype(np.float32)


@dataclass
class MiniMaxH3SchedulerState:
    """One rectified-flow scheduler (use two: video shift 12, audio shift 3)."""

    shift: float
    sigmas: np.ndarray
    timesteps: np.ndarray  # 1 - sigmas[:-1], fp32

    @classmethod
    def create(cls, shift: float, num_denoise_steps: int) -> "MiniMaxH3SchedulerState":
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        sigmas = minimax_h3_sigmas(shift, num_denoise_steps)
        return cls(shift=shift, sigmas=sigmas, timesteps=(1.0 - sigmas[:-1]).astype(np.float32))

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])

    def step(self, model_output, step_index: int, sample):
        """Data-ward Euler: x0 = x_t + sigma*v; blend toward x0 by sigma ratio.

        fp32 blend for fp16/bf16 samples, matching the reference.
        """
        import mlx.core as mx

        sigma_from_timestep = float(1.0 - self.timesteps[step_index])
        denoised = sample + sigma_from_timestep * model_output
        sigma = float(self.sigmas[step_index])
        sigma_next = float(self.sigmas[step_index + 1])
        ratio = sigma_next / sigma
        prev = ratio * sample.astype(mx.float32) + (1.0 - ratio) * denoised.astype(mx.float32)
        return prev.astype(sample.dtype)

    def scale_noise(self, sample, timestep: float, noise):
        """Conditioning noise-aug: t*sample + (1-t)*noise (t=0.999 for keyframes)."""
        return timestep * sample + (1.0 - timestep) * noise


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


def _as_mx_indices(indices):
    """NumPy/python index arrays are not accepted by MLX indexing; convert."""
    import mlx.core as mx

    if isinstance(indices, mx.array):
        return indices
    return mx.array(np.asarray(indices, dtype=np.int64))


def rope_cos_sin(position_ids, rope_freq_dim: int, rope_theta: float):
    """(S, 3) positions -> (cos, sin) each (S, 6*rope_freq_dim), fp32.

    Shared 16-freq inv_freq across the three axes; the (t, h, w) blocks are
    concatenated and doubled, so the rotary width is 6 * rope_freq_dim (96 of
    the 128 head dims at full size).
    """
    import mlx.core as mx

    positions = position_ids.astype(mx.float32)
    inv_freq = 1.0 / (rope_theta
                      **(mx.arange(0, 2 * rope_freq_dim, 2, dtype=mx.float32) / (2 * rope_freq_dim)))
    freqs = positions[:, :, None] * inv_freq[None, None, :]  # (S, 3, F)
    freqs_t, freqs_h, freqs_w = freqs[:, 0], freqs[:, 1], freqs[:, 2]
    freqs = mx.concatenate([freqs_t, freqs_h, freqs_w], axis=-1)
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(freqs), mx.sin(freqs)


def apply_h3_rotary(x, cos, sin):
    """Rotate the RoPE prefix of each head, preserving the rest.

    x: (S, H, D); cos/sin: (S, R) with R <= D. Half-split (GPT-NeoX style)
    rotation inside the rotary prefix, computed in fp32.
    """
    import mlx.core as mx

    rotary_dim = cos.shape[-1]
    x_rotary = x[..., :rotary_dim].astype(mx.float32)
    x_pass = x[..., rotary_dim:]
    cos_b = cos[:, None, :].astype(mx.float32)
    sin_b = sin[:, None, :].astype(mx.float32)
    first_half, second_half = mx.split(x_rotary, 2, axis=-1)
    rotated = mx.concatenate([-second_half, first_half], axis=-1)
    out = x_rotary * cos_b + rotated * sin_b
    return mx.concatenate([out.astype(x.dtype), x_pass], axis=-1)


def _attention(weights: dict[str, Any], x, cos, sin, *, num_heads: int, head_dim: int, eps: float, use_rope: bool):
    """H3 attention: bias-free qkv, per-head qk RMSNorm, partial RoPE."""
    import mlx.core as mx

    seq_len = x.shape[0]
    q = linear(x, weights["attn.to_q.weight"]).reshape(seq_len, num_heads, head_dim)
    k = linear(x, weights["attn.to_k.weight"]).reshape(seq_len, num_heads, head_dim)
    v = linear(x, weights["attn.to_v.weight"]).reshape(seq_len, num_heads, head_dim)
    q = rms_norm(q, weights["attn.norm_q.weight"], eps=eps)
    k = rms_norm(k, weights["attn.norm_k.weight"], eps=eps)
    if use_rope:
        q = apply_h3_rotary(q, cos, sin)
        k = apply_h3_rotary(k, cos, sin)
    attended = mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None],
        k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None],
        scale=head_dim**-0.5,
    )[0].transpose(1, 0, 2)
    return linear(attended.reshape(seq_len, num_heads * head_dim), weights["attn.to_out.0.weight"])


def _feed_forward(weights: dict[str, Any], x):
    """Bias-free SwiGLU with value-first packed halves: h * silu(gate)."""
    import mlx.core as mx

    hidden = linear(x, weights["ff.net.0.proj.weight"])
    value, gate = mx.split(hidden, 2, axis=-1)
    return linear(value * silu(gate), weights["ff.net.2.weight"])


def _adaln_tables(weights: dict[str, Any], temb):
    """Six (n_t * 3, hidden) modulation tables from (n_t, time_embed_dim)."""
    import mlx.core as mx

    projected = linear(
        silu(temb).astype(weight_dtype(weights["adaln_proj.linear.weight"])),
        weights["adaln_proj.linear.weight"],
        weights["adaln_proj.linear.bias"],
    )
    hidden = weights["adaln_proj.linear.bias"].shape[0] // (6 * MINIMAX_H3_MODALITY_NUM)
    tables = projected.reshape(-1, 6 * hidden)
    return mx.split(tables, 6, axis=-1)


def _transformer_block(
    weights: dict[str, Any],
    hidden_states,
    tables,
    adaln_indices,
    cos,
    sin,
    *,
    num_heads: int,
    head_dim: int,
    norm_eps: float,
    qk_norm_eps: float,
):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = tables

    residual = hidden_states
    normed = rms_norm(hidden_states, weights["norm1.weight"], eps=norm_eps)
    normed = normed * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    attn_output = _attention(
        weights,
        normed.astype(weight_dtype(weights["attn.to_q.weight"])),
        cos,
        sin,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=qk_norm_eps,
        use_rope=True,
    )
    hidden_states = residual + gate_msa[adaln_indices] * attn_output

    residual = hidden_states
    normed = rms_norm(hidden_states, weights["norm2.weight"], eps=norm_eps)
    normed = normed * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
    ff_output = _feed_forward(weights, normed.astype(weight_dtype(weights["ff.net.0.proj.weight"])))
    return residual + gate_mlp[adaln_indices] * ff_output


def _refiner_block(
    weights: dict[str, Any],
    hidden_states,
    *,
    num_heads: int,
    head_dim: int,
    norm_eps: float,
    qk_norm_eps: float,
):
    residual = hidden_states
    normed = rms_norm(hidden_states, weights["norm1.weight"], eps=norm_eps)
    attn_output = _attention(
        weights,
        normed.astype(weight_dtype(weights["attn.to_q.weight"])),
        None,
        None,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=qk_norm_eps,
        use_rope=False,
    )
    hidden_states = residual + attn_output
    normed = rms_norm(hidden_states, weights["norm2.weight"], eps=norm_eps)
    return hidden_states + _feed_forward(weights, normed.astype(weight_dtype(weights["ff.net.0.proj.weight"])))


@dataclass
class MiniMaxH3StepCache:
    """Precomputed AdaLN tables + norm_out modulation for a fixed schedule.

    Holds one row set per distinct timestep in ``timesteps`` (the union of
    every denoise step's video/audio/condition timesteps). Blocks then index
    tables directly and the per-block ``adaln_proj`` weights can be freed —
    ~40% of H3's parameters never need to be resident during denoise.
    """

    timesteps: np.ndarray  # (n,) fp32, sorted
    block_tables: list[tuple[Any, ...]]  # per block: 6 tables of (n*3, hidden)
    norm_out_shift: Any  # (n, hidden)
    norm_out_scale: Any  # (n, hidden)

    def positions(self, step_timesteps: np.ndarray) -> np.ndarray:
        """Map a step's unique timesteps to rows in the cached union."""
        positions = np.searchsorted(self.timesteps, step_timesteps)
        if positions.size and not np.allclose(self.timesteps[positions], step_timesteps, atol=1e-6):
            raise ValueError(f"Step timesteps {step_timesteps} are not in the cached schedule union.")
        return positions.astype(np.int64)


class MLXMiniMaxH3DiT:
    """MiniMax H3 joint audio-video DiT in MLX (batch-1 packed forward).

    Weight dicts keep the released checkpoint's key names. ``blocks`` holds
    the main transformer blocks, ``refiner`` the text refiner blocks; each is
    a dict of arrays / :class:`QuantizedMatrix`.
    """

    def __init__(
        self,
        weights: dict[str, Any],
        blocks: list[dict[str, Any]],
        refiner: list[dict[str, Any]],
        config: dict[str, Any],
        *,
        compile: bool = False,
    ) -> None:
        import os

        self.weights = weights
        self.blocks = blocks
        self.refiner = refiner
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = int(config["attention_head_dim"])
        self.ffn_dim = int(config["ffn_dim"])
        self.in_channels = int(config["in_channels"])
        self.audio_in_channels = int(config["audio_in_channels"])
        self.patch_size = tuple(config["patch_size"])
        self.text_dim = int(config["text_dim"])
        self.freq_dim = int(config["freq_dim"])
        self.time_embed_dim = int(config["time_embed_dim"])
        self.rope_freq_dim = int(config["rope_freq_dim"])
        self.rope_theta = float(config["rope_theta"])
        self.norm_eps = float(config["norm_eps"])
        self.qk_norm_eps = float(config["qk_norm_eps"])
        self.final_norm_eps = float(config["final_norm_eps"])
        self.patch_dim = self.in_channels * math.prod(self.patch_size)
        self._enable_compile = compile or os.environ.get("FASTVIDEO_MLX_COMPILE", "0") == "1"
        self._compiled_forward = None
        self._adaln_cache: MiniMaxH3StepCache | None = None

    # -- conditioning ------------------------------------------------------

    def compute_temb(self, timesteps):
        """(n,) timesteps -> (n, time_embed_dim) through time_proj + MLP."""
        t_freq = timestep_embedding(timesteps, self.freq_dim).astype(
            weight_dtype(self.weights["time_embedder.linear_1.weight"]))
        temb = linear(
            t_freq,
            self.weights["time_embedder.linear_1.weight"],
            self.weights["time_embedder.linear_1.bias"],
        )
        return linear(
            silu(temb),
            self.weights["time_embedder.linear_2.weight"],
            self.weights["time_embedder.linear_2.bias"],
        )

    def refine_text(self, text_rows):
        hidden = linear(
            text_rows.astype(weight_dtype(self.weights["context_embedder.weight"])),
            self.weights["context_embedder.weight"],
            self.weights["context_embedder.bias"],
        )
        for block in self.refiner:
            hidden = _refiner_block(
                block,
                hidden,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=self.norm_eps,
                qk_norm_eps=self.qk_norm_eps,
            )
        return rms_norm(hidden, self.weights["token_refiner.final_norm.weight"], eps=self.final_norm_eps)

    # -- AdaLN cache --------------------------------------------------------

    def precompute_adaln(self, timesteps: np.ndarray, *, drop_weights: bool = True) -> MiniMaxH3StepCache:
        """Evaluate every modulation table for a fixed schedule, once.

        ``timesteps`` is the union (sorted, unique) of all per-step
        video/audio/condition timesteps the sampler will use. With
        ``drop_weights=True`` the per-block ``adaln_proj.linear`` weights are
        released afterwards — the memory win this runtime exists for.
        """
        import mlx.core as mx

        timesteps = np.unique(np.asarray(timesteps, dtype=np.float32))
        temb = self.compute_temb(mx.array(timesteps))
        block_tables = [_adaln_tables(block, temb) for block in self.blocks]
        shift_scale = linear(
            silu(temb).astype(weight_dtype(self.weights["norm_out.linear.weight"])),
            self.weights["norm_out.linear.weight"],
            self.weights["norm_out.linear.bias"],
        )
        norm_out_shift, norm_out_scale = mx.split(shift_scale, 2, axis=-1)
        mx.eval(block_tables, norm_out_scale, norm_out_shift)
        self._adaln_cache = MiniMaxH3StepCache(
            timesteps=timesteps,
            block_tables=block_tables,
            norm_out_shift=norm_out_shift,
            norm_out_scale=norm_out_scale,
        )
        if drop_weights:
            for block in self.blocks:
                block["adaln_proj.linear.weight"] = None
                block["adaln_proj.linear.bias"] = None
        return self._adaln_cache

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        video_rows,
        audio_rows,
        text_rows,
        *,
        position_ids,
        token_tags,
        timestep_indices,
        timesteps,
        video_indices,
        audio_indices,
        text_indices,
    ):
        """Faithful port of the torch forward (batch-1, rows already patchified).

        Returns (video_output, audio_output) rows. The torch reference
        projects *all* packed rows through both heads and then selects; this
        selects first and projects only the relevant rows — row-wise
        identical math at a fraction of the cost.
        """
        import mlx.core as mx

        sequence_length = position_ids.shape[0]
        cos, sin = rope_cos_sin(position_ids, self.rope_freq_dim, self.rope_theta)

        video_embeds = linear(
            video_rows.astype(weight_dtype(self.weights["proj_in.weight"])),
            self.weights["proj_in.weight"],
            self.weights["proj_in.bias"],
        )
        audio_embeds = linear(
            audio_rows.astype(weight_dtype(self.weights["audio_proj_in.weight"])),
            self.weights["audio_proj_in.weight"],
            self.weights["audio_proj_in.bias"],
        )
        text_embeds = self.refine_text(text_rows)

        packed = mx.zeros((sequence_length, self.hidden_size), dtype=text_embeds.dtype)
        packed[_as_mx_indices(text_indices)] = text_embeds
        packed[_as_mx_indices(video_indices)] = video_embeds.astype(text_embeds.dtype)
        packed[_as_mx_indices(audio_indices)] = audio_embeds.astype(text_embeds.dtype)

        temb = self.compute_temb(timesteps)
        adaln_indices = (timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags).astype(mx.int32)

        for block in self.blocks:
            tables = _adaln_tables(block, temb)
            packed = _transformer_block(
                block,
                packed,
                tables,
                adaln_indices,
                cos,
                sin,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=self.norm_eps,
                qk_norm_eps=self.qk_norm_eps,
            )

        shift_scale = linear(
            silu(temb).astype(weight_dtype(self.weights["norm_out.linear.weight"])),
            self.weights["norm_out.linear.weight"],
            self.weights["norm_out.linear.bias"],
        )
        shift_rows, scale_rows = mx.split(shift_scale, 2, axis=-1)
        normed = rms_norm(packed, self.weights["norm_out.norm.weight"], eps=self.final_norm_eps)
        normed = normed * (1.0 + scale_rows[timestep_indices]) + shift_rows[timestep_indices]
        video_output = linear(
            normed[_as_mx_indices(video_indices)].astype(weight_dtype(self.weights["proj_out.weight"])),
            self.weights["proj_out.weight"],
            self.weights["proj_out.bias"],
        )
        audio_output = linear(
            normed[_as_mx_indices(audio_indices)].astype(weight_dtype(self.weights["audio_proj_out.weight"])),
            self.weights["audio_proj_out.weight"],
            self.weights["audio_proj_out.bias"],
        )
        return video_output, audio_output

    def forward_with_cache(
        self,
        video_rows,
        audio_rows,
        text_rows,
        *,
        layout: MiniMaxH3PackedLayout,
        step_timesteps: np.ndarray,
        row_timestep_inverse: np.ndarray,
    ):
        """Denoise-step forward served entirely from the AdaLN cache.

        ``step_timesteps`` are this step's unique timesteps (sorted);
        ``row_timestep_inverse`` maps each packed row to one of them (the
        ``build_row_timesteps`` inverse). The cache must cover every value in
        ``step_timesteps``.
        """
        import mlx.core as mx

        if self._adaln_cache is None:
            raise RuntimeError("precompute_adaln() must run before forward_with_cache().")
        cache = self._adaln_cache
        positions = mx.array(cache.positions(step_timesteps))

        sequence_length = layout.sequence_length
        position_ids = mx.array(layout.position_ids)
        cos, sin = rope_cos_sin(position_ids, self.rope_freq_dim, self.rope_theta)

        video_embeds = linear(
            video_rows.astype(weight_dtype(self.weights["proj_in.weight"])),
            self.weights["proj_in.weight"],
            self.weights["proj_in.bias"],
        )
        audio_embeds = linear(
            audio_rows.astype(weight_dtype(self.weights["audio_proj_in.weight"])),
            self.weights["audio_proj_in.weight"],
            self.weights["audio_proj_in.bias"],
        )
        text_embeds = self.refine_text(text_rows)

        packed = mx.zeros((sequence_length, self.hidden_size), dtype=text_embeds.dtype)
        packed[mx.array(layout.text_indices)] = text_embeds
        packed[mx.array(layout.video_indices)] = video_embeds.astype(text_embeds.dtype)
        packed[mx.array(layout.audio_indices)] = audio_embeds.astype(text_embeds.dtype)

        token_tags = mx.array(layout.token_tags)
        row_inverse = mx.array(row_timestep_inverse)
        adaln_indices = (positions[row_inverse] * MINIMAX_H3_MODALITY_NUM + token_tags).astype(mx.int32)

        for block, tables in zip(self.blocks, cache.block_tables):
            packed = _transformer_block(
                block,
                packed,
                tables,
                adaln_indices,
                cos,
                sin,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=self.norm_eps,
                qk_norm_eps=self.qk_norm_eps,
            )

        row_positions = positions[row_inverse]
        normed = rms_norm(packed, self.weights["norm_out.norm.weight"], eps=self.final_norm_eps)
        normed = normed * (1.0 + cache.norm_out_scale[row_positions]) + cache.norm_out_shift[row_positions]
        video_output = linear(
            normed[mx.array(layout.video_indices)].astype(weight_dtype(self.weights["proj_out.weight"])),
            self.weights["proj_out.weight"],
            self.weights["proj_out.bias"],
        )
        audio_output = linear(
            normed[mx.array(layout.audio_indices)].astype(weight_dtype(self.weights["audio_proj_out.weight"])),
            self.weights["audio_proj_out.weight"],
            self.weights["audio_proj_out.bias"],
        )
        return video_output, audio_output

    __call__ = forward


# ---------------------------------------------------------------------------
# Checkpoint loading (diffusers-layout safetensors) and pre-quantized save/load
# ---------------------------------------------------------------------------


def _load_array(handle, name: str, dtype):
    import mlx.core as mx
    import torch

    tensor = handle.get_tensor(name)
    if dtype == mx.float16:
        tensor = tensor.to(torch.float16)
    elif dtype == mx.float32:
        tensor = tensor.to(torch.float32)
    elif dtype == mx.bfloat16:
        tensor = tensor.to(torch.float32)
    array = mx.array(tensor.numpy())
    del tensor
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    mx.eval(array)
    return array


def _eval_value(value) -> None:
    import mlx.core as mx

    if isinstance(value, QuantizedMatrix):
        args = [value.weight, value.scales]
        if value.biases is not None:
            args.append(value.biases)
        mx.eval(*args)
    else:
        mx.eval(value)


def _safetensors_shards(transformer_dir: str | Path) -> list[Path]:
    transformer_dir = Path(transformer_dir)
    if transformer_dir.is_file():
        return [transformer_dir]
    index = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted({transformer_dir / shard for shard in weight_map.values()})
    single = transformer_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(transformer_dir.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors found under {transformer_dir}")


def mlx_h3_dit_from_diffusers_safetensors(
    transformer_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    dtype: str = "fp16",
    num_blocks: int | None = None,
    quantization: str | MLXQuantizationSpec | None = None,
) -> MLXMiniMaxH3DiT:
    """Load the released H3 transformer (diffusers layout) into MLX.

    ``transformer_path`` is the ``transformer/`` directory of the HF repo (or
    a single safetensors file, e.g. a student checkpoint). ``config`` defaults
    to ``config.json`` next to the weights. fp32-release modules stay fp32;
    attention/FFN matrices are quantized when ``quantization`` is set.
    """
    import mlx.core as mx
    from safetensors import safe_open

    transformer_path = Path(transformer_path)
    if config is None:
        config_path = transformer_path / "config.json" if transformer_path.is_dir() else transformer_path.with_name(
            "config.json")
        config = json.loads(Path(config_path).read_text())
    total_blocks = int(config["num_layers"])
    if num_blocks is None:
        num_blocks = total_blocks
    num_refiner = int(config["num_refiner_layers"])
    cast_dtype = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[dtype]
    spec = MLXQuantizationSpec.from_name(quantization) if (quantization is None or isinstance(
        quantization, str)) else quantization
    ensure_quantization_supported(spec)

    weights: dict[str, Any] = {}
    blocks: list[dict[str, Any] | None] = [None] * num_blocks
    refiner: list[dict[str, Any] | None] = [None] * num_refiner

    def assign(key: str, value) -> None:
        if key.startswith("transformer_blocks."):
            _, index_str, sub = key.split(".", 2)
            index = int(index_str)
            if index < num_blocks:
                if blocks[index] is None:
                    blocks[index] = {}
                blocks[index][sub] = value
        elif key.startswith("token_refiner.refiner_blocks."):
            _, _, index_str, sub = key.split(".", 3)
            index = int(index_str)
            if refiner[index] is None:
                refiner[index] = {}
            refiner[index][sub] = value
        else:
            weights[key] = value

    for shard in _safetensors_shards(transformer_path):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("transformer_blocks."):
                    index = int(key.split(".")[1])
                    if index >= num_blocks:
                        continue
                if key.startswith("rope."):
                    continue  # non-persistent analytic buffer, rebuilt on the fly
                keep_fp32 = key.split(".", 1)[0] in FP32_MODULE_PREFIXES
                target_dtype = mx.float32 if keep_fp32 else cast_dtype
                array = _load_array(handle, key, target_dtype)
                if spec is not None and _is_quantizable(key) and not keep_fp32:
                    value = quantize_matrix(array, spec)
                    del array
                else:
                    value = array
                _eval_value(value)
                assign(key, value)

    if any(block is None for block in blocks):
        missing = [i for i, block in enumerate(blocks) if block is None]
        raise KeyError(f"Missing transformer block weights for indices {missing}.")
    if any(block is None for block in refiner):
        raise KeyError("Missing token refiner weights.")
    return MLXMiniMaxH3DiT(
        weights,
        [block for block in blocks if block is not None],
        [block for block in refiner if block is not None],
        config,
    )


H3_FORMAT_VERSION = 1
H3_WEIGHTS_FILENAME = "mlx_h3_dit.safetensors"
H3_MANIFEST_FILENAME = "mlx_h3_dit.json"

_DTYPE_TO_NAME = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


def _dtype_name(dtype) -> str:
    import mlx.core as mx

    for raw, name in _DTYPE_TO_NAME.items():
        if dtype == getattr(mx, raw):
            return name
    raise ValueError(f"Unsupported MLX dtype for checkpointing: {dtype}")


def _name_to_dtype(name: str):
    import mlx.core as mx

    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]


def _flatten_h3_weights(dit: MLXMiniMaxH3DiT) -> dict[str, Any]:
    flat: dict[str, Any] = dict(dit.weights)
    for index, block in enumerate(dit.blocks):
        for name, value in block.items():
            if value is not None:
                flat[f"blocks.{index}.{name}"] = value
    for index, block in enumerate(dit.refiner):
        for name, value in block.items():
            flat[f"refiner.{index}.{name}"] = value
    return flat


def save_mlx_h3_checkpoint(dit: MLXMiniMaxH3DiT, checkpoint_dir: str | Path) -> Path:
    """Persist a (possibly quantized, possibly AdaLN-dropped) H3 DiT."""
    import mlx.core as mx

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {}
    quantized: dict[str, dict[str, Any]] = {}
    spec: MLXQuantizationSpec | None = None
    for key, value in _flatten_h3_weights(dit).items():
        if isinstance(value, QuantizedMatrix):
            if spec is not None and value.spec != spec:
                raise ValueError(f"Mixed quantization specs in one checkpoint ({spec} vs {value.spec} at '{key}').")
            spec = value.spec
            arrays[key] = value.weight
            arrays[f"{key}.scales"] = value.scales
            if value.biases is not None:
                arrays[f"{key}.biases"] = value.biases
            quantized[key] = {
                "dequantized_dtype": _dtype_name(value.dequantized_dtype),
                "has_biases": value.biases is not None,
            }
        else:
            arrays[key] = value

    manifest = {
        "format_version": H3_FORMAT_VERSION,
        "config": dit.config,
        "num_blocks": len(dit.blocks),
        "num_refiner_blocks": len(dit.refiner),
        "quantization": None if spec is None else {
            "mode": spec.mode,
            "bits": spec.bits,
            "group_size": spec.group_size,
        },
        "quantized_keys": quantized,
    }
    weights_path = checkpoint_dir / H3_WEIGHTS_FILENAME
    mx.save_safetensors(str(weights_path), arrays)
    (checkpoint_dir / H3_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    logger.info("Saved MLX H3 DiT checkpoint (%d arrays, quantization=%s) to %s",
                len(arrays), spec.label if spec else "none", checkpoint_dir)
    return checkpoint_dir


def load_mlx_h3_checkpoint(checkpoint_dir: str | Path) -> MLXMiniMaxH3DiT:
    """Rebuild an H3 DiT saved by :func:`save_mlx_h3_checkpoint`.

    Refuses a quantization grid the installed MLX cannot execute, loudly —
    never silently dequantizes onto a different grid.
    """
    import mlx.core as mx

    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / H3_MANIFEST_FILENAME
    weights_path = checkpoint_dir / H3_WEIGHTS_FILENAME
    if not manifest_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Not an MLX H3 checkpoint directory: {checkpoint_dir} "
            f"(expected {H3_MANIFEST_FILENAME} and {H3_WEIGHTS_FILENAME}).")

    manifest = json.loads(manifest_path.read_text())
    version = manifest.get("format_version")
    if version != H3_FORMAT_VERSION:
        raise ValueError(
            f"MLX H3 checkpoint {checkpoint_dir} has format_version={version}; "
            f"this build reads version {H3_FORMAT_VERSION}. Re-export the checkpoint.")

    spec = None
    if manifest["quantization"] is not None:
        spec = MLXQuantizationSpec(**manifest["quantization"])
        ensure_quantization_supported(spec)

    arrays = mx.load(str(weights_path))
    quantized_keys: dict[str, dict[str, Any]] = manifest["quantized_keys"]

    def rebuild(key: str):
        if key not in quantized_keys:
            return arrays[key]
        info = quantized_keys[key]
        assert spec is not None, f"Quantized key '{key}' in a checkpoint without a quantization spec"
        return QuantizedMatrix(
            weight=arrays[key],
            scales=arrays[f"{key}.scales"],
            biases=arrays[f"{key}.biases"] if info["has_biases"] else None,
            spec=spec,
            dequantized_dtype=_name_to_dtype(info["dequantized_dtype"]),
        )

    weights: dict[str, Any] = {}
    blocks: list[dict[str, Any] | None] = [None] * int(manifest["num_blocks"])
    refiner: list[dict[str, Any] | None] = [None] * int(manifest["num_refiner_blocks"])
    for key in arrays:
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        value = rebuild(key)
        if key.startswith("blocks."):
            _, index_str, sub = key.split(".", 2)
            index = int(index_str)
            if blocks[index] is None:
                blocks[index] = {}
            blocks[index][sub] = value
        elif key.startswith("refiner."):
            _, index_str, sub = key.split(".", 2)
            index = int(index_str)
            if refiner[index] is None:
                refiner[index] = {}
            refiner[index][sub] = value
        else:
            weights[key] = value

    if any(block is None for block in blocks) or any(block is None for block in refiner):
        raise ValueError(f"MLX H3 checkpoint {checkpoint_dir} is missing block weights.")
    return MLXMiniMaxH3DiT(
        weights,
        [block for block in blocks if block is not None],
        [block for block in refiner if block is not None],
        manifest["config"],
    )
