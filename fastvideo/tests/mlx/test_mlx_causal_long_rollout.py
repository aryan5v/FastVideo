# SPDX-License-Identifier: Apache-2.0
"""Long causal rollout: bounded KV memory under rolling eviction + sink tokens.

Proves the north-star property of the streaming runtime: with a positive
``local_attn_size``, generating many more blocks than the attention window keeps
the K/V tensor size fixed at ``local_attn_size * frame_seqlen`` while
``global_end_index`` advances and ``local_end_index`` saturates at the window.
Backend-agnostic (Metal or mlx[cpu]); uses the tiny random-weight builders from
``test_mlx_causal_dit_parity``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for the long-rollout test")

from fastvideo.layers.rotary_embedding import get_rotary_pos_embed  # noqa: E402
from fastvideo.mlx_runtime.causal import max_attention_size  # noqa: E402
from fastvideo.mlx_runtime.causal_sampler import build_dmd_schedule, stream_causal_latents  # noqa: E402
from fastvideo.tests.mlx.test_mlx_causal_dit_parity import (  # noqa: E402
    ARCH,
    HEAD_DIM,
    HEIGHT,
    NUM_HEADS,
    WIDTH,
    _build_torch_model,
    _mlx_from_torch,
)

# Many more blocks than the window so eviction + sinks are exercised end-to-end.
NUM_BLOCKS = 28
LOCAL_ATTN_SIZE = 2  # frames kept in the rolling window
SINK_SIZE = 1  # frames protected as attention sinks
# Fewer DMD steps than production — control-flow / memory bound is the goal.
DMD_STEPS = [1000, 500]


def _rotary_for_frames(num_frames: int):
    """Frame-major cos/sin tables long enough for the full rollout."""
    d = (NUM_HEADS * HEAD_DIM) // NUM_HEADS
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    cos, sin = get_rotary_pos_embed(
        (num_frames, HEIGHT // 2, WIDTH // 2),
        NUM_HEADS * HEAD_DIM,
        NUM_HEADS,
        rope_dim_list,
        dtype=torch.float64,
        rope_theta=10000,
    )
    return mx.array(cos.float().numpy()), mx.array(sin.float().numpy())


@pytest.mark.usefixtures("distributed_setup")
def test_long_rollout_kv_cache_bounded_independent_of_length() -> None:
    """Stream many blocks; KV length stays at the window, local_end saturates."""
    model = _mlx_from_torch(_build_torch_model())
    # Windowed attention — the product path for videos longer than ~5s.
    model.local_attn_size = LOCAL_ATTN_SIZE
    model.sink_size = SINK_SIZE

    frame_seqlen = (HEIGHT // 2) * (WIDTH // 2)
    window = max_attention_size(LOCAL_ATTN_SIZE, frame_seqlen)
    assert window == LOCAL_ATTN_SIZE * frame_seqlen

    cos_full, sin_full = _rotary_for_frames(NUM_BLOCKS)
    schedule, timesteps = build_dmd_schedule(DMD_STEPS, flow_shift=8.0, warp_denoising_step=True)

    rng = np.random.default_rng(7)
    noise = mx.array(
        rng.standard_normal((1, ARCH["in_channels"], NUM_BLOCKS, HEIGHT, WIDTH)).astype(np.float32))
    text = mx.array((rng.standard_normal((1, 24, ARCH["text_dim"])) * 0.1).astype(np.float32))

    kv_caches, crossattn_caches = model.allocate_caches(batch=1, frame_seqlen=frame_seqlen, dtype=mx.float32)
    # Pre-stream: allocated to exactly the attention window (bounded memory).
    assert kv_caches[0].k.shape[1] == window
    assert kv_caches[0].v.shape[1] == window
    assert kv_caches[0].sink_tokens == SINK_SIZE * frame_seqlen

    blocks = list(
        stream_causal_latents(
            model,
            text,
            noise,
            cos_full,
            sin_full,
            schedule,
            timesteps,
            frame_seqlen=frame_seqlen,
            seed=0,
            kv_caches=kv_caches,
            crossattn_caches=crossattn_caches,
        ))

    assert len(blocks) == NUM_BLOCKS
    for index, (block_index, latent) in enumerate(blocks):
        assert block_index == index
        arr = np.array(latent.astype(mx.float32))
        assert arr.shape == (1, ARCH["out_channels"], 1, HEIGHT, WIDTH)
        assert np.isfinite(arr).all(), f"block {index} produced non-finite latents"

        # Memory bound holds after every block — independent of how far we are.
        assert kv_caches[0].k.shape[1] == window
        assert kv_caches[0].v.shape[1] == window
        for layer_cache in kv_caches:
            assert layer_cache.k.shape[1] == window
            assert layer_cache.v.shape[1] == window

    # global_end advances with the full sequence; local_end saturates at window.
    expected_global = NUM_BLOCKS * frame_seqlen  # nfb == 1
    assert kv_caches[0].global_end_index == expected_global
    assert kv_caches[0].local_end_index == window
    # Cross-attn was populated once and reused across the long rollout.
    assert crossattn_caches[0]["is_init"] is True
    # Many more tokens generated than the window → eviction path was used.
    assert expected_global > window

