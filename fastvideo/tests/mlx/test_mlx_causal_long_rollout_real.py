# SPDX-License-Identifier: Apache-2.0
"""Metal real-weight long stream: peak memory plateaus under rolling KV cache.

Loads SFWan-1.3B when present, streams 21+ latent frames with a bounded
``local_attn_size``, and records time-to-first-block, steady per-block latency,
and peak MLX memory after each block. Asserts the K/V tensor length equals the
attention window for the whole run (memory bound independent of length).

Skipped without Metal or without the local SFWan root so Linux ``mlx[cpu]`` CI
stays green. Numbers from a real Mac run are pasted into
``docs/design/apple_silicon_benchmark_baseline.md`` (Long-video streaming).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the real long-rollout test")

_ROOT = Path(os.environ.get("FASTVIDEO_SFWAN_ROOT", str(Path.home() / "models" / "sfwan_t2v_1.3b")))
_CHECKPOINT = _ROOT / "transformer" / "diffusion_pytorch_model.safetensors"
_CONFIG = _ROOT / "transformer" / "config.json"

_HAS_METAL = bool(getattr(mx, "metal", None) and mx.metal.is_available())
_HAS_WEIGHTS = _CHECKPOINT.exists() and _CONFIG.exists()

pytestmark = [
    pytest.mark.skipif(not _HAS_METAL, reason="Metal required for the real-weight long stream"),
    pytest.mark.skipif(not _HAS_WEIGHTS, reason=f"SFWan checkpoint not found under {_ROOT}"),
]

# Modest latent spatial size keeps the run tractable while still using real weights.
# Product shape (480×832) is exercised by the streaming demo; this gate proves the
# memory plateau property end-to-end on the released checkpoint.
LATENT_H = LATENT_W = 32
NUM_FRAMES = 24  # > local_attn_size so eviction is exercised
LOCAL_ATTN_SIZE = 6  # frames
SINK_SIZE = 1
DMD_STEPS = [1000, 750, 500, 250]


def test_real_long_stream_peak_memory_plateaus(tmp_path: Path) -> None:
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.mlx_runtime.causal import max_attention_size
    from fastvideo.mlx_runtime.causal_dit import mlx_causal_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.causal_sampler import build_dmd_schedule, stream_causal_latents
    import torch

    model = mlx_causal_dit_from_diffusers_safetensors(
        _CHECKPOINT,
        _CONFIG,
        dtype="fp16",
        local_attn_size=LOCAL_ATTN_SIZE,
        sink_size=SINK_SIZE,
        num_frames_per_block=1,
    )
    config = model.config
    in_channels = int(config["in_channels"])
    text_dim = int(config["text_dim"])
    text_len = int(config.get("text_len", 512))
    frame_seqlen = (LATENT_H // 2) * (LATENT_W // 2)
    window = max_attention_size(LOCAL_ATTN_SIZE, frame_seqlen)

    head_dim = int(config["attention_head_dim"])
    num_heads = int(config["num_attention_heads"])
    rope_dim_list = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    cos_t, sin_t = get_rotary_pos_embed(
        (NUM_FRAMES, LATENT_H // 2, LATENT_W // 2),
        num_heads * head_dim,
        num_heads,
        rope_dim_list,
        dtype=torch.float32,
        rope_theta=10000,
    )
    cos_full = mx.array(cos_t.float().numpy())
    sin_full = mx.array(sin_t.float().numpy())

    rng = np.random.default_rng(0)
    text = mx.array((rng.standard_normal((1, min(24, text_len), text_dim)) * 0.1).astype(np.float16))
    noise = mx.array(
        rng.standard_normal((1, in_channels, NUM_FRAMES, LATENT_H, LATENT_W)).astype(np.float16))

    schedule, timesteps = build_dmd_schedule(DMD_STEPS, flow_shift=8.0, warp_denoising_step=True)
    kv_caches, crossattn_caches = model.allocate_caches(batch=1, frame_seqlen=frame_seqlen, dtype=mx.float16)
    assert kv_caches[0].k.shape[1] == window

    mx.clear_cache()
    mx.reset_peak_memory()
    stream_start = time.perf_counter()
    block_latencies: list[float] = []
    peak_gib_by_block: list[float] = []
    prev = stream_start
    time_to_first = None

    for block_index, latent in stream_causal_latents(
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
    ):
        now = time.perf_counter()
        block_latencies.append(now - prev)
        prev = now
        if time_to_first is None:
            time_to_first = now - stream_start
        peak_gib_by_block.append(mx.get_peak_memory() / (1024**3))
        arr = np.array(latent.astype(mx.float32))
        assert arr.shape[2] == 1
        assert np.isfinite(arr).all(), f"block {block_index} non-finite"
        assert kv_caches[0].k.shape[1] == window

    total_s = time.perf_counter() - stream_start
    assert kv_caches[0].global_end_index == NUM_FRAMES * frame_seqlen
    assert kv_caches[0].local_end_index == window
    assert len(block_latencies) == NUM_FRAMES

    steady = statistics.median(block_latencies[1:]) if len(block_latencies) > 1 else block_latencies[0]
    # Peak memory after the window fills should not keep climbing with length.
    # Allow small allocator noise; the plateau is the property under test.
    mid = len(peak_gib_by_block) // 2
    late_max = max(peak_gib_by_block[mid:])
    early_at_window = peak_gib_by_block[min(LOCAL_ATTN_SIZE + SINK_SIZE, len(peak_gib_by_block) - 1)]
    # Late peak within 25% of post-window peak (plateau, not O(T) growth).
    assert late_max <= early_at_window * 1.25 + 0.05, (
        f"peak memory grew with length: early={early_at_window:.3f} late_max={late_max:.3f} "
        f"series={peak_gib_by_block}")

    metrics = {
        "model_root": str(_ROOT),
        "num_frames": NUM_FRAMES,
        "latent_hw": f"{LATENT_H}x{LATENT_W}",
        "local_attn_size": LOCAL_ATTN_SIZE,
        "sink_size": SINK_SIZE,
        "frame_seqlen": frame_seqlen,
        "kv_window_tokens": window,
        "dmd_steps": len(timesteps),
        "time_to_first_block_s": round(time_to_first, 3),
        "block_latency_steady_s": round(steady, 3),
        "block_latencies_s": [round(x, 3) for x in block_latencies],
        "peak_gib_by_block": [round(x, 3) for x in peak_gib_by_block],
        "peak_gib": round(max(peak_gib_by_block), 3),
        "total_stream_s": round(total_s, 2),
        "global_end_index": kv_caches[0].global_end_index,
        "local_end_index": kv_caches[0].local_end_index,
    }
    out = tmp_path / "long_stream_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    # Also print so a local `pytest -s` run captures pasteable numbers for the baseline doc.
    print("\n=== long-video streaming metrics (real SFWan) ===")
    print(json.dumps(metrics, indent=2))

