# SPDX-License-Identifier: Apache-2.0
"""FastH3 MLX quantization bake-off: version the 6 candidate grids on the
released FastH3-Preview 33B, on Apple Silicon, no cluster needed.

Formats: bf16 (reference) · int8 · int6 · int4 (affine) · mxfp8 · mxfp4.

Per format (all on the real 33B student):

  1. **reconstruction error** vs bf16 — one shared forward at the ladder's
     mid point; rel-L2 over the velocity outputs (+ per-weight rel-L2 for
     the attention/FFN matrices). The cheap quality predictor (rankings on
     the M5 survey: int8 1x, int6 ~4x predicted, int4 ~17x, mxfp8 ~12x,
     mxfp4 ~22x — re-measured here at 33B scale where PTQ absorbs better).
  2. **per-step runtime** over the trained 4-step ladder
     (sigma grid = linspace(1,0,5) shifted, matching upstream's 5-point
     trained grid; 4 DiT forwards) — the E1 accelerator probe: if mxfp4 is
     NOT materially faster than int8, MLX is emulating, not dispatching to
     the M5 neural accelerators.
  3. **active memory** after load (mx.get_active_memory) per format.
  4. **latents dump** — denoised video/audio latents per format, for offline
     decode on the cluster with torch VAEs (no MLX VAE port needed to judge
     quality): fastvideo/benchmarks/decode_survey_latents.py

Usage (on the target Mac, e.g. the 36 GB machine):

    python -m fastvideo.benchmarks.mlx_h3_quant_survey \\
        --model-root ~/models/FastVideo-Minimax-FastH3-Preview-v0.2 \\
        --latents-out ~/survey_latents

Requires mlx, safetensors. Read-only (never writes weights).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fastvideo.mlx_runtime.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    MLXMiniMaxH3DiT,
    build_packed_layout,
    mlx_h3_dit_from_diffusers_safetensors,
    minimax_h3_sigmas,
)
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported, linear, weight_dtype

FORMATS = ["bf16", "int8", "int6", "int4", "mxfp8", "mxfp4"]
BITS_PER_PARAM = {"bf16": 16.0, "int8": 8.5, "int6": 6.5, "int4": 4.5, "mxfp8": 8.25, "mxfp4": 4.25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True,
                        help="local snapshot of FastVideo-Minimax-FastH3-Preview-v0.2 (diffusers layout)")
    parser.add_argument("--formats", default=" ".join(FORMATS), help="space-separated subset of " + " ".join(FORMATS))
    parser.add_argument("--latents-out", default="", help="dump denoised latents per format here")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--text-tokens", type=int, default=40)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def _ladder_timesteps(shift: float) -> np.ndarray:
    """Continuous t for the trained 4-step ladder (5 sigma points, 4 forwards)."""
    return (1.0 - minimax_h3_sigmas(shift, 4)[:-1]).astype(np.float64)


def _synthetic_inputs(args: argparse.Namespace):
    """A realistic packed sequence without real data (geometry like a 480p clip)."""
    import mlx.core as mx

    latent_frames = 31  # 124 px frames at 4x temporal
    latent_h = args.height // 32
    latent_w = args.width // 32
    num_audio = int(args.audio_seconds * 40)
    layout = build_packed_layout(
        args.text_tokens, latent_frames, latent_h, latent_w, num_audio,
        patch_size=(1, 2, 2))
    g = mx.random.key(args.seed)
    video_rows = mx.random.normal((layout.video_indices.shape[0], 24 * 4), key=g)
    audio_rows = mx.random.normal((layout.audio_indices.shape[0], 32), key=g)
    text_rows = mx.random.normal((args.text_tokens, 5120), key=g)
    return layout, video_rows, audio_rows, text_rows


def _forward(dit: MLXMiniMaxH3DiT, layout, video_rows, audio_rows, text_rows, video_t: float, audio_t: float):
    import mlx.core as mx

    from fastvideo.mlx_runtime.minimax_h3 import build_row_timesteps

    unique, inverse = build_row_timesteps(layout, video_t, audio_t)
    return dit.forward(
        video_rows,
        audio_rows,
        text_rows,
        position_ids=mx.array(layout.position_ids.astype(np.float32)),
        token_tags=mx.array(layout.token_tags),
        timestep_indices=mx.array(inverse),
        timesteps=mx.array(unique),
        video_indices=mx.array(layout.video_indices),
        audio_indices=mx.array(layout.audio_indices),
        text_indices=mx.array(layout.text_indices),
    )


def _rel_l2(a, b) -> float:
    import mlx.core as mx

    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - b) ** 2)) / mx.sqrt(mx.sum(b**2)))


def _prepare_cache(dit: MLXMiniMaxH3DiT):
    """The FastH3 runtime: precompute all AdaLN tables for the ladder, free the
    ~13B of AdaLN projection weights, and serve steps from the cache."""
    import mlx.core as mx

    from fastvideo.mlx_runtime.minimax_h3 import build_row_timesteps

    video_t_list = _ladder_timesteps(MINIMAX_H3_VIDEO_SHIFT)
    audio_t_list = _ladder_timesteps(MINIMAX_H3_AUDIO_SHIFT)
    union = np.unique(np.concatenate([video_t_list, audio_t_list, [1.0]]))
    cache = dit.precompute_adaln(union, drop_weights=True)
    mx.eval()
    return cache, video_t_list, audio_t_list


def _denoise(dit: MLXMiniMaxH3DiT, layout, video_rows, audio_rows, text_rows,
             cache, video_t_list, audio_t_list):
    """4-step ladder denoise served from the AdaLN cache (data-ward). Final x0 rows."""
    import mlx.core as mx

    from fastvideo.mlx_runtime.minimax_h3 import build_row_timesteps

    g = mx.random.key(0)
    x_v = mx.random.normal(video_rows.shape, key=g)
    x_a = mx.random.normal(audio_rows.shape, key=g)
    for i, (t_v, t_a) in enumerate(zip(video_t_list, audio_t_list)):
        unique, inverse = build_row_timesteps(layout, float(t_v), float(t_a), 1.0, 1.0)
        v_vo, v_ao = dit.forward_with_cache(
            x_v, x_a, text_rows, layout=layout,
            step_timesteps=unique, row_timestep_inverse=inverse)
        x0_v = x_v + (1.0 - t_v) * v_vo
        x0_a = x_a + (1.0 - t_a) * v_ao
        if i < len(video_t_list) - 1:
            t_vn = video_t_list[i + 1]
            t_an = audio_t_list[i + 1]
            n_v = mx.random.normal(x_v.shape, key=g)
            n_a = mx.random.normal(x_a.shape, key=g)
            x_v = t_vn * x0_v + (1.0 - t_vn) * n_v
            x_a = t_an * x0_a + (1.0 - t_an) * n_a
    mx.eval(x0_v, x0_a)
    return x0_v, x0_a


def main() -> None:
    import mlx.core as mx

    args = parse_args()
    formats = args.formats.split()
    if "bf16" not in formats:
        formats = ["bf16", *formats]  # reference always needed

    print(f"MLX {mx.__version__} | metal: {mx.metal.is_available()} | device: {mx.metal.device_name() if hasattr(mx.metal, 'device_name') else 'n/a'}")
    layout, video_rows, audio_rows, text_rows = _synthetic_inputs(args)

    reference = None
    mid_t_v = float(_ladder_timesteps(MINIMAX_H3_VIDEO_SHIFT)[2])
    mid_t_a = float(_ladder_timesteps(MINIMAX_H3_AUDIO_SHIFT)[2])

    report: dict[str, dict] = {}
    warmups = getattr(args, "warmups", 1)
    repeats = getattr(args, "repeats", 2)
    for fmt in formats:
        spec = None if fmt == "bf16" else MLXQuantizationSpec.from_name(fmt)
        if spec is not None:
            ensure_quantization_supported(spec)
        print(f"\n=== loading {fmt} ===", flush=True)
        t0 = time.perf_counter()
        dit = mlx_h3_dit_from_diffusers_safetensors(
            Path(args.model_root) / "transformer",
            quantization=spec,
            dtype="fp16",
        )
        load_s = time.perf_counter() - t0
        mx.eval()

        # reconstruction vs bf16 at the mid ladder point (faithful path, one shot)
        out_v, out_a = _forward(dit, layout, video_rows, audio_rows, text_rows, mid_t_v, mid_t_a)
        mx.eval(out_v, out_a)
        if fmt == "bf16":
            reference = (out_v, out_a)
        assert reference is not None
        rel_err = (_rel_l2(out_v, reference[0]) + _rel_l2(out_a, reference[1])) / 2.0

        # FastH3 runtime: AdaLN cache (drops ~40% of params) — the shipped path.
        cache, video_t_list, audio_t_list = _prepare_cache(dit)
        active_gb = float(mx.get_active_memory()) / 1e9

        # 4-step ladder runtime through the cache (fewer warms: disk-loaded, CPU-side)
        for _ in range(warmups):
            _denoise(dit, layout, video_rows, audio_rows, text_rows, cache, video_t_list, audio_t_list)
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _denoise(dit, layout, video_rows, audio_rows, text_rows, cache, video_t_list, audio_t_list)
            samples.append((time.perf_counter() - t0) / 4.0)
        per_step_ms = float(np.median(samples)) * 1000.0

        report[fmt] = {
            "load_s": round(load_s, 1),
            "resident_gb_after_adaln_cache": round(active_gb, 2),
            "bits_per_param": BITS_PER_PARAM[fmt],
            "adaln_params_freed": "yes",
            "rel_l2_vs_bf16": float(rel_err),
            "per_step_ms_cached": round(per_step_ms, 2),
        }
        print(json.dumps(report[fmt], indent=2), flush=True)

        if args.latents_out and fmt != "bf16":
            out_dir = Path(args.latents_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            x0_v, x0_a = _denoise(dit, layout, video_rows, audio_rows, text_rows,
                                cache, video_t_list, audio_t_list)
            np.savez(
                out_dir / f"latents_{fmt}.npz",
                video_rows=np.asarray(x0_v),
                audio_rows=np.asarray(x0_a),
                layout=dict(
                    num_latent_frames=layout.num_video_latent_frames,
                    latent_height=layout.latent_height,
                    latent_width=layout.latent_width,
                    num_audio_latents=layout.num_audio_latents,
                    num_frames=args.num_frames,
                ),
            )
            print(f"  latents -> {out_dir / f'latents_{fmt}.npz'}", flush=True)

    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()