# SPDX-License-Identifier: Apache-2.0
"""Deployment smoke for the Wan2.2-TI2V-5B DiT under a given MLX quant mode.

Proves the trained model's *deploy* path before we spend GPU hours: load the 5B
transformer in ``--quant`` (fp16 / int8 / mxfp4 / ...), run a few denoise forwards
on random latents + text embeds, and report peak memory, per-step time, and that
the output is finite. No text encoder / VAE needed — this validates the DiT path,
which is what quantization touches.

    python -m fastvideo.benchmarks.smoke_5b_quant --quant mxfp4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Wan2.2-5B MLX quant deployment smoke.")
    ap.add_argument("--quant", default="mxfp4",
                    help="fp16 | int8 | int4 | mxfp8 | mxfp4 | nvfp4 (fp16 = no quantization).")
    ap.add_argument("--model-root", type=Path, default=Path.home() / "models/fastwan22_5b")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import mlx.core as mx

    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings

    ckpt = args.model_root / "transformer" / "diffusion_pytorch_model.safetensors"
    cfg_path = args.model_root / "transformer" / "config.json"
    config = json.loads(cfg_path.read_text())

    # Wan2.2: 16x spatial, 4x temporal, 48 latent channels.
    lat_h, lat_w = args.height // 16, args.width // 16
    lat_t = (args.num_frames - 1) // 4 + 1
    in_ch = int(config["in_channels"])
    pt, ph, pw = tuple(config["patch_size"])
    tokens = lat_t * (lat_h // ph) * (lat_w // pw)
    quant = None if args.quant == "fp16" else args.quant

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    dit = mlx_wan22_dit_from_diffusers_safetensors(ckpt, cfg_path, dtype="fp16", quantization=quant)
    load_s = time.perf_counter() - t0
    load_gib = mx.get_peak_memory() / 1024**3
    print(f"[5b/{args.quant}] loaded in {load_s:.1f}s, load peak {load_gib:.2f} GiB, {tokens} tokens")

    rng = np.random.default_rng(0)
    latents = mx.array(rng.standard_normal((1, in_ch, lat_t, lat_h, lat_w)).astype(np.float32)).astype(mx.float16)
    text = mx.array((rng.standard_normal((1, 512, int(config["text_dim"]))) * 0.1).astype(np.float16))
    freqs = make_rotary_embeddings(config, latent_frames=lat_t, latent_height=lat_h, latent_width=lat_w)

    mx.reset_peak_memory()
    step_times: list[float] = []
    finite = True
    for i in range(args.steps):
        ts = mx.full((1, tokens), float(1000 - i * 300), dtype=mx.float32)
        t = time.perf_counter()
        out = dit(latents.astype(mx.float16), text, ts, freqs)
        mx.eval(out)
        step_times.append(time.perf_counter() - t)
        finite = finite and bool(mx.isfinite(out).all().item())
    peak_gib = mx.get_peak_memory() / 1024**3
    steady = min(step_times[1:]) if len(step_times) > 1 else step_times[0]

    report = {
        "quant": args.quant,
        "resolution": f"{args.height}x{args.width}x{args.num_frames}",
        "tokens": tokens,
        "load_s": round(load_s, 2),
        "load_peak_gib": round(load_gib, 3),
        "forward_peak_gib": round(peak_gib, 3),
        "step_first_s": round(step_times[0], 3),
        "step_steady_s": round(steady, 3),
        "output_finite": finite,
    }
    print(json.dumps(report, indent=2))
    if not finite:
        raise SystemExit("[5b] output not finite — quant path is broken")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
