# SPDX-License-Identifier: Apache-2.0
"""Memory / latency prove-out for Wan2.2-TI2V-5B on Apple Silicon (Track D Rung 3).

Loads the released 5B Diffusers transformer in fp16 and int8, runs a few 3-step
DMD denoise steps at a product-class latent shape, and records weight footprint,
peak unified memory, and steady per-step latency. Denoise-side only — random
text embeds (content-independent). Decode is noted separately: the 5B VAE is
z_dim=48; TAEHV taew2_1.pth targets Wan2.1 and is not a drop-in for 2.2.

    PYTHONPATH=$PWD python -m fastvideo.benchmarks.mlx_wan22_5b_bench \
      --model-root ~/models/fastwan22_5b --modes fp16,int8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _mx_dtype(name: str):
    import mlx.core as mx

    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]


def _weight_bytes(model) -> int:
    """Approximate resident weight bytes (plain + quantized matrices)."""
    total = 0
    for value in model.weights.values():
        total += _array_nbytes(value)
    for block in model.blocks:
        for value in block.weights.values():
            total += _array_nbytes(value)
    return total


def _array_nbytes(value) -> int:
    # QuantizedMatrix from fastwan has .weight / .scales / .biases.
    if hasattr(value, "weight") and hasattr(value, "scales"):
        n = int(np.prod(value.weight.shape)) * _dtype_itemsize(value.weight)
        n += int(np.prod(value.scales.shape)) * _dtype_itemsize(value.scales)
        if getattr(value, "biases", None) is not None:
            n += int(np.prod(value.biases.shape)) * _dtype_itemsize(value.biases)
        return n
    try:
        return int(np.prod(value.shape)) * _dtype_itemsize(value)
    except Exception:  # noqa: BLE001
        return 0


def _dtype_itemsize(arr) -> int:
    name = str(getattr(arr, "dtype", "float16"))
    if "64" in name:
        return 8
    if "32" in name:
        return 4
    if "16" in name or "bfloat" in name:
        return 2
    if "8" in name or "uint" in name or "int8" in name:
        return 1
    return 2


def _build_per_token_timestep(mx, *, batch: int, frames: int, height: int, width: int, patch_size, video_t: float):
    """2-D timestep: frame 0 at 0 (I2V image lock), remaining tokens at ``video_t``."""
    p_t, p_h, p_w = patch_size
    tokens_per_frame = (height // p_h) * (width // p_w)
    num_tokens = (frames // p_t) * tokens_per_frame
    levels = [0.0] + [float(video_t)] * (frames // p_t - 1)
    flat = [levels[i // tokens_per_frame] for i in range(num_tokens)]
    return mx.array(np.array([flat] * batch, dtype=np.float32))


def run_mode(
    *,
    model_root: Path,
    mode: str,
    height: int,
    width: int,
    num_frames: int,
    dmd_steps: list[int],
    flow_shift: float,
    seed: int,
) -> dict:
    import mlx.core as mx

    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step, pred_noise_to_pred_video
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    import torch

    config = json.loads((model_root / "transformer" / "config.json").read_text())
    # Wan2.2 VAE is 16× spatial (z_dim=48); Diffusers TI2V-5B uses latent H=height/16.
    # For denoise-only we treat height/width as *pixel* size and derive latent size
    # from the VAE spatial factor 16 (config does not store it; z_dim=48 ⇒ 2.2 VAE).
    latent_h, latent_w = height // 16, width // 16
    # Latent temporal: TI2V keeps (num_frames-1)//4+1 style for 4× time compress.
    latent_frames = (num_frames - 1) // 4 + 1
    if latent_frames < 1:
        latent_frames = 1
    # Ensure divisible by patch temporal size.
    p_t = int(config["patch_size"][0])
    if latent_frames % p_t != 0:
        latent_frames += p_t - (latent_frames % p_t)

    quant = None if mode == "fp16" else mode
    mx.clear_cache()
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    model = mlx_wan22_dit_from_diffusers_safetensors(
        model_root / "transformer" / "diffusion_pytorch_model.safetensors",
        model_root / "transformer" / "config.json",
        dtype="fp16",
        quantization=quant,
    )
    # Force materialization of weights so peak memory reflects the load.
    probe = next(iter(model.weights.values()))
    if hasattr(probe, "weight"):
        mx.eval(probe.weight)
    else:
        mx.eval(probe)
    load_s = time.perf_counter() - load_start
    load_peak = mx.get_peak_memory() / (1024**3)
    weight_gib = _weight_bytes(model) / (1024**3)

    text_dim = int(config["text_dim"])
    text_len = 64  # short random embeds (latency is content-independent)
    rng = np.random.default_rng(seed)
    text = mx.array((rng.standard_normal((1, text_len, text_dim)) * 0.1).astype(np.float16))
    noise = mx.array(
        rng.standard_normal((1, int(config["in_channels"]), latent_frames, latent_h, latent_w)).astype(np.float16))
    freqs_cis = make_rotary_embeddings(config,
                                       latent_frames=latent_frames,
                                       latent_height=latent_h,
                                       latent_width=latent_w)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(1000, device="cpu")
    schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)
    steps = torch.tensor(dmd_steps, dtype=torch.long)
    warped = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    timesteps = [float(t) for t in warped[1000 - steps]]

    # Warmup one step so compile/caches settle before timing.
    ts0 = _build_per_token_timestep(mx,
                                    batch=1,
                                    frames=latent_frames,
                                    height=latent_h,
                                    width=latent_w,
                                    patch_size=tuple(config["patch_size"]),
                                    video_t=timesteps[0])
    _ = model(noise, text, ts0, freqs_cis)
    mx.eval(_)

    mx.reset_peak_memory()
    step_latencies: list[float] = []
    latents = noise
    for i, ts_val in enumerate(timesteps):
        ts = _build_per_token_timestep(mx,
                                       batch=1,
                                       frames=latent_frames,
                                       height=latent_h,
                                       width=latent_w,
                                       patch_size=tuple(config["patch_size"]),
                                       video_t=ts_val)
        t0 = time.perf_counter()
        pred = model(latents, text, ts, freqs_cis)
        mx.eval(pred)
        step_latencies.append(time.perf_counter() - t0)
        if i < len(timesteps) - 1:
            renoise = mx.random.normal(latents.shape).astype(mx.float32)
            latents = dmd_step(
                latents=latents.astype(mx.float32),
                noise_input_latent=latents.astype(mx.float32),
                pred_noise=pred.astype(mx.float32),
                schedule=schedule,
                timestep=ts_val,
                next_timestep=timesteps[i + 1],
                noise=renoise,
            ).astype(mx.float16)
        else:
            latents = pred_noise_to_pred_video(pred.astype(mx.float32), latents.astype(mx.float32),
                                               schedule.sigma_for(ts_val)).astype(mx.float16)
        mx.eval(latents)

    denoise_peak = mx.get_peak_memory() / (1024**3)
    steady = statistics.median(step_latencies)
    return {
        "mode": mode,
        "quantization": quant or "none",
        "pixel_hw": f"{height}x{width}",
        "latent_shape": f"1x{config['in_channels']}x{latent_frames}x{latent_h}x{latent_w}",
        "dmd_steps": dmd_steps,
        "flow_shift": flow_shift,
        "load_s": round(load_s, 2),
        "weight_gib": round(weight_gib, 3),
        "load_peak_gib": round(load_peak, 3),
        "denoise_peak_gib": round(denoise_peak, 3),
        "step_latencies_s": [round(x, 3) for x in step_latencies],
        "steady_step_s": round(steady, 3),
        "total_denoise_s": round(sum(step_latencies), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2-5B MLX memory/latency benchmark.")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path.home() / "models" / "fastwan22_5b",
        help="Root with transformer/config.json + diffusion_pytorch_model.safetensors",
    )
    parser.add_argument("--modes", default="fp16,int8", help="Comma-separated: fp16,int8,...")
    parser.add_argument("--height", type=int, default=480, help="Pixel height (latent = H/16).")
    parser.add_argument("--width", type=int, default=832, help="Pixel width (latent = W/16).")
    parser.add_argument("--num-frames", type=int, default=33, help="Pixel frames (latent T ≈ (F-1)/4+1).")
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Wan2.2-TI2V default shift.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args()

    ckpt = args.model_root / "transformer" / "diffusion_pytorch_model.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    dmd_steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for mode in modes:
        print(f"\n=== mode={mode} ===", flush=True)
        row = run_mode(
            model_root=args.model_root,
            mode=mode,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            dmd_steps=dmd_steps,
            flow_shift=args.flow_shift,
            seed=args.seed,
        )
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)

    payload = {
        "model":
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "model_root":
        str(args.model_root),
        "rows":
        rows,
        "decode_note": ("Wan2.2 VAE z_dim=48; TAEHV taew2_1.pth is Wan2.1-only. Full Wan2.2 VAE "
                        "decode on torch-MPS is the decode path until a 2.2-compatible TAE lands; "
                        "chunked/tiled decode may be needed for memory on 32 GB."),
    }
    print("\n=== summary ===")
    print(json.dumps(payload, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
