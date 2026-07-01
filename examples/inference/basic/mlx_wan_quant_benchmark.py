"""Benchmark MLX FastWan quantization modes with one shared prompt encode."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.inference.basic.mlx_wan_prompt_to_video import (
    DEFAULT_MODEL_ROOT,
    decode_latents_to_video,
    encode_prompt,
    make_rotary_embeddings,
)


def _parse_modes(raw: str) -> list[str]:
    modes = [mode.strip() for mode in raw.split(",") if mode.strip()]
    allowed = {"none", "int8", "int4", "mxfp8", "mxfp4", "nvfp4"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}")
    return modes


def _latent_delta_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float32) - baseline.astype(np.float32)
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    signal = float(np.mean(np.square(baseline.astype(np.float32))))
    return {
        "latent_mse_vs_fp16": mse,
        "latent_mae_vs_fp16": mae,
        "latent_max_abs_vs_fp16": max_abs,
        "latent_snr_db_vs_fp16": float(10.0 * np.log10(signal / mse)) if mse > 0 else float("inf"),
    }


def _torch_mps_memory() -> dict[str, int | None]:
    try:
        import torch
    except ImportError:
        return {
            "torch_mps_current_allocated_bytes": None,
            "torch_mps_driver_allocated_bytes": None,
            "torch_mps_recommended_max_bytes": None,
        }
    if not torch.backends.mps.is_available():
        return {
            "torch_mps_current_allocated_bytes": None,
            "torch_mps_driver_allocated_bytes": None,
            "torch_mps_recommended_max_bytes": None,
        }
    return {
        "torch_mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "torch_mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "torch_mps_recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _decode_with_metrics(*, args, latents: np.ndarray, output_path: Path) -> dict[str, float | int | None | str]:
    before = _torch_mps_memory()
    decode_start = time.perf_counter()
    decode_latents_to_video(
        model_root=args.model_root,
        latents_np=latents,
        output_path=output_path,
        fps=args.fps,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
        backend=args.decode_backend,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
    )
    decode_time = time.perf_counter() - decode_start
    after = _torch_mps_memory()
    return {
        "decode_export_s": decode_time,
        "decode_torch_mps_current_before_bytes": before["torch_mps_current_allocated_bytes"],
        "decode_torch_mps_current_after_bytes": after["torch_mps_current_allocated_bytes"],
        "decode_torch_mps_driver_before_bytes": before["torch_mps_driver_allocated_bytes"],
        "decode_torch_mps_driver_after_bytes": after["torch_mps_driver_allocated_bytes"],
        "decode_torch_mps_recommended_max_bytes": after["torch_mps_recommended_max_bytes"],
    }


def _run_one_mode(
    *,
    mode: str,
    args,
    config: dict,
    checkpoint_path: Path,
    config_path: Path,
    prompt_embeds,
    freqs_cis,
):
    import mlx.core as mx
    import torch

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.models.utils import pred_noise_to_pred_video
    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors

    mx_dtype = mx.float16 if args.mlx_dtype == "fp16" else mx.float32
    quantization = None if mode == "none" else mode
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8

    load_start = time.perf_counter()
    mx.clear_cache()
    mx.reset_peak_memory()
    dit = mlx_dit_from_diffusers_safetensors(
        checkpoint_path,
        config_path,
        dtype=args.mlx_dtype,
        quantization=quantization,
    )
    load_time = time.perf_counter() - load_start
    load_peak_memory = mx.get_peak_memory()

    scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
    timesteps = torch.tensor([int(step.strip()) for step in args.dmd_denoising_steps.split(",") if step.strip()])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_torch = torch.randn(
        (1, int(config["in_channels"]), latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    latents = mx.array(latents_torch.numpy()).astype(mx_dtype)
    encoder_hidden_states = mx.array(prompt_embeds.numpy()).astype(mx_dtype)

    denoise_start = time.perf_counter()
    mx.reset_peak_memory()
    for step_index, timestep in enumerate(timesteps):
        noise_latents = latents
        timestep_mx = mx.array([float(timestep.item())]).astype(mx.float32)
        noise_pred = dit(latents.astype(mx_dtype), encoder_hidden_states, timestep_mx, freqs_cis)
        mx.eval(noise_pred)

        noise_pred_torch = torch.from_numpy(np.array(noise_pred.astype(mx.float32)))
        latents_torch = torch.from_numpy(np.array(latents.astype(mx.float32)))
        noise_latents_torch = torch.from_numpy(np.array(noise_latents.astype(mx.float32)))
        pred_video_btc = pred_noise_to_pred_video(
            pred_noise=noise_pred_torch.permute(0, 2, 1, 3, 4).flatten(0, 1),
            noise_input_latent=noise_latents_torch.permute(0, 2, 1, 3, 4).flatten(0, 1),
            timestep=timestep.repeat(noise_pred_torch.shape[0]),
            scheduler=scheduler,
        ).unflatten(0, (noise_pred_torch.shape[0], noise_pred_torch.shape[2]))
        if step_index < len(timesteps) - 1:
            next_timestep = timesteps[step_index + 1].reshape(1)
            noise = torch.randn(latents_torch.shape, generator=generator, dtype=latents_torch.dtype)
            latents_btc = scheduler.add_noise(
                pred_video_btc.flatten(0, 1),
                noise.permute(0, 2, 1, 3, 4).flatten(0, 1),
                next_timestep,
            ).unflatten(0, pred_video_btc.shape[:2])
            latents_torch = latents_btc.permute(0, 2, 1, 3, 4)
        else:
            latents_torch = pred_video_btc.permute(0, 2, 1, 3, 4)

        latents = mx.array(latents_torch.numpy()).astype(mx_dtype)
        mx.eval(latents)
    denoise_time = time.perf_counter() - denoise_start
    denoise_peak_memory = mx.get_peak_memory()
    active_memory = mx.get_active_memory()
    return {
        "mode": mode,
        "latents": np.array(latents.astype(mx.float32)),
        "metrics": {
            "mlx_dit_load_s": load_time,
            "mlx_denoise_s": denoise_time,
            "mlx_load_peak_bytes": int(load_peak_memory),
            "mlx_denoise_peak_bytes": int(denoise_peak_memory),
            "mlx_active_after_denoise_bytes": int(active_memory),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX FastWan quantization modes.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prompt", default="A snow leopard walks across a windy mountain ridge.")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--mlx-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--modes", default="none,int8,int4,mxfp8,mxfp4,nvfp4")
    parser.add_argument("--output-dir", type=Path, default=Path("video_samples/mlx_quant_benchmark"))
    parser.add_argument("--decode-backend", choices=("none", "wan-vae", "taehv"), default="taehv")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8

    prompt_start = time.perf_counter()
    prompt_embeds = encode_prompt(
        model_root=args.model_root,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
    )
    prompt_time = time.perf_counter() - prompt_start
    freqs_cis = make_rotary_embeddings(
        config,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    baseline_latents = None
    rows = []
    for mode in _parse_modes(args.modes):
        print(f"=== MLX quant mode: {mode} ===")
        mode_start = time.perf_counter()
        result = _run_one_mode(
            mode=mode,
            args=args,
            config=config,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
        )
        latents = result["latents"]
        if baseline_latents is None:
            baseline_latents = latents
        latent_path = args.output_dir / f"latents_{mode}.npy"
        np.save(latent_path, latents)

        decode_time = 0.0
        decode_metrics = {}
        output_path = None
        if args.decode_backend != "none":
            output_path = args.output_dir / f"video_{mode}_{args.decode_backend}_{args.height}x{args.width}x{args.num_frames}.mp4"
            decode_metrics = _decode_with_metrics(args=args, latents=latents, output_path=output_path)
            decode_time = float(decode_metrics["decode_export_s"])

        mode_total = time.perf_counter() - mode_start
        mlx_denoise_peak_bytes = int(result["metrics"]["mlx_denoise_peak_bytes"])
        mlx_active_bytes = int(result["metrics"]["mlx_active_after_denoise_bytes"])
        metrics = {
            "mode": mode,
            "prompt_encode_shared_s": prompt_time,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "decode_backend": args.decode_backend,
            "decode_export_s": decode_time,
            "mode_total_excluding_shared_prompt_s": mode_total,
            "mode_total_including_shared_prompt_s": mode_total + prompt_time,
            "latents_path": str(latent_path),
            "output_path": str(output_path) if output_path else None,
            "mlx_denoise_peak_gib": mlx_denoise_peak_bytes / (1024**3),
            "mlx_active_after_denoise_gib": mlx_active_bytes / (1024**3),
            "mlx_dit_peak_under_16gb": mlx_denoise_peak_bytes < 16 * 1024**3,
            "mlx_dit_active_under_16gb": mlx_active_bytes < 16 * 1024**3,
            "mac_16gb_status": (
                "dit_memory_fits_16gb_measured_decode_separately"
                if mlx_denoise_peak_bytes < 16 * 1024**3 else "dit_memory_exceeds_16gb"
            ),
            **result["metrics"],
            **decode_metrics,
            **_latent_delta_metrics(latents, baseline_latents),
        }
        rows.append(metrics)
        print(json.dumps(metrics, indent=2))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote benchmark metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
