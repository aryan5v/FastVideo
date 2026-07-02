# SPDX-License-Identifier: Apache-2.0
"""Prove-out benchmark for the MLX FastWan runtime (Apple Silicon).

Sweeps ``{dtype/quant} x {decoder}``, generates a clip per cell, and records the
latency breakdown, peak unified memory, and MS-SSIM (optionally LPIPS) against a
reference video. It emits a JSON blob and a markdown table -- the artifact that
turns "int8 + TAEHV looks good" into defensible numbers, and (via
``--assert-min-ssim``) a regression gate for the ``mx.compile`` work.

Design notes:
- Generation reuses the hybrid POC helpers in
  ``examples/inference/basic/mlx_wan_prompt_to_video.py`` (torch-MPS UMT5 encode
  and Wan-VAE/TAEHV decode) plus the on-device MLX DMD sampler
  (``fastvideo/mlx_runtime/sampling.py``); the denoise loop never leaves the
  device.
- Quality reuses the tested MS-SSIM primitive
  ``fastvideo/tests/utils.py::compute_video_ssim_torchvision``.
- Reference: by default each cell is scored against the highest-fidelity cell
  in the sweep (``fp16`` + ``wan-vae``), which needs no CUDA box and answers
  "how much does int8/int4/TAEHV degrade vs the best local config". Pass
  ``--reference PATH`` to score against an external clip instead (e.g. the
  torch-MPS or CUDA FastVideo output of the same model) for a "vs. the original
  model" column.

Run on an Apple Silicon Mac (needs ``mlx`` + a torch build with MPS):

    python fastvideo/benchmarks/mlx_fastwan_bench.py \
        --modes fp16,bf16,int8,int4 --decoders taehv,wan-vae
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.inference.basic.mlx_wan_prompt_to_video import (
    DEFAULT_MODEL_ROOT,
    decode_latents_to_video,
    encode_prompt,
    make_rotary_embeddings,
)

# The highest-fidelity cell; used as the default SSIM reference when no external
# reference video is supplied.
REFERENCE_MODE = "fp16"
REFERENCE_DECODER = "wan-vae"

ALLOWED_MODES = ("fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4")
ALLOWED_DECODERS = ("taehv", "wan-vae")


@dataclass
class Cell:
    mode: str
    decoder: str
    video_path: Path
    latents: np.ndarray
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)


def _mode_to_dtype_quant(mode: str) -> tuple[str, str | None]:
    """Map a sweep mode to (MLX compute dtype, quantization spec).

    Quantized modes keep fp16 activations and quantize only the DiT linear
    weights (matching ``mlx_dit_from_diffusers_safetensors``).
    """
    if mode == "bf16":
        return "bf16", None
    if mode == "fp16":
        return "fp16", None
    # int8/int4/mxfp*/nvfp4 -> fp16 activations + quantized weights.
    return "fp16", mode


def _mx_dtype(mx, base: str):
    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[base]


def _parse_list(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {unknown} (allowed: {list(allowed)})")
    return items


def _denoise_dmd_on_device(
    *,
    mx,
    dit,
    latents,
    encoder_hidden_states,
    freqs_cis,
    timesteps: list[int],
    renoise_by_step: list[np.ndarray],
    schedule,
    dmd_step,
    mx_dtype,
) -> "np.ndarray":
    """Run the FastWan DMD loop entirely on the MLX device.

    Mirrors the loop in ``mlx_wan_prompt_to_video.py`` (fp32 affine math, MLX RNG
    re-noise) so the benchmark measures exactly the shipped path.
    """
    for step_index, timestep in enumerate(timesteps):
        noise_input_latent = latents
        timestep_mx = mx.array([float(timestep)]).astype(mx.float32)
        noise_pred = dit(latents.astype(mx_dtype), encoder_hidden_states, timestep_mx, freqs_cis)

        noise_input_f32 = noise_input_latent.astype(mx.float32)
        pred_noise_f32 = noise_pred.astype(mx.float32)
        if step_index < len(timesteps) - 1:
            next_ts: float | None = float(timesteps[step_index + 1])
            renoise = mx.array(renoise_by_step[step_index]).astype(mx.float32)
        else:
            next_ts, renoise = None, None
        latents = dmd_step(
            latents=noise_input_f32,
            noise_input_latent=noise_input_f32,
            pred_noise=pred_noise_f32,
            schedule=schedule,
            timestep=float(timestep),
            next_timestep=next_ts,
            noise=renoise,
        ).astype(mx_dtype)
        mx.eval(latents)
    return np.array(latents.astype(mx.float32))


def _peak_memory_bytes(mx) -> int:
    try:
        return int(mx.get_peak_memory())
    except Exception:  # noqa: BLE001 - best-effort telemetry only.
        return 0


def _latent_delta_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float32) - baseline.astype(np.float32)
    mse = float(np.mean(np.square(diff)))
    signal = float(np.mean(np.square(baseline.astype(np.float32))))
    return {
        "latent_mse_vs_ref_mode": mse,
        "latent_snr_db_vs_ref_mode": float(10.0 * np.log10(signal / mse)) if mse > 0 else float("inf"),
    }


def _ms_ssim(reference_video: Path, candidate_video: Path, *, required: bool = False) -> float | None:
    """Mean MS-SSIM between two mp4s, via the repo's tested helper."""
    if not reference_video.exists() or not candidate_video.exists():
        return None
    try:
        from fastvideo.tests.utils import compute_video_ssim_torchvision
    except ImportError as exc:
        message = (
            "MS-SSIM is unavailable because `pytorch-msssim` is not installed. "
            "Install FastVideo with the test extra, e.g. `uv pip install -e '.[mlx,test]'`, "
            "or run without an SSIM assertion."
        )
        if required:
            raise RuntimeError(message) from exc
        print(f"{message} Skipping MS-SSIM.")
        return None

    ssim_values = compute_video_ssim_torchvision(str(reference_video), str(candidate_video), use_ms_ssim=True)
    return float(ssim_values[0])


def _markdown_table(rows: list[dict]) -> str:
    columns = [
        ("mode", "mode"),
        ("decoder", "decoder"),
        ("denoise_s", "denoise s"),
        ("decode_s", "decode s"),
        ("total_s", "total s"),
        ("peak_gib", "peak GiB"),
        ("ms_ssim_vs_ref", "MS-SSIM"),
        ("lpips_vs_ref", "LPIPS"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif value is None:
                cells.append("-")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _generate_cell(
    *,
    args,
    mode: str,
    decoder: str,
    checkpoint_path: Path,
    config_path: Path,
    encoder_hidden_states,
    freqs_cis,
    timesteps: list[int],
    latents_seed: np.ndarray,
    renoise_by_step: list[np.ndarray],
) -> Cell:
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import mlx_dit_from_diffusers_safetensors
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    base_dtype, quantization = _mode_to_dtype_quant(mode)
    mx_dtype = _mx_dtype(mx, base_dtype)

    mx.clear_cache()
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    dit = mlx_dit_from_diffusers_safetensors(
        checkpoint_path,
        config_path,
        dtype=base_dtype,
        quantization=quantization,
    )
    load_s = time.perf_counter() - load_start
    load_peak = _peak_memory_bytes(mx)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
    schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)

    latents = mx.array(latents_seed).astype(mx_dtype)
    mx.reset_peak_memory()
    denoise_start = time.perf_counter()
    latents_np = _denoise_dmd_on_device(
        mx=mx,
        dit=dit,
        latents=latents,
        encoder_hidden_states=encoder_hidden_states.astype(mx_dtype),
        freqs_cis=freqs_cis,
        timesteps=timesteps,
        renoise_by_step=renoise_by_step,
        schedule=schedule,
        dmd_step=dmd_step,
        mx_dtype=mx_dtype,
    )
    denoise_s = time.perf_counter() - denoise_start
    denoise_peak = _peak_memory_bytes(mx)

    video_path = args.output_dir / f"video_{mode}_{decoder}_{args.height}x{args.width}x{args.num_frames}.mp4"
    decode_start = time.perf_counter()
    decode_latents_to_video(
        model_root=args.model_root,
        latents_np=latents_np,
        output_path=video_path,
        fps=args.fps,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
        backend=decoder,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
    )
    decode_s = time.perf_counter() - decode_start

    metrics: dict[str, float | int | str | bool | None] = {
        "mode": mode,
        "decoder": decoder,
        "load_s": load_s,
        "denoise_s": denoise_s,
        "decode_s": decode_s,
        "total_s": load_s + denoise_s + decode_s,
        "load_peak_gib": load_peak / (1024**3),
        "peak_gib": max(load_peak, denoise_peak) / (1024**3),
        "quantization": quantization or "none",
        "compute_dtype": base_dtype,
        "compile": os.environ.get("FASTVIDEO_MLX_COMPILE", "0") == "1",
        "fast_norm": os.environ.get("FASTVIDEO_MLX_FAST_NORM", "0") == "1",
    }
    return Cell(mode=mode, decoder=decoder, video_path=video_path, latents=latents_np, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX FastWan prove-out benchmark (latency + quality).")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prompt", default="A paper boat sails through a shallow stream in a mossy forest.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--modes", default="fp16,bf16,int8,int4")
    parser.add_argument("--decoders", default="taehv,wan-vae")
    parser.add_argument("--output-dir", type=Path, default=Path("video_samples/mlx_fastwan_bench"))
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="External reference mp4 to score every cell against. Defaults to the fp16+wan-vae cell.",
    )
    parser.add_argument("--assert-min-ssim", type=float, default=None,
                        help="Fail if any cell's MS-SSIM vs the reference falls below this value.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable mx.compile on the DiT forward (sets FASTVIDEO_MLX_COMPILE=1).")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS (needs the `lpips` package).")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    args = parser.parse_args()

    if args.compile:
        os.environ["FASTVIDEO_MLX_COMPILE"] = "1"

    import mlx.core as mx
    import torch

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modes = _parse_list(args.modes, ALLOWED_MODES, "modes")
    decoders = _parse_list(args.decoders, ALLOWED_DECODERS, "decoders")

    config_path = args.model_root / "transformer/config.json"
    checkpoint_path = args.model_root / "transformer/diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8

    # Shared prompt encode + rotary + initial noise (identical across all cells).
    prompt_embeds = encode_prompt(
        model_root=args.model_root,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
    )
    encoder_hidden_states = mx.array(prompt_embeds.numpy())
    freqs_cis = make_rotary_embeddings(
        config,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_seed = torch.randn(
        (1, int(config["in_channels"]), latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    ).numpy()

    timesteps = [int(step.strip()) for step in args.dmd_denoising_steps.split(",") if step.strip()]
    # Keep DMD stochasticity identical across benchmark cells. Without this,
    # FP16/INT8/decoder comparisons can accidentally measure different re-noise
    # samples instead of only quantization or decoder differences.
    renoise_by_step = [
        torch.randn(latents_seed.shape, generator=generator, dtype=torch.float32).numpy()
        for _ in range(max(0, len(timesteps) - 1))
    ]

    cells: list[Cell] = []
    for mode in modes:
        for decoder in decoders:
            print(f"=== cell: mode={mode} decoder={decoder} ===")
            cells.append(
                _generate_cell(
                    args=args,
                    mode=mode,
                    decoder=decoder,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    encoder_hidden_states=encoder_hidden_states,
                    freqs_cis=freqs_cis,
                    timesteps=timesteps,
                    latents_seed=latents_seed,
                    renoise_by_step=renoise_by_step,
                )
            )

    # Resolve the SSIM/latent reference: external clip, else the fp16+wan-vae cell
    # (or the first cell if that combination was not swept).
    reference_video = args.reference
    reference_latents = None
    if reference_video is None:
        ref_cell = next(
            (c for c in cells if c.mode == REFERENCE_MODE and c.decoder == REFERENCE_DECODER),
            cells[0],
        )
        reference_video = ref_cell.video_path
        reference_latents = ref_cell.latents
        print(f"Using internal reference cell: mode={ref_cell.mode} decoder={ref_cell.decoder}")

    lpips_fn = _load_lpips() if args.lpips else None

    rows: list[dict] = []
    failures: list[str] = []
    for cell in cells:
        ms_ssim = _ms_ssim(Path(reference_video), cell.video_path, required=args.assert_min_ssim is not None)
        cell.metrics["ms_ssim_vs_ref"] = ms_ssim
        if reference_latents is not None:
            cell.metrics.update(_latent_delta_metrics(cell.latents, reference_latents))
        cell.metrics["lpips_vs_ref"] = (
            _lpips_between(lpips_fn, Path(reference_video), cell.video_path) if lpips_fn else None
        )
        if args.assert_min_ssim is not None and ms_ssim is not None and ms_ssim < args.assert_min_ssim:
            failures.append(f"{cell.mode}/{cell.decoder}: MS-SSIM {ms_ssim:.4f} < {args.assert_min_ssim}")
        rows.append(dict(cell.metrics))
        print(json.dumps(cell.metrics, indent=2))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2))
    table_path = args.output_dir / "metrics.md"
    table = _markdown_table(rows)
    table_path.write_text(table + "\n")
    print("\n" + table)
    print(f"\nWrote {metrics_path} and {table_path}")

    if failures:
        raise SystemExit("SSIM regression gate failed:\n  " + "\n  ".join(failures))


def _load_lpips():
    """Return an LPIPS model, or ``None`` if the optional dep is unavailable."""
    try:
        import lpips  # noqa: PLC0415 - optional dependency.
    except ImportError:
        print("LPIPS requested but the `lpips` package is not installed; skipping (install `.[eval]`).")
        return None
    return lpips.LPIPS(net="alex")


def _lpips_between(lpips_fn, reference_video: Path, candidate_video: Path) -> float | None:
    if lpips_fn is None or not reference_video.exists() or not candidate_video.exists():
        return None
    import torch

    ref = _read_video_frames(reference_video)
    cand = _read_video_frames(candidate_video)
    if ref is None or cand is None or ref.shape != cand.shape:
        return None
    # LPIPS expects NCHW in [-1, 1].
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    cand_t = torch.from_numpy(cand).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad():
        scores = lpips_fn(ref_t, cand_t)
    return float(scores.mean().item())


def _read_video_frames(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    main()
