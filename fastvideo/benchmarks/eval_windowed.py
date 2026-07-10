# SPDX-License-Identifier: Apache-2.0
"""End-to-end windowed self-attention eval for the MLX Wan DiT.

Sweeps ``FASTVIDEO_MLX_WINDOW ∈ {0, 4096, 2048, 1024}`` (0 = full attention)
on a fixed-seed fox prompt at 480×832×81, int8 + TAEHV, and reports:

- ``denoise_steady_step_s`` and ``peak_gib`` per window
- MS-SSIM of each windowed clip vs the full-attention (``window=0``) clip

Does **not** reimplement the denoise pipeline: each cell reuses
``fastvideo.benchmarks.mlx_fastwan_bench._generate_cell`` (same load / DMD /
decode path) with only the env gate flipped.

Usage::

    source /Users/aryank/claude-fastvideo/FastVideo/.venv/bin/activate
    export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD
    python -m fastvideo.benchmarks.eval_windowed \\
        --model-root /Users/aryank/models/qad_int8_v2 \\
        --output-dir bench/window_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Ensure flag-off is the default until a cell explicitly sets a window.
os.environ.pop("FASTVIDEO_MLX_WINDOW", None)
os.environ.pop("FASTVIDEO_MLX_WINDOW_SINK", None)

from examples.inference.basic.mlx_wan_prompt_to_video import (  # noqa: E402
    encode_prompt,
    make_rotary_embeddings,
)
from fastvideo.benchmarks.mlx_fastwan_bench import (  # noqa: E402
    _generate_cell,
    _ms_ssim,
)
from fastvideo.mlx_runtime.memory import apply_memory_limits  # noqa: E402

FOX_PROMPT = "A fox runs through a misty pine forest, leaves kicking up behind it."
DEFAULT_WINDOWS = (0, 4096, 2048, 1024)
DEFAULT_SEED = 1024
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_MODE = "int8"
DEFAULT_DECODER = "taehv"


def _parse_windows(raw: str) -> list[int]:
    items: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            w = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid window value {part!r}: must be an int") from exc
        if w < 0:
            raise SystemExit(f"Invalid window value {w}: must be >= 0")
        items.append(w)
    if not items:
        raise SystemExit("No windows provided")
    if 0 not in items:
        raise SystemExit("windows must include 0 (full attention) as the SSIM reference")
    return items


def _set_window_env(window: int, sink: int) -> None:
    """Gate self-attention for the next DiT forward (read each layer call)."""
    if window <= 0:
        os.environ.pop("FASTVIDEO_MLX_WINDOW", None)
        os.environ.pop("FASTVIDEO_MLX_WINDOW_SINK", None)
    else:
        os.environ["FASTVIDEO_MLX_WINDOW"] = str(window)
        os.environ["FASTVIDEO_MLX_WINDOW_SINK"] = str(sink)


def _video_name(window: int, height: int, width: int, num_frames: int) -> str:
    tag = "full" if window <= 0 else f"w{window}"
    return f"fox_{tag}_{height}x{width}x{num_frames}.mp4"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Honest end-to-end eval of flag-gated windowed self-attention on MLX Wan DiT."
    )
    parser.add_argument("--model-root", type=Path, default=Path("/Users/aryank/models/qad_int8_v2"))
    parser.add_argument("--prompt", default=FOX_PROMPT)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=("fp16", "bf16", "int8", "int4"))
    parser.add_argument("--decoder", default=DEFAULT_DECODER, choices=("taehv", "wan-vae"))
    parser.add_argument(
        "--windows",
        default=",".join(str(w) for w in DEFAULT_WINDOWS),
        help="Comma-separated window sizes; must include 0 (full attention).",
    )
    parser.add_argument(
        "--sink",
        type=int,
        default=0,
        help="FASTVIDEO_MLX_WINDOW_SINK for windowed cells (ignored when window=0).",
    )
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/window_eval"))
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--taehv-source-path", type=Path, default=None)
    parser.add_argument("--taehv-checkpoint-path", type=Path, default=None)
    parser.add_argument("--taehv-parallel", action="store_true")
    parser.add_argument(
        "--mlx-checkpoint-cache",
        type=Path,
        default=None,
        help="Optional per-mode MLX checkpoint cache (same as mlx_fastwan_bench).",
    )
    args = parser.parse_args(argv)

    if args.sink < 0:
        raise SystemExit(f"--sink must be >= 0, got {args.sink}")

    windows = _parse_windows(args.windows)
    # Full attention first so the SSIM reference exists before windowed cells.
    windows = [0] + [w for w in windows if w != 0]

    model_root = args.model_root.expanduser().resolve()
    config_path = model_root / "transformer" / "config.json"
    checkpoint_path = model_root / "transformer" / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file():
        raise SystemExit(f"Missing DiT config: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing DiT checkpoint: {checkpoint_path}")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Flag-off until the first windowed cell; also ensures import-time paths are clean.
    _set_window_env(0, 0)

    import mlx.core as mx
    import torch

    apply_memory_limits(mx_module=mx)
    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.loads(config_path.read_text())
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
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
    if not timesteps:
        raise SystemExit("No DMD timesteps parsed from --dmd-denoising-steps")
    renoise_by_step = [
        torch.randn(latents_seed.shape, generator=generator, dtype=torch.float32).numpy()
        for _ in range(max(0, len(timesteps) - 1))
    ]

    print(f"Encoding prompt once (seed={args.seed}) …", flush=True)
    prompt_embeds = encode_prompt(
        model_root=model_root,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
        device_arg=args.torch_device,
        dtype_arg=args.torch_dtype,
    )
    encoder_hidden_states = mx.array(prompt_embeds.numpy())

    # Namespace expected by mlx_fastwan_bench._generate_cell.
    cell_args = SimpleNamespace(
        model_root=model_root,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        flow_shift=args.flow_shift,
        torch_device=args.torch_device,
        torch_dtype=args.torch_dtype,
        taehv_source_path=args.taehv_source_path,
        taehv_checkpoint_path=args.taehv_checkpoint_path,
        taehv_parallel=args.taehv_parallel,
        mlx_checkpoint_cache=args.mlx_checkpoint_cache,
        mlx_memory_limit_gib=None,
        mlx_cache_limit_gib=None,
        mlx_disable_cache=False,
        mlx_wired_limit_gib=None,
        torch_mps_high_watermark_ratio=None,
        torch_mps_low_watermark_ratio=None,
        benchmark_preset="window-eval",
        current_prompt_id="fox-forest",
        current_prompt=args.prompt,
        output_dir=args.output_dir,
    )

    rows: list[dict] = []
    ref_video: Path | None = None
    full_steady: float | None = None

    for window in windows:
        _set_window_env(window, args.sink)
        env_window = os.environ.get("FASTVIDEO_MLX_WINDOW", "0")
        tag = "full" if window <= 0 else f"w{window}"
        print(
            f"\n=== window={window} (env FASTVIDEO_MLX_WINDOW={env_window}) "
            f"mode={args.mode} decoder={args.decoder} ===",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            cell = _generate_cell(
                args=cell_args,
                mode=args.mode,
                decoder=args.decoder,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                encoder_hidden_states=encoder_hidden_states,
                freqs_cis=freqs_cis,
                timesteps=timesteps,
                latents_seed=latents_seed,
                renoise_by_step=renoise_by_step,
            )
        except Exception as exc:  # noqa: BLE001 - surface and record; continue other windows
            err = f"{type(exc).__name__}: {exc}"
            print(f"ERROR window={window}: {err}", file=sys.stderr, flush=True)
            rows.append(
                {
                    "window": window,
                    "status": "error",
                    "error": err,
                    "mode": args.mode,
                    "decoder": args.decoder,
                    "denoise_steady_step_s": None,
                    "peak_gib": None,
                    "ms_ssim_vs_full": None,
                    "speedup_vs_full": None,
                    "video_path": None,
                }
            )
            continue
        wall_s = time.perf_counter() - t0

        # Stable name under bench/window_eval/ (bench writes nested prompt_id/ by default).
        target_name = _video_name(window, args.height, args.width, args.num_frames)
        target_path = args.output_dir / target_name
        if cell.video_path.resolve() != target_path.resolve():
            target_path.write_bytes(cell.video_path.read_bytes())
            # Keep original nested path too; summary points at the flat name.

        steady = cell.metrics.get("denoise_steady_step_s")
        peak = cell.metrics.get("peak_gib")
        if window == 0:
            ref_video = target_path
            full_steady = float(steady) if steady is not None else None
            ms_ssim: float | None = 1.0
            speedup: float | None = 1.0
        else:
            if ref_video is None or not ref_video.is_file():
                raise SystemExit("window=0 reference video missing; cannot score MS-SSIM")
            ms_ssim = _ms_ssim(ref_video, target_path, required=True)
            if full_steady is not None and steady is not None and float(steady) > 0:
                speedup = float(full_steady) / float(steady)
            else:
                speedup = None

        row = {
            "window": window,
            "status": "ok",
            "mode": args.mode,
            "decoder": args.decoder,
            "seed": args.seed,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "sink": args.sink if window > 0 else 0,
            "denoise_s": cell.metrics.get("denoise_s"),
            "denoise_first_step_s": cell.metrics.get("denoise_first_step_s"),
            "denoise_steady_step_s": steady,
            "decode_s": cell.metrics.get("decode_s"),
            "load_s": cell.metrics.get("load_s"),
            "peak_gib": peak,
            "ms_ssim_vs_full": ms_ssim,
            "speedup_vs_full": speedup,
            "video_path": target_name,
            "wall_s": wall_s,
            "fastvideo_mlx_window_env": env_window if window > 0 else "0/unset",
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    # Reset env so subsequent imports in the same process are clean.
    _set_window_env(0, 0)

    summary = {
        "model_root": str(model_root),
        "mode": args.mode,
        "decoder": args.decoder,
        "seed": args.seed,
        "prompt": args.prompt,
        "shape": f"{args.height}x{args.width}x{args.num_frames}",
        "sink": args.sink,
        "windows": windows,
        "rows": rows,
        "table": [
            {
                "window": r["window"],
                "s_per_step": r.get("denoise_steady_step_s"),
                "speedup_vs_full": r.get("speedup_vs_full"),
                "peak_gib": r.get("peak_gib"),
                "ms_ssim_vs_full": r.get("ms_ssim_vs_full"),
                "status": r.get("status"),
            }
            for r in rows
        ],
        "notes": (
            "Sequence is a flattened (frame × H × W) grid; 1D window is a crude receptive "
            "field. Low MS-SSIM likely needs a 3D-aware local window, not abandoning the idea."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Human-readable table on stdout.
    print("\n| window | s/step | speedup vs full | peak GiB | ms_ssim_vs_full | status |")
    print("|--------|--------|-----------------|----------|-----------------|--------|")
    for r in rows:
        def _f(v, digits=3):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.{digits}f}"
            return str(v)

        print(
            f"| {r['window']} | {_f(r.get('denoise_steady_step_s'))} | "
            f"{_f(r.get('speedup_vs_full'))} | {_f(r.get('peak_gib'))} | "
            f"{_f(r.get('ms_ssim_vs_full'), 4)} | {r.get('status')} |"
        )
    print(f"\nWrote {summary_path}")
    ok = all(r.get("status") == "ok" for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
