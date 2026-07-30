"""Reproducible end-to-end A/B benchmark for the promoted Wan fusions.

Run this script once with ``--mode native`` and once with ``--mode fused``.
Each process loads the same checkpoint, performs one untimed warmup, measures
two generations with identical request parameters, and saves the first
measured frame tensor for an offline parity comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan fusion end-to-end A/B")
    parser.add_argument("--mode", choices=("native", "fused"), required=True)
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    os.environ["FASTVIDEO_WAN_FUSIONS"] = ("1" if args.mode == "fused" else "0")

    from fastvideo import VideoGenerator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    request = {
        "prompt": ("A curious raccoon peers through a vibrant field of yellow "
                   "sunflowers, soft natural light, steady cinematic camera."),
        "sampling": {
            "seed": 1024,
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "num_inference_steps": args.steps,
            "guidance_scale": 5.0,
        },
        "output": {
            "save_video": False,
            "return_frames": True,
        },
    }

    timings: list[float] = []
    generation_times: list[float | None] = []
    peak_memory: list[float | None] = []
    try:
        generator.generate(request)
        for run in range(args.runs):
            start = time.perf_counter()
            result = generator.generate(request)
            timings.append(time.perf_counter() - start)
            generation_times.append(result.generation_time)
            peak_memory.append(result.peak_memory_mb)
            if run == 0:
                np.save(
                    args.output_dir / f"{args.mode}_frames.npy",
                    np.asarray(result.frames),
                )
    finally:
        generator.shutdown()

    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "model": args.model,
        "request": request,
        "warmups": 1,
        "runs": args.runs,
        "wall_seconds": timings,
        "median_wall_seconds": statistics.median(timings),
        "generation_seconds": generation_times,
        "peak_memory_mb": peak_memory,
    }
    path = args.output_dir / f"{args.mode}_result.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
