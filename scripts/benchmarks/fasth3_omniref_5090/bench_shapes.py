# SPDX-License-Identifier: Apache-2.0
"""RTX 5090 OmniRef fast path: one generator, warm + measured run per shape."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHAPES = "832x480x124,832x480x243,1344x768x124,1344x768x243"


def load_example():
    path = ROOT / "examples/inference/basic/basic_fasth3_omniref_pdd.py"
    spec = importlib.util.spec_from_file_location("omniref_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--adaln-table", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--warm", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--prefix", default="", choices=("", "fp8", "bf16"))
    parser.add_argument("--lossless-accel", action="store_true")
    parser.add_argument("--compile-vae", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    os.environ.setdefault("FASTVIDEO_NVFP4_MM_BACKEND", "cudnn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")
    os.environ["FASTVIDEO_H3_VSA_FP4_PREFIX"] = args.prefix

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig
    from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

    example = load_example()
    profile = example.DEVICE_PROFILES["rtx-5090"]
    example.apply_profile(profile, args.decoder, args.adaln_table)
    if args.lossless_accel:
        example.apply_lossless_accel_env()
    config = example.build_generator_config(args.model, 1, lossless_accel=args.lossless_accel, profile=profile)
    if args.compile_vae:
        import dataclasses

        from fastvideo.api import CompileConfig
        config = dataclasses.replace(
            config, engine=dataclasses.replace(config.engine, compile=CompileConfig(enabled=True, vae_enabled=True)))
    contract = example.load_contract(args.model)
    reference = MiniMaxH3Reference(source=str(ROOT / "assets/images/couple.jpg"), media_type="image")
    prompt = "A person on a sofa, lit by a warm lamp, speaking to camera."
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "model": str(args.model),
        "decoder": args.decoder,
        "adaln_table": args.adaln_table,
        "prefix": args.prefix,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "contract": contract,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True).strip(),
        "command": [sys.executable, *sys.argv],
        "env": {k: v for k, v in os.environ.items() if k.startswith(("FASTVIDEO_", "PYTORCH_", "CUDNN_"))},
        "seed": 1234,
        "warm": args.warm,
        "reps": args.reps,
        "runs": [],
    }
    destination = args.output / "results.json"

    def save() -> None:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(destination)

    generator = None
    try:
        started = time.perf_counter()
        generator = VideoGenerator.from_config(config)
        results["load_s"] = time.perf_counter() - started
        save()
        for shape_text in args.shapes.split(","):
            width, height, frames = (int(v) for v in shape_text.lower().split("x"))
            for run in range(args.warm + args.reps):
                tag = f"_warm{run}" if run < args.warm else f"_r{run - args.warm}"
                path = args.output / f"{width}x{height}x{frames}{tag}.mp4"
                request = GenerationRequest(
                    prompt=prompt,
                    negative_prompt="",
                    inputs=InputConfig(references=[reference]),
                    sampling=SamplingConfig(seed=1234, height=height, width=width, num_frames=frames, fps=24,
                                            num_inference_steps=contract["num_inference_steps"],
                                            guidance_scale=1.0, batch_cfg=False),
                    output=OutputConfig(output_path=str(path), save_video=True, return_frames=False),
                )
                started = time.perf_counter()
                result = generator.generate(request)
                elapsed = time.perf_counter() - started
                stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
                record = {
                    "shape": f"{width}x{height}x{frames}",
                    "tag": tag,
                    "warmup": run < args.warm,
                    "wall_s": round(elapsed, 2),
                    "output": str(result.video_path),
                    "peak_alloc_mb": result.peak_memory_mb,
                    "stages_s": {
                        name: round(float(metrics["execution_time"]), 3)
                        for name, metrics in stages.items() if metrics.get("execution_time") is not None
                    },
                }
                results["runs"].append(record)
                save()
                print("RESULT " + json.dumps(record), flush=True)
    except BaseException as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        save()


if __name__ == "__main__":
    main()
