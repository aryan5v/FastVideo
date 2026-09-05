# SPDX-License-Identifier: Apache-2.0
"""Run the locked FastH3 T2VA baseline matrix with persistent and W&B receipts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def _load_basic_fasth3() -> Any:
    """Load the repository example without colliding with site-packages ``examples``."""
    example_path = Path(__file__).resolve().parents[2] / "examples/inference/basic/basic_fasth3.py"
    spec = importlib.util.spec_from_file_location("fasth3_sprint_basic_inference", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FastH3 inference entrypoint from {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


basic_fasth3 = _load_basic_fasth3()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-role", required=True)
    parser.add_argument("--attention", choices=("dense", "vsa"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--steps", type=int, default=5,
                        help="Sigma-grid points; five points execute the released four-call schedule.")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--profile", choices=("strict", "all"), default="strict")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-vae", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fa4", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-videos", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _checkpoint_revision(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        parts = path.resolve().parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                return parts[index + 1]
        return "local"
    from huggingface_hub import HfApi
    return str(HfApi().model_info(model_path, token=os.environ.get("HF_TOKEN")).sha)


def _probe_media(path: Path) -> dict[str, Any]:
    # PyAV is part of the FastVideo runtime and validates actual decoding.  Do
    # not depend on a host ffprobe binary that is absent from the NGC image.
    import av

    with av.open(str(path)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise RuntimeError(f"Expected one video and one audio stream in {path}, "
                               f"got video={len(video_streams)} audio={len(audio_streams)}")
        video_stream = video_streams[0]
        audio_stream = audio_streams[0]
        video_context = video_stream.codec_context
        audio_context = audio_stream.codec_context
        channels = len(audio_context.layout.channels) if audio_context.layout is not None else 0
        sample_rate = int(audio_context.sample_rate or 0)
        width = int(video_context.width or 0)
        height = int(video_context.height or 0)
        video_index = int(video_stream.index)
        audio_index = int(audio_stream.index)
        video_codec_name = str(video_context.name)
        audio_codec_name = str(audio_context.name)
        duration = float(container.duration / av.time_base) if container.duration is not None else None
        decoded_video_frames = 0
        decoded_audio_frames = 0
        decoded_audio_samples = 0
        for packet in container.demux(video_stream, audio_stream):
            for frame in packet.decode():
                if packet.stream.type == "video":
                    decoded_video_frames += 1
                elif packet.stream.type == "audio":
                    decoded_audio_frames += 1
                    decoded_audio_samples += int(frame.samples)

    if channels != 2:
        raise RuntimeError(f"Expected stereo audio in {path}, got {channels} channels")
    if sample_rate != 32000:
        raise RuntimeError(f"Expected 32 kHz audio in {path}, got {sample_rate}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions in {path}: {width}x{height}")
    if decoded_video_frames <= 0 or decoded_audio_samples <= 0:
        raise RuntimeError(f"Media decode produced no content for {path}: "
                           f"video_frames={decoded_video_frames} audio_samples={decoded_audio_samples}")
    probe = {
        "probe_backend": "pyav",
        "format": {"duration": duration},
        "streams": [
            {
                "index": video_index,
                "codec_type": "video",
                "codec_name": video_codec_name,
                "width": width,
                "height": height,
                "decoded_frames": decoded_video_frames,
            },
            {
                "index": audio_index,
                "codec_type": "audio",
                "codec_name": audio_codec_name,
                "channels": channels,
                "sample_rate": sample_rate,
                "decoded_frames": decoded_audio_frames,
                "decoded_samples": decoded_audio_samples,
            },
        ],
    }
    return probe


def _denoise_seconds(result: object) -> float | None:
    stages = getattr(getattr(result, "logging_info", None), "stages", None)
    if not stages:
        return None
    for stage_name, metrics in stages.items():
        if "denois" in stage_name.lower() and metrics.get("execution_time") is not None:
            return float(metrics["execution_time"])
    return None


def _stage_timings(result: object) -> dict[str, dict[str, Any]]:
    """Return JSON-safe pipeline stage receipts from the profiled result."""
    stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
    receipt: dict[str, dict[str, Any]] = {}
    for stage_name, values in stages.items():
        if not isinstance(values, dict):
            continue
        stage: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                stage[str(key)] = value
        receipt[str(stage_name)] = stage
    return receipt


def _shifted_schedule(num_points: int, shift: float) -> dict[str, list[float]]:
    """Materialize the exact shared-base H3 sigma grid for a modality shift."""
    base = torch.linspace(1.0, 0.0, num_points, dtype=torch.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    return {
        "sigmas": [float(value) for value in shifted],
        "transformer_timesteps": [float(value) for value in (1.0 - shifted[:-1])],
    }


def _gpu_receipt() -> list[dict[str, Any]]:
    receipt = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        receipt.append({
            "index": index,
            "name": props.name,
            "total_memory": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return receipt


def _inference_args(args: argparse.Namespace, first_prompt: str) -> argparse.Namespace:
    parser = basic_fasth3.build_parser()
    inference_args = parser.parse_args([
        "--model-path", args.model_path,
        "--prompt", first_prompt,
        "--output", str(args.output_dir),
        "--profile", args.profile,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-frames", str(args.num_frames),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--num-gpus", str(args.num_gpus),
        "--repeats", "1",
        "--no-warmup",
        "--vsa-kernel", "sm100a",
        "--vsa-sparsity", "0.9",
        "--vsa-tile-size", "64",
        "--replicated-dit",
        "--parallel-vae",
        "--compile-vae" if args.compile_vae else "--no-compile-vae",
        "--pin-cpu-memory",
        "--no-torch-compile",
        "--inference-torch-compile" if args.compile else "--no-inference-torch-compile",
        "--fa4" if args.fa4 else "--no-fa4",
    ])
    inference_args.vsa = args.attention == "vsa"
    return basic_fasth3.validate_args(parser, inference_args)


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text())
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError("The baseline prompt set is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    revision = _checkpoint_revision(args.model_path)
    inference_args = _inference_args(args, prompts[0]["prompt"])
    environment = basic_fasth3.configure_environment(inference_args)
    basic_fasth3.validate_profile_dependencies(inference_args)
    gpu_receipt = _gpu_receipt()

    import wandb
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for sprint runs")
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="baseline",
        config={
            "source_commit": args.source_commit,
            "model_path": args.model_path,
            "checkpoint_revision": revision,
            "checkpoint_role": args.checkpoint_role,
            "attention": args.attention,
            "profile": args.profile,
            "compile_vae": args.compile_vae,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "steps_grid_points": args.steps,
            "dit_calls": args.steps - 1,
            "video_shift": 12,
            "audio_shift": 3,
            "seed": args.seed,
            "backend_environment": environment,
            "gpu_receipt": gpu_receipt,
            "persistent_output": str(args.output_dir),
        },
    )
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "wandb_run": run.url,
        "source_commit": args.source_commit,
        "model_path": args.model_path,
        "checkpoint_revision": revision,
        "checkpoint_role": args.checkpoint_role,
        "attention": args.attention,
        "quantization": "none",
        "compile_vae": args.compile_vae,
        "negative_prompt": "",
        "seed": args.seed,
        "resolution": {"width": args.width, "height": args.height},
        "num_frames": args.num_frames,
        "fps": 24,
        "audio_sample_rate": 32000,
        "guidance_scale": 1.0,
        "schedule": {
            "shared_base_sigmas": [float(value) for value in torch.linspace(1.0, 0.0, args.steps)],
            "grid_points": args.steps,
            "transformer_calls": args.steps - 1,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "video": _shifted_schedule(args.steps, 12.0),
            "audio": _shifted_schedule(args.steps, 3.0),
        },
        "backend_environment": environment,
        "gpu_receipt": gpu_receipt,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "started_at_unix": time.time(),
        "media_contract": "one video stream, stereo 32 kHz audio; semantic synchronization requires review",
        "results": [],
    }
    generator = basic_fasth3.VideoGenerator.from_config(basic_fasth3.build_generator_config(inference_args))
    wall_times: list[float] = []
    try:
        for index, prompt in enumerate(prompts):
            output_path = args.output_dir / f"{index:02d}_{prompt['id']}.mp4"
            started = time.perf_counter()
            result = generator.generate(basic_fasth3.build_request(inference_args, output_path, args.seed))
            wall_seconds = time.perf_counter() - started
            actual_path = basic_fasth3._actual_output_path(result, output_path)
            probe = _probe_media(actual_path)
            denoise_seconds = _denoise_seconds(result)
            stage_timings = _stage_timings(result)
            peak_memory_mb = getattr(result, "peak_memory_mb", None)
            wall_times.append(wall_seconds)
            record = {
                "index": index,
                "id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "expected_audio": prompt["expected_audio"],
                "seed": args.seed,
                "artifact_path": str(actual_path),
                "wall_seconds": wall_seconds,
                "denoise_seconds": denoise_seconds,
                "generation_seconds": getattr(result, "generation_time", None),
                "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb is not None else None,
                "stage_timings": stage_timings,
                "ffprobe": probe,
                "sync_review": "pending",
                "completed_at_unix": time.time(),
            }
            manifest["results"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            metrics: dict[str, Any] = {
                "baseline/prompt_index": index,
                "baseline/wall_seconds": wall_seconds,
                "baseline/media_contract_pass": 1,
            }
            if denoise_seconds is not None:
                metrics["baseline/denoise_seconds"] = denoise_seconds
            if peak_memory_mb is not None:
                metrics["baseline/peak_memory_mb"] = float(peak_memory_mb)
            wandb.log(metrics, step=index)
            if args.upload_videos:
                artifact = wandb.Artifact(f"{args.run_id}-{prompt['id']}", type="baseline-video", metadata=record)
                artifact.add_file(str(actual_path))
                run.log_artifact(artifact)
        manifest["completed_at_unix"] = time.time()
        manifest["median_wall_seconds"] = statistics.median(wall_times)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        summary_artifact = wandb.Artifact(f"{args.run_id}-manifest", type="run-manifest")
        summary_artifact.add_file(str(manifest_path))
        summary_artifact.add_file(str(args.prompts))
        run.log_artifact(summary_artifact)
        run.summary["persistent_manifest"] = str(manifest_path)
        run.summary["checkpoint_revision"] = revision
        run.summary["media_contract_pass_count"] = len(manifest["results"])
        run.summary["median_wall_seconds"] = manifest["median_wall_seconds"]
    finally:
        generator.shutdown()
        run.finish()


if __name__ == "__main__":
    main()
