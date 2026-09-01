# SPDX-License-Identifier: Apache-2.0
"""Exercise the H3 OpenAI-compatible server with cURL and preserve receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="fasth3-v1")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-decode-backend", choices=("h3-vae", "taeh3"), default="h3-vae")
    parser.add_argument("--expected-cases", type=int, default=10)
    parser.add_argument("--generation-timeout", type=float, default=3600)
    parser.add_argument("--wandb-project", default="fasth3-serve-cookbook-eval")
    return parser.parse_args()


def curl(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    output: Path | None = None,
    timeout: float = 60,
) -> tuple[bytes, float, list[str]]:
    command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--max-time",
        str(max(1, int(timeout))),
        "-X",
        method,
        url,
    ]
    body = None
    if payload is not None:
        command.extend(["-H", "Content-Type: application/json", "--data-binary", "@-"])
        body = json.dumps(payload, ensure_ascii=False).encode()
    if output is not None:
        command.extend(["--output", str(output)])
    started = time.perf_counter()
    result = subprocess.run(command, input=body, capture_output=True, check=False, timeout=timeout + 10)
    elapsed = time.perf_counter() - started
    if result.returncode:
        detail = (result.stderr + result.stdout).decode(errors="replace")
        raise RuntimeError(f"cURL failed ({result.returncode}) for {method} {url}: {detail}")
    return result.stdout, elapsed, command


def curl_json(url: str, **kwargs: Any) -> tuple[dict[str, Any], float, list[str]]:
    body, elapsed, command = curl(url, **kwargs)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Expected JSON from {url}, got {body[:500]!r}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}, got {type(parsed).__name__}")
    return parsed, elapsed, command


def probe_media(path: Path, expected_width: int, expected_height: int) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise RuntimeError(f"Expected one video and one audio stream, got {len(video_streams)} and {len(audio_streams)}")
        video = video_streams[0].codec_context
        audio = audio_streams[0].codec_context
        width, height = int(video.width or 0), int(video.height or 0)
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(f"Expected {expected_width}x{expected_height}, got {width}x{height}")
        channels = len(audio.layout.channels) if audio.layout is not None else 0
        sample_rate = int(audio.sample_rate or 0)
        decoded_video_frames = 0
        decoded_audio_samples = 0
        for packet in container.demux(video_streams[0], audio_streams[0]):
            for frame in packet.decode():
                if packet.stream.type == "video":
                    decoded_video_frames += 1
                else:
                    decoded_audio_samples += int(frame.samples)
        duration = float(container.duration / av.time_base) if container.duration is not None else None
    if decoded_video_frames <= 0 or decoded_audio_samples <= 0:
        raise RuntimeError("Decoded media is missing video frames or audio samples")
    return {
        "duration_s": duration,
        "width": width,
        "height": height,
        "video_codec": video.name,
        "decoded_video_frames": decoded_video_frames,
        "audio_codec": audio.name,
        "audio_channels": channels,
        "audio_sample_rate": sample_rate,
        "decoded_audio_samples": decoded_audio_samples,
        "stream_contract_pass": channels == 2 and sample_rate == 32000,
    }


def wait_for_server(base_url: str, timeout: float = 3600) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            health, _, _ = curl_json(f"{base_url}/health", timeout=10)
            if health.get("status") == "ok":
                return health, time.perf_counter() - started
        except Exception as error:  # server startup is intentionally polled
            last_error = error
        time.sleep(5)
    raise TimeoutError(f"Server did not become healthy in {timeout:g}s: {last_error}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipts = args.output_root / "receipts"
    receipts.mkdir(exist_ok=True)
    cases = json.loads(args.prompts.read_text())
    if len(cases) != args.expected_cases:
        raise ValueError(f"Expected exactly {args.expected_cases} prompt cases, got {len(cases)}")

    started_at = time.time()
    health, readiness_wait_s = wait_for_server(args.base_url)
    models, models_latency_s, _ = curl_json(f"{args.base_url}/v1/models")
    advertised = [item.get("id") for item in models.get("data", [])]
    if args.model not in advertised:
        raise RuntimeError(f"Model alias {args.model!r} not in /v1/models: {advertised}")
    playground_html, playground_latency_s, _ = curl(f"{args.base_url}/playground/")
    playground_config, playground_config_latency_s, _ = curl_json(f"{args.base_url}/playground/config")
    playground_js, _, _ = curl(f"{args.base_url}/playground/playground.js")
    playground_css, _, _ = curl(f"{args.base_url}/playground/playground.css")
    if b"Generate video" not in playground_html or not playground_js or not playground_css:
        raise RuntimeError("Prompt Playground assets did not satisfy the expected content contract")

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="serving-eval",
        config={
            "source_commit": args.source_commit,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "served_model": args.model,
            "gpu_count": 4,
            "num_cases": len(cases),
            "num_frames": 124,
            "fps": 24,
            "grid_points": 5,
            "transformer_forwards": 4,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "video_decode_backend": args.video_decode_backend,
            "vsa_sparsity": 0.9,
            "persistent_output": str(args.output_root),
        },
    )
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "wandb_url": run.url,
        "source_commit": args.source_commit,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "served_model": args.model,
        "video_decode_backend": args.video_decode_backend,
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "slurm_nodes": __import__("os").environ.get("SLURM_JOB_NODELIST"),
        "started_at_unix": started_at,
        "server_readiness_wait_s": readiness_wait_s,
        "cookbook_receipt": {
            "health": health,
            "models_latency_s": models_latency_s,
            "playground_latency_s": playground_latency_s,
            "playground_config_latency_s": playground_config_latency_s,
            "playground_config": playground_config,
            "playground_html_bytes": len(playground_html),
            "playground_js_bytes": len(playground_js),
            "playground_css_bytes": len(playground_css),
            "curl_client": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
        },
        "results": [],
    }
    summary_path = args.output_root / "summary.json"
    requests_path = receipts / "requests.jsonl"
    commands_path = receipts / "curl_commands.sh"
    requests_path.write_text("")
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n")

    try:
        for index, case in enumerate(cases):
            payload = {
                "model": args.model,
                "prompt": case["prompt"],
                "size": f"{case['width']}x{case['height']}",
                "num_frames": 124,
                "fps": 24,
                "num_inference_steps": 5,
                "guidance_scale": 1.0,
                "seed": case["seed"],
            }
            with requests_path.open("a") as handle:
                handle.write(json.dumps({"case": case, "payload": payload}, ensure_ascii=False) + "\n")
            job_started = time.perf_counter()
            job, submit_latency_s, command = curl_json(
                f"{args.base_url}/v1/videos", method="POST", payload=payload, timeout=60
            )
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"Submission for {case['id']} returned no job id: {job}")
            reproducible = command[:-2] + ["--data-binary", json.dumps(payload, ensure_ascii=False)]
            with commands_path.open("a") as handle:
                handle.write(f"# {case['id']}\n{shlex.join(reproducible)}\n\n")
            deadline = time.perf_counter() + args.generation_timeout
            while job.get("status") not in {"completed", "failed"}:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f"Generation {job_id} timed out; server job may still be running")
                time.sleep(5)
                job, _, _ = curl_json(f"{args.base_url}/v1/videos/{job_id}", timeout=60)
            if job.get("status") != "completed":
                raise RuntimeError(f"Generation {job_id} failed: {job.get('error')}")
            output_dir = args.output_root / "videos" / case["resolution"]
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"{index + 1:02d}_{case['id']}.mp4"
            _, download_latency_s, _ = curl(
                f"{args.base_url}/v1/videos/{job_id}/content",
                output=output,
                timeout=300,
            )
            media = probe_media(output, int(case["width"]), int(case["height"]))
            wall_s = time.perf_counter() - job_started
            record = {
                "index": index,
                "case": case,
                "is_warmup": case.get("role") == "warmup",
                "job_id": job_id,
                "status": job["status"],
                "submit_latency_s": submit_latency_s,
                "download_latency_s": download_latency_s,
                "request_to_download_wall_s": wall_s,
                "server_inference_time_s": job.get("inference_time_s"),
                "server_stage_durations": job.get("stage_durations"),
                "server_peak_memory_mb": job.get("peak_memory_mb"),
                "artifact": str(output),
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "media": media,
                "completed_at_unix": time.time(),
            }
            summary["results"].append(record)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            with (receipts / "results.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            metrics = {
                "clip/index": index,
                "clip/width": case["width"],
                "clip/height": case["height"],
                "clip/request_to_download_wall_s": wall_s,
                "clip/submit_latency_s": submit_latency_s,
                "clip/download_latency_s": download_latency_s,
                "clip/bytes": output.stat().st_size,
                "clip/media_contract_pass": int(media["stream_contract_pass"]),
                "clip/is_warmup": int(case.get("role") == "warmup"),
            }
            if job.get("inference_time_s") is not None:
                metrics["clip/server_inference_time_s"] = float(job["inference_time_s"])
            if job.get("peak_memory_mb") is not None:
                metrics["clip/server_peak_memory_mb"] = float(job["peak_memory_mb"])
            for stage, duration in (job.get("stage_durations") or {}).items():
                metrics[f"clip/stage/{stage}_s"] = float(duration)
            wandb.log(metrics, step=index)
            print(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"completed={index + 1}/{len(cases)} case={case['id']} job={job_id} "
                f"wall_s={wall_s:.2f} artifact={output}",
                flush=True,
            )
    finally:
        summary["completed_at_unix"] = time.time()
        summary["total_wall_s"] = summary["completed_at_unix"] - started_at
        summary["completed_cases"] = len(summary["results"])
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        wandb.finish()


if __name__ == "__main__":
    main()
