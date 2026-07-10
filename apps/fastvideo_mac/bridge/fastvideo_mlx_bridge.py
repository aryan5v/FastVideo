#!/usr/bin/env python3
"""Machine-readable bridge between FastVideo Mac and the MLX Python lane.

The Swift application owns UI and durable history. This process owns model
installation, runtime diagnosis, and one generation child process. Every line
written to stdout is a JSON object so app progress never depends on scraping a
terminal transcript directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_child: subprocess.Popen[str] | None = None


def emit(event_type: str, **payload: Any) -> None:
    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def resolve_checkpoint(model_root: Path, variant: str, explicit: str | None = None) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.exists() else None

    names = (
        f"mlx_dit_{variant}",
        f"mlx_dit-{variant}",
        f"mlx_dit/{variant}",
        f"{variant}/mlx_dit",
    )
    for name in names:
        candidate = model_root / name
        if candidate.exists():
            return candidate
    return None


def checkpoint_is_valid(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    return (path / "mlx_dit.json").is_file() and (path / "mlx_dit.safetensors").is_file()


def resolve_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    if not executable:
        return None
    return executable if Path(executable).is_file() and os.access(executable, os.X_OK) else None


def model_components_present(model_root: Path) -> bool:
    required = (
        model_root / "tokenizer",
        model_root / "text_encoder",
        model_root / "transformer" / "config.json",
    )
    return all(path.exists() for path in required)


def command_diagnose(args: argparse.Namespace) -> int:
    model_root = Path(args.model_root).expanduser()
    raw = resolve_checkpoint(model_root, "raw", args.raw_checkpoint)
    ema = resolve_checkpoint(model_root, "ema", args.ema_checkpoint)

    mlx_available = importlib.util.find_spec("mlx") is not None
    torch_available = importlib.util.find_spec("torch") is not None
    mps_available = False
    if torch_available:
        try:
            import torch

            mps_available = bool(torch.backends.mps.is_available())
        except Exception:
            mps_available = False

    payload = {
        "platform_supported": platform.system() == "Darwin" and platform.machine() == "arm64",
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": sys.version.split()[0],
        "mlx_available": mlx_available,
        "torch_available": torch_available,
        "mps_available": mps_available,
        "ffmpeg_available": resolve_ffmpeg() is not None,
        "model_root": str(model_root),
        "model_components_present": model_components_present(model_root),
        "raw_checkpoint": str(raw) if raw else None,
        "raw_available": checkpoint_is_valid(raw),
        "ema_checkpoint": str(ema) if ema else None,
        "ema_available": checkpoint_is_valid(ema),
    }
    payload["ready"] = all((
        payload["platform_supported"],
        mlx_available,
        torch_available,
        mps_available,
        payload["ffmpeg_available"],
        payload["model_components_present"],
        payload["raw_available"] or payload["ema_available"],
    ))
    emit("diagnosis", **payload)
    return 0


def _model_total_size(repo_id: str, revision: str | None) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        total = 0
        for sibling in info.siblings or []:
            size = sibling.size
            if size is None and sibling.lfs is not None:
                size = sibling.lfs.get("size")
            total += int(size or 0)
        return total or None
    except Exception as exc:
        emit("log", level="warning", message=f"Could not query model size: {exc}")
        return None


def command_install_model(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        emit("error", message="huggingface_hub is not installed in the selected runtime.")
        return 2

    destination = Path(args.model_root).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = _model_total_size(args.repo_id, args.revision)
    stop = threading.Event()

    def report_progress() -> None:
        while not stop.wait(0.75):
            downloaded = directory_size(destination)
            fraction = downloaded / total_bytes if total_bytes else None
            emit(
                "progress",
                phase="Downloading model",
                bytes_completed=downloaded,
                bytes_total=total_bytes,
                fraction=min(fraction, 0.99) if fraction is not None else None,
            )

    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()
    emit("phase", phase="Downloading model", message=f"Fetching {args.repo_id}")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(destination),
        )
    except Exception as exc:
        emit("error", message=f"Model download failed: {exc}")
        return 1
    finally:
        stop.set()
        reporter.join(timeout=1)

    downloaded = directory_size(destination)
    emit(
        "complete",
        phase="Model ready",
        output_path=str(destination),
        bytes_completed=downloaded,
        bytes_total=total_bytes,
        fraction=1.0,
    )
    return 0


def _handle_signal(signum: int, _frame: Any) -> None:
    global _child
    if _child is not None and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _child.kill()
    raise SystemExit(128 + signum)


def _generation_command(request: dict[str, Any], request_path: Path) -> list[str]:
    repo_root = Path(request.get("repo_root") or Path(__file__).resolve().parents[3])
    script = repo_root / "examples" / "inference" / "basic" / "mlx_wan_prompt_to_video.py"
    if not script.is_file():
        raise FileNotFoundError(f"FastVideo MLX entrypoint not found: {script}")

    model_root = Path(request["model_root"]).expanduser()
    checkpoint = Path(request["checkpoint_path"]).expanduser()
    output_path = Path(request["output_path"]).expanduser()
    metrics_path = output_path.with_suffix(".metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(script),
        "--model-root",
        str(model_root),
        "--mlx-checkpoint",
        str(checkpoint),
        "--mlx-quantization",
        "int8",
        "--height",
        str(int(request.get("height", 480))),
        "--width",
        str(int(request.get("width", 832))),
        "--num-frames",
        str(int(request.get("num_frames", 81))),
        "--num-inference-steps",
        "3",
        "--dmd-denoising-steps",
        str(request.get("dmd_denoising_steps", "1000,757,522")),
        "--decode-backend",
        "taehv",
        "--fps",
        str(int(request.get("fps", 16))),
        "--seed",
        str(int(request.get("seed", 1024))),
        "--prompt",
        str(request["prompt"]),
        "--output-path",
        str(output_path),
        "--metrics-json",
        str(metrics_path),
        "--preview-dir",
        str(output_path.parent / "previews"),
        "--preview-every",
        "1",
    ]
    memory_limit = request.get("mlx_memory_limit_gib")
    if memory_limit is not None:
        command.extend(("--mlx-memory-limit-gib", str(float(memory_limit))))
    if request.get("taehv_parallel"):
        command.append("--taehv-parallel")
    return command


_DENOISE_RE = re.compile(r"denoise step\s+(\d+)/(\d+)\s+complete", re.IGNORECASE)
_PREVIEW_RE = re.compile(r"Preview written to:\s+(.+?)\s+\(step\s+(\d+)/(\d+)\)", re.IGNORECASE)


def command_generate(args: argparse.Namespace) -> int:
    global _child
    request_path = Path(args.request).expanduser()
    request = json.loads(request_path.read_text())
    command = _generation_command(request, request_path)
    output_path = Path(request["output_path"]).expanduser()
    metrics_path = output_path.with_suffix(".metrics.json")

    emit("phase", phase="Preparing", message="Starting the Apple-native runtime")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        env.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg)
        env["PATH"] = f"{Path(ffmpeg).parent}{os.pathsep}{env.get('PATH', '')}"

    try:
        _child = subprocess.Popen(
            command,
            cwd=str(Path(request.get("repo_root") or Path(__file__).resolve().parents[3])),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        emit("error", message=f"Could not start generation: {exc}")
        return 1

    assert _child.stdout is not None
    for raw_line in iter(_child.stdout.readline, ""):
        line = raw_line.strip()
        if not line:
            continue
        match = _DENOISE_RE.search(line)
        preview_match = _PREVIEW_RE.search(line)
        if preview_match:
            preview_path = preview_match.group(1)
            current, total = int(preview_match.group(2)), int(preview_match.group(3))
            emit(
                "preview",
                phase=f"Preview {current} ready",
                current=current,
                total=total,
                fraction=0.12 + 0.76 * current / total,
                preview_path=preview_path,
                message="A rough x0 preview is ready while MLX keeps refining.",
            )
        elif match:
            current, total = int(match.group(1)), int(match.group(2))
            emit(
                "progress",
                phase="Denoising on MLX",
                current=current,
                total=total,
                fraction=0.12 + 0.76 * current / total,
                message=line,
            )
        elif "Downloading" in line:
            emit("phase", phase="Preparing decoder", message=line)
        elif "Output written to:" in line:
            emit("phase", phase="Saving video", message=line)
        else:
            emit("log", level="info", message=line)

    return_code = _child.wait()
    _child = None
    if return_code != 0:
        emit("error", message=f"Generation exited with code {return_code}.")
        return return_code
    if not output_path.is_file():
        emit("error", message="Generation finished but no MP4 was written.")
        return 1

    preview_dir = output_path.parent / "previews"
    if preview_dir.is_dir():
        shutil.rmtree(preview_dir, ignore_errors=True)

    metrics: dict[str, Any] | None = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            metrics = None
    emit(
        "complete",
        phase="Ready",
        fraction=1.0,
        output_path=str(output_path),
        metrics=metrics,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FastVideo Mac MLX bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--model-root", required=True)
    diagnose.add_argument("--raw-checkpoint")
    diagnose.add_argument("--ema-checkpoint")
    diagnose.set_defaults(handler=command_diagnose)

    install = subparsers.add_parser("install-model")
    install.add_argument("--repo-id", required=True)
    install.add_argument("--revision")
    install.add_argument("--model-root", required=True)
    install.set_defaults(handler=command_install_model)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--request", required=True)
    generate.set_defaults(handler=command_generate)
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
