#!/usr/bin/env python3
"""Machine-readable bridge between FastVideo Mac and the MLX Python lane.

The Swift application owns UI and durable history. This process owns model
installation, runtime diagnosis, and one generation child process. Every line
written to stdout is a JSON object so app progress never depends on scraping a
terminal transcript directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_child: subprocess.Popen[str] | None = None


def emit(event_type: str, **payload: Any) -> None:
    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


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


def rife_weights_present(model_root: Path) -> bool:
    rife_root = model_root / "rife" / "RIFE-4.25"
    return (rife_root / "config.json").is_file() and (rife_root / "model.safetensors").is_file()


def command_diagnose(args: argparse.Namespace) -> int:
    model_root = Path(args.model_root).expanduser()
    raw = resolve_checkpoint(model_root, "raw", args.raw_checkpoint)
    ema = resolve_checkpoint(model_root, "ema", args.ema_checkpoint)

    mlx_available = importlib.util.find_spec("mlx") is not None
    rife_available = importlib.util.find_spec("rife_mlx") is not None and rife_weights_present(model_root)
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
        "rife_available": rife_available,
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


def _download_release_asset(asset: dict[str, Any], archive_path: Path, index: int, count: int) -> None:
    url = str(asset.get("url") or "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "file"):
        raise ValueError("The bundled model catalog contains an invalid download URL.")

    request = urllib.request.Request(url, headers={"User-Agent": "FastWan-QAD-Mac/1"})
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as destination:
        expected_size = int(response.headers.get("Content-Length") or asset.get("bytes") or 0)
        completed = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            destination.write(chunk)
            completed += len(chunk)
            item_fraction = completed / expected_size if expected_size else 0
            emit(
                "progress",
                phase="Downloading FastWan QAD",
                bytes_completed=completed,
                bytes_total=expected_size or None,
                fraction=min((index + item_fraction) / count, 0.98),
            )

    expected_sha256 = str(asset.get("sha256") or "").lower()
    if expected_sha256:
        digest = hashlib.sha256()
        with archive_path.open("rb") as downloaded:
            for chunk in iter(lambda: downloaded.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("The downloaded model did not match the release checksum.")


def _extract_release_asset(archive_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-install-", dir=destination.parent))
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            root = temporary.resolve()
            for member in archive.getmembers():
                member_path = (temporary / member.name).resolve()
                if root not in member_path.parents and member_path != root:
                    raise ValueError("The model archive contains an unsafe path.")
            archive.extractall(temporary, filter="data")
        entries = list(temporary.iterdir())
        payload = entries[0] if len(entries) == 1 and entries[0].is_dir() else temporary
        staged = destination.parent / f".{destination.name}-ready"
        if staged.exists():
            shutil.rmtree(staged)
        if payload == temporary:
            os.replace(temporary, staged)
        else:
            os.replace(payload, staged)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def command_install_release(args: argparse.Namespace) -> int:
    catalog = json.loads(Path(args.catalog).expanduser().read_text())
    variant = str(args.variant).lower()
    if variant not in ("ema", "raw"):
        emit("error", message="Unknown FastWan QAD model variant.")
        return 2

    model_root = Path(args.model_root).expanduser()
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    assets: list[tuple[str, dict[str, Any], Path]] = []
    if not model_components_present(model_root):
        assets.append(("Core model", catalog["shared"], model_root))
    if not rife_weights_present(model_root):
        assets.append(("Fast generation", catalog["fast_mode"], model_root / "rife" / "RIFE-4.25"))
    assets.append((f"{variant.upper()} weights", catalog["variants"][variant], checkpoint_root))

    work_root = Path(tempfile.mkdtemp(prefix="fastwan-qad-download-"))
    emit("phase", phase="Preparing download", message=f"Installing FastWan QAD {variant.upper()}")
    try:
        for index, (label, asset, destination) in enumerate(assets):
            emit("phase", phase=f"Downloading {label}", message=f"Downloading {label}")
            archive_path = work_root / f"asset-{index}.tar.gz"
            _download_release_asset(asset, archive_path, index, len(assets))
            emit("phase", phase=f"Installing {label}", message=f"Installing {label}")
            _extract_release_asset(archive_path, destination)
    except Exception as exc:
        emit("error", message=f"Model installation failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    emit(
        "complete",
        phase="Model ready",
        output_path=str(checkpoint_root),
        fraction=1.0,
    )
    return 0


def command_install_fast_mode(args: argparse.Namespace) -> int:
    catalog = json.loads(Path(args.catalog).expanduser().read_text())
    model_root = Path(args.model_root).expanduser()
    destination = model_root / "rife" / "RIFE-4.25"
    if rife_weights_present(model_root):
        emit("complete", phase="Fast generation ready", output_path=str(destination), fraction=1.0)
        return 0

    work_root = Path(tempfile.mkdtemp(prefix="fastwan-qad-fast-mode-"))
    emit("phase", phase="Preparing fast generation", message="Installing MLX-native RIFE")
    try:
        archive_path = work_root / "rife-4.25.tar.gz"
        _download_release_asset(catalog["fast_mode"], archive_path, 0, 1)
        emit("phase", phase="Installing fast generation", message="Installing RIFE 4.25")
        _extract_release_asset(archive_path, destination)
    except Exception as exc:
        emit("error", message=f"Fast generation installation failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    if not rife_weights_present(model_root):
        emit("error", message="The Fast generation archive is incomplete.")
        return 1
    emit("complete", phase="Fast generation ready", output_path=str(destination), fraction=1.0)
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
    if request.get("fast"):
        command.append("--fast")
        rife_weights = model_root / "rife" / "RIFE-4.25"
        if rife_weights.is_dir():
            command.extend(("--fast-rife-weights-dir", str(rife_weights)))
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

    fast_mode = bool(request.get("fast"))
    emit(
        "phase",
        phase="Preparing fast generation" if fast_mode else "Preparing",
        message="Starting the Apple-native runtime",
    )
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
        elif fast_mode and line.startswith("[fast] generating"):
            emit("phase", phase="Fast mode · fewer source frames", fraction=0.08, message=line)
        elif fast_mode and "[fast] RIFE" in line:
            emit("progress", phase="Finishing motion", fraction=0.97, message=line)
        elif "Downloading" in line:
            emit("phase", phase="Preparing decoder", message=line)
        elif "Output written to:" in line:
            emit(
                "phase",
                phase="Interpolating motion" if fast_mode else "Saving video",
                fraction=0.9 if fast_mode else None,
                message=line,
            )
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

    install = subparsers.add_parser("install-release")
    install.add_argument("--catalog", required=True)
    install.add_argument("--variant", choices=("ema", "raw"), required=True)
    install.add_argument("--model-root", required=True)
    install.add_argument("--checkpoint-root", required=True)
    install.set_defaults(handler=command_install_release)

    install_fast = subparsers.add_parser("install-fast-mode")
    install_fast.add_argument("--catalog", required=True)
    install_fast.add_argument("--model-root", required=True)
    install_fast.set_defaults(handler=command_install_fast_mode)

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
