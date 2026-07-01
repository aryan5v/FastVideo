# SPDX-License-Identifier: Apache-2.0
"""Optional TAEHV decode helpers for Apple Silicon FastWan experiments."""

from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path

import numpy as np


TAEHV_SOURCE_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py"
TAEW2_1_CHECKPOINT_URL = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "fastvideo" / "taehv"


def _download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)  # noqa: S310 - explicit public model artifact URL.
    return path


def ensure_taew2_1_artifacts(
    *,
    source_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[Path, Path]:
    cache_dir = _default_cache_dir()
    source_path = source_path or cache_dir / "taehv.py"
    checkpoint_path = checkpoint_path or cache_dir / "taew2_1.pth"
    return (
        _download_if_missing(TAEHV_SOURCE_URL, source_path),
        _download_if_missing(TAEW2_1_CHECKPOINT_URL, checkpoint_path),
    )


def _load_taehv_class(source_path: Path):
    spec = importlib.util.spec_from_file_location("fastvideo_external_taehv", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TAEHV source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TAEHV


def decode_latents_to_video_taehv(
    *,
    latents_np: np.ndarray,
    output_path: Path,
    fps: int,
    device,
    dtype,
    parallel: bool,
    source_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    """Decode Wan/FastWan diffusion latents with TAEW2.1 and export MP4.

    TAEHV's Wan wrapper expects the diffusion latents directly, without applying
    the standard Wan VAE's `latents_mean` / `latents_std` shift.
    """
    import torch
    from diffusers.utils import export_to_video

    source_path, checkpoint_path = ensure_taew2_1_artifacts(
        source_path=source_path,
        checkpoint_path=checkpoint_path,
    )
    TAEHV = _load_taehv_class(source_path)
    taehv = TAEHV(str(checkpoint_path)).to(device=device, dtype=dtype)
    taehv.eval()

    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    with torch.no_grad():
        video_ntchw = taehv.decode_video(
            latents.transpose(1, 2),
            parallel=parallel,
            show_progress_bar=False,
        )
    video = video_ntchw.transpose(1, 2)
    video_np = video[0].permute(1, 2, 3, 0).float().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_np, str(output_path), fps=fps)
