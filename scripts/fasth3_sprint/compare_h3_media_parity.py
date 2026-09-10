#!/usr/bin/env python3
"""Gate two same-prompt/same-seed H3 renders on video SSIM and decoded audio."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import numpy as np


def only_video(root: Path) -> Path:
    videos = sorted(root.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected exactly one MP4 under {root}, found {len(videos)}")
    return videos[0]


def decoded_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"no video frames decoded from {path}")
    return np.stack(frames)


def decoded_audio(path: Path) -> np.ndarray:
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        for frame in container.decode(audio=0):
            chunks.append(frame.to_ndarray().astype(np.float32).reshape(-1))
    if not chunks:
        raise ValueError(f"no audio samples decoded from {path}")
    return np.concatenate(chunks)


def similarity(reference: np.ndarray, candidate: np.ndarray, scale: float) -> tuple[float, float, float]:
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError(f"decoded media shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    difference = cand - ref
    rmse = float(np.sqrt(np.mean(difference**2)))
    relative_rmse = rmse / max(float(np.sqrt(np.mean(ref**2))), scale * 1e-12)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / max(denominator, 1e-30))
    return rmse, relative_rmse, cosine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-video-rmse", type=float, default=2.0)
    parser.add_argument("--min-video-cosine", type=float, default=0.9999)
    parser.add_argument("--max-audio-relative-rmse", type=float, default=0.02)
    parser.add_argument("--min-audio-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    reference = only_video(args.reference)
    candidate = only_video(args.candidate)
    ref_video = decoded_video(reference)
    cand_video = decoded_video(candidate)
    video_rmse, video_relative_rmse, video_cosine = similarity(ref_video, cand_video, 255.0)
    ref_audio = decoded_audio(reference)
    cand_audio = decoded_audio(candidate)
    audio_rmse, audio_relative_rmse, audio_cosine = similarity(ref_audio, cand_audio, 1.0)
    passed = (all(math.isfinite(value) for value in (
        video_rmse, video_relative_rmse, video_cosine, audio_rmse, audio_relative_rmse, audio_cosine))
              and video_rmse <= args.max_video_rmse
              and video_cosine >= args.min_video_cosine
              and audio_relative_rmse <= args.max_audio_relative_rmse
              and audio_cosine >= args.min_audio_cosine)
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "video_rmse_8bit": video_rmse,
        "video_relative_rmse": video_relative_rmse,
        "video_cosine": video_cosine,
        "audio_rmse": audio_rmse,
        "audio_relative_rmse": audio_relative_rmse,
        "audio_cosine": audio_cosine,
        "thresholds": {
            "max_video_rmse_8bit": args.max_video_rmse,
            "min_video_cosine": args.min_video_cosine,
            "max_audio_relative_rmse": args.max_audio_relative_rmse,
            "min_audio_cosine": args.min_audio_cosine,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
