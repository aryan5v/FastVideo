# SPDX-License-Identifier: Apache-2.0
"""Probe all held-out reference MP4s and cross-check their frozen metadata."""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/validation/heldout64.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def probe(row: dict[str, Any]) -> dict[str, Any]:
    import av

    with av.open(row["ref_video"]) as container:
        videos = list(container.streams.video)
        audios = list(container.streams.audio)
        if len(videos) != 1 or len(audios) != 1:
            raise ValueError(f"{row['sample_id']}: expected one video and one audio stream")
        video, audio = videos[0], audios[0]
        actual = {
            "width": int(video.width),
            "height": int(video.height),
            "num_frames": int(video.frames),
            "fps": float(video.average_rate),
            "audio_sample_rate": int(audio.rate),
            "audio_channels": int(audio.channels),
        }
    if actual["num_frames"] <= 0:
        raise ValueError(f"{row['sample_id']}: container has no indexed video frame count")
    expected = {
        "width": int(row["width"]),
        "height": int(row["height"]),
        "num_frames": int(row["num_frames"]),
        "fps": float(row["fps"]),
        "audio_sample_rate": int(row["audio_sample_rate"]),
        "audio_channels": int(row["audio_channels"]),
    }
    if actual != expected:
        raise ValueError(f"{row['sample_id']}: ffprobe {actual} != manifest {expected}")
    return actual


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("heldout manifest must be an object with a data list")
    rows = payload["data"]
    ids = [str(row["sample_id"]) for row in rows]
    paths = [str(Path(row["ref_video"]).resolve()) for row in rows]
    if len(rows) != 64 or len(set(ids)) != 64 or len(set(paths)) != 64:
        raise ValueError("heldout manifest must contain 64 unique ids and reference paths")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(probe, rows))
    sources = collections.Counter(row["source"] for row in rows)
    shapes = collections.Counter(f"{row['width']}x{row['height']}-{row['num_frames']}f" for row in rows)
    print(f"validated 64 reference MP4 stream headers with PyAV; sources={dict(sorted(sources.items()))}")
    print(f"shape buckets ({len(shapes)}): {dict(sorted(shapes.items()))}")


if __name__ == "__main__":
    main()
