# SPDX-License-Identifier: Apache-2.0
"""Side-by-side comparison media from ``bench_decoder.py --save-videos`` output.

For every clip it writes, into ``--output-dir``:
  <clip>_grid.mp4       all panels, scaled to fit, labelled with LPIPS vs the original decoder
  <clip>_crop.mp4       the same panels as 1:1 crops of the most detailed region
  <clip>_crop_mid.png   lossless middle frame of the crop grid

Panels are given as ``key:Label`` pairs; ``source`` is the ground-truth video
when the benchmark had one. Missing panels are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

DEFAULT_PANELS = ("source:Source video,full_fp32:Original VAE (fp32),light_int8:Release today (light + INT8),"
                  "light_prod:Light bf16,full_nvfp4_unit:C: full 36L NVFP4,light_nvfp4_unit:A: light 26L NVFP4,"
                  "light_d8_nvfp4_unit:B: 8L NVFP4,taeh3:TAEH3")
LABEL_HEIGHT = 30
COLUMNS = 4


def load_panels(bench_dir: Path, clip: str, panels: list[tuple[str, str]]) -> list[tuple[str, str, torch.Tensor]]:
    loaded = []
    for key, label in panels:
        path = bench_dir / f"{clip}__{key}.pt"
        if path.exists():
            loaded.append((key, label, torch.load(path, map_location="cpu")[0]))  # 3, T, H, W uint8
    frames = min(video.shape[1] for _, _, video in loaded)
    return [(key, label, video[:, :frames]) for key, label, video in loaded]


def lpips_by_variant(results: list[dict], clip: str) -> dict[str, float]:
    return {
        row["variant"]: row["vs_full_fp32"]["lpips"]
        for row in results if row["clip"] == clip and "lpips" in row.get("vs_full_fp32", {})
    }


def detailed_crop(video: torch.Tensor, size: int) -> tuple[int, int]:
    """Top-left of the ``size`` crop with the most high-frequency detail in the middle frame."""
    frame = video[:, video.shape[1] // 2].float().mean(0)
    detail = (frame[1:, 1:] - frame[:-1, 1:]).abs() + (frame[1:, 1:] - frame[1:, :-1]).abs()
    pooled = F.avg_pool2d(detail[None, None], size, stride=size // 4)[0, 0]
    index = int(pooled.argmax())
    top, left = divmod(index, pooled.shape[1])
    return top * (size // 4), left * (size // 4)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label_strip(width: int, text: str) -> np.ndarray:
    strip = Image.new("RGB", (width, LABEL_HEIGHT), (16, 16, 16))
    ImageDraw.Draw(strip).text((6, 5), text, fill=(240, 240, 240), font=font(18))
    return np.asarray(strip)


def tile(panels: list[tuple[str, np.ndarray]], frame: int) -> np.ndarray:
    """``panels`` of (label, [T, H, W, 3]) -> one grid frame."""
    height, width = panels[0][1].shape[1:3]
    cells = [np.concatenate([label_strip(width, label), video[frame]], axis=0) for label, video in panels]
    blank = np.zeros_like(cells[0])
    while len(cells) % COLUMNS:
        cells.append(blank)
    rows = [np.concatenate(cells[i:i + COLUMNS], axis=1) for i in range(0, len(cells), COLUMNS)]
    grid = np.concatenate(rows, axis=0)
    return grid[:grid.shape[0] // 2 * 2, :grid.shape[1] // 2 * 2]


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12", "preset": "medium"}
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def to_thw3(video: torch.Tensor) -> np.ndarray:
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def compose_clip(bench_dir: Path, out_dir: Path, clip: str, panels: list[tuple[str, str]], results: list[dict],
                 scale: float, crop: int, fps: int) -> None:
    loaded = load_panels(bench_dir, clip, panels)
    scores = lpips_by_variant(results, clip)

    def label(key: str, text: str) -> str:
        return f"{text}  LPIPS {scores[key]:.4f}" if key in scores else text

    full = []
    for key, text, video in loaded:
        scaled = F.interpolate(video.float(), scale_factor=scale, mode="bilinear", antialias=True)
        full.append((label(key, text), scaled.clamp(0, 255).round().to(torch.uint8)))
    frames = full[0][1].shape[1]
    write_mp4(out_dir / f"{clip}_grid.mp4", [tile([(t, to_thw3(v)) for t, v in full], i) for i in range(frames)], fps)

    reference = next(video for key, _, video in loaded if key == "full_fp32")
    top, left = detailed_crop(reference, crop)
    crops = [(label(key, text), to_thw3(video[:, :, top:top + crop, left:left + crop])) for key, text, video in loaded]
    crop_frames = [tile(crops, i) for i in range(frames)]
    write_mp4(out_dir / f"{clip}_crop.mp4", crop_frames, fps)
    Image.fromarray(crop_frames[frames // 2]).save(out_dir / f"{clip}_crop_mid.png")
    print(f"{clip}: {len(loaded)} panels, {frames} frames, crop at ({top}, {left})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", required=True, help="bench_decoder.py --output-dir with --save-videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panels", default=DEFAULT_PANELS)
    parser.add_argument("--scale", type=float, default=0.5, help="panel scale for the full-frame grid")
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    bench_dir, out_dir = Path(args.bench_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [tuple(item.split(":", 1)) for item in args.panels.split(",")]
    results = json.loads((bench_dir / "results.json").read_text())
    clips = sorted({row["clip"] for row in results})
    for clip in clips:
        compose_clip(bench_dir, out_dir, clip, panels, results, args.scale, args.crop, args.fps)


if __name__ == "__main__":
    main()
