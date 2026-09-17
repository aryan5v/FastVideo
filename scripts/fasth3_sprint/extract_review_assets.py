#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract deterministic visual and audio review assets from a FastH3 MP4."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def _contact_sheet(media: Path, output: Path, count: int) -> dict:
    with av.open(str(media)) as container:
        frames = [(float(frame.time or 0.0), frame.to_image().convert("RGB")) for frame in container.decode(video=0)]
    if not frames:
        raise RuntimeError(f"No video frames decoded from {media}")
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    tile_width, tile_height, label_height = 416, 240, 24
    columns = 4
    rows = math.ceil(len(indices) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "#111111")
    draw = ImageDraw.Draw(sheet)
    timestamps = []
    for slot, index in enumerate(indices):
        timestamp, image = frames[int(index)]
        x = (slot % columns) * tile_width
        y = (slot // columns) * (tile_height + label_height)
        sheet.paste(ImageOps.fit(image, (tile_width, tile_height)), (x, y))
        draw.text((x + 8, y + tile_height + 4), f"frame {int(index):03d}  {timestamp:0.2f}s", fill="white")
        timestamps.append(timestamp)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"decoded_video_frames": len(frames), "contact_sheet_timestamps": timestamps}


def _decode_audio(media: Path) -> tuple[np.ndarray, int]:
    chunks = []
    sample_rate = 32000
    with av.open(str(media)) as container:
        stream = container.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate or sample_rate)
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=sample_rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError(f"No audio frames decoded from {media}")
    interleaved = np.concatenate(chunks).astype(np.int16, copy=False)
    if interleaved.size % 2:
        raise RuntimeError(f"Stereo PCM sample count is not divisible by two: {interleaved.size}")
    return interleaved.reshape(-1, 2), sample_rate


def _write_waveform(samples: np.ndarray, sample_rate: int, output: Path) -> dict:
    normalized = samples.astype(np.float32) / 32768.0
    width, height = 1600, 360
    image = Image.new("RGB", (width, height), "#101217")
    draw = ImageDraw.Draw(image)
    colors = ("#55d6be", "#ffb86c")
    frames_per_pixel = max(1, math.ceil(len(normalized) / width))
    for channel, color in enumerate(colors):
        center = int((channel + 0.5) * height / 2)
        scale = height / 4 - 12
        draw.line((0, center, width, center), fill="#434957")
        for x in range(width):
            segment = normalized[x * frames_per_pixel:(x + 1) * frames_per_pixel, channel]
            if not len(segment):
                break
            low = center - int(float(segment.max()) * scale)
            high = center - int(float(segment.min()) * scale)
            draw.line((x, low, x, high), fill=color)
    duration = len(normalized) / sample_rate
    draw.text((10, 8), f"stereo  {sample_rate} Hz  {duration:0.3f}s", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    rms = np.sqrt(np.mean(np.square(normalized), axis=0))
    peak = np.max(np.abs(normalized), axis=0)
    mono = normalized.mean(axis=1)
    window = max(1, int(sample_rate * 0.02))
    usable = len(mono) // window * window
    window_rms = np.sqrt(np.mean(np.square(mono[:usable].reshape(-1, window)), axis=1))
    return {
        "sample_rate": sample_rate,
        "channels": 2,
        "samples_per_channel": len(normalized),
        "duration_seconds": duration,
        "rms": [float(value) for value in rms],
        "peak": [float(value) for value in peak],
        "active_20ms_fraction_above_minus_40db": float(np.mean(window_rms > 0.01)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.media.stem
    contact_path = args.output_dir / f"{stem}_contact_sheet.jpg"
    waveform_path = args.output_dir / f"{stem}_waveform.png"
    wav_path = args.output_dir / f"{stem}.wav"
    video = _contact_sheet(args.media, contact_path, args.frames)
    samples, sample_rate = _decode_audio(args.media)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    audio = _write_waveform(samples, sample_rate, waveform_path)
    manifest = {
        "media": str(args.media.resolve()),
        "contact_sheet": str(contact_path.resolve()),
        "waveform": str(waveform_path.resolve()),
        "wav": str(wav_path.resolve()),
        **video,
        **audio,
    }
    manifest_path = args.output_dir / f"{stem}_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
