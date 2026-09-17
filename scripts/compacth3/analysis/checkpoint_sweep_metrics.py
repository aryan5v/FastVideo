#!/usr/bin/env python3
"""Technical A/V retention grading for paired FastH3 checkpoint renders.

The script intentionally reports a *retention index*, not a learned perceptual
quality score.  Every candidate is compared prompt-by-prompt with a reference
render made with the same prompt and seed.  This makes exposure, detail,
motion, temporal stability, and audio regressions visible without claiming to
measure anatomy, semantic correctness, or human preference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np


VIDEO_WEIGHTS = {
    "luma": 0.16,
    "contrast": 0.18,
    "tonal_range": 0.16,
    "sharpness": 0.32,
    "edge_density": 0.18,
}
TEMPORAL_WEIGHTS = {
    "motion": 0.34,
    "jerk_ratio": 0.42,
    "flicker": 0.18,
    "freeze_fraction": 0.06,
}
AUDIO_WEIGHTS = {
    "lufs": 0.30,
    "true_peak": 0.12,
    "silence_fraction": 0.18,
    "spectral_centroid": 0.12,
    "spectral_flatness": 0.16,
    "voice_band_ratio": 0.12,
}


def run_bytes(command: list[str]) -> bytes:
    return subprocess.run(command, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def video_metrics(path: Path) -> dict[str, float]:
    width, height = 208, 120
    raw = run_bytes([
        "ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
        f"fps=6,scale={width}:{height}:flags=area,format=gray", "-f",
        "rawvideo", "-pix_fmt", "gray", "-",
    ])
    frame_size = width * height
    usable = len(raw) // frame_size * frame_size
    frames = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, height,
                                                                  width).astype(np.float32)
    if len(frames) < 3:
        raise RuntimeError(f"Too few decoded frames in {path}")

    frame_means = frames.mean(axis=(1, 2))
    frame_stds = frames.std(axis=(1, 2))
    p05 = np.percentile(frames, 5, axis=(1, 2))
    p95 = np.percentile(frames, 95, axis=(1, 2))

    lap = (-4.0 * frames[:, 1:-1, 1:-1] + frames[:, :-2, 1:-1] +
           frames[:, 2:, 1:-1] + frames[:, 1:-1, :-2] +
           frames[:, 1:-1, 2:])
    gx = np.abs(frames[:, :, 1:] - frames[:, :, :-1])
    gy = np.abs(frames[:, 1:, :] - frames[:, :-1, :])
    sharpness = np.var(lap, axis=(1, 2))
    edge_density = 0.5 * ((gx > 18).mean(axis=(1, 2)) +
                          (gy > 18).mean(axis=(1, 2)))

    delta = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    accel = np.abs(frames[2:] - 2.0 * frames[1:-1] + frames[:-2]).mean(axis=(1, 2))
    motion = float(np.median(delta))
    jerk = float(np.median(accel))
    return {
        "luma": float(np.mean(frame_means)),
        "contrast": float(np.mean(frame_stds)),
        "tonal_range": float(np.mean(p95 - p05)),
        "sharpness": float(np.median(sharpness)),
        "edge_density": float(np.mean(edge_density)),
        "motion": motion,
        "jerk_ratio": jerk / max(motion, 1e-6),
        "flicker": float(np.std(np.diff(frame_means))),
        "freeze_fraction": float(np.mean(delta < 0.55)),
        "black_fraction": float(np.mean(frame_means < 5.0)),
    }


def ebur128(path: Path) -> tuple[float, float]:
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                          check=True)
    summaries = proc.stderr.split("Summary:")
    text = summaries[-1] if len(summaries) > 1 else proc.stderr
    integrated = re.search(r"I:\s*(-?[0-9.]+)\s+LUFS", text)
    peak = re.search(r"Peak:\s*(-?[0-9.]+)\s+dBFS", text)
    return (float(integrated.group(1)) if integrated else math.nan,
            float(peak.group(1)) if peak else math.nan)


def audio_metrics(path: Path) -> dict[str, float]:
    sample_rate = 16_000
    raw = run_bytes([
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
        "-ar", str(sample_rate), "-f", "f32le", "-",
    ])
    audio = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if len(audio) < 2048:
        raise RuntimeError(f"Too little decoded audio in {path}")
    lufs, true_peak = ebur128(path)

    frame = 1024
    hop = 512
    count = 1 + (len(audio) - frame) // hop
    windows = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, frame),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(windows * windows, axis=1) + 1e-15)
    silence_fraction = float(np.mean(20.0 * np.log10(rms + 1e-15) < -45.0))
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(frame), axis=1)) + 1e-12
    power = spectrum * spectrum
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    centroid = np.sum(power * frequencies, axis=1) / np.sum(power, axis=1)
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum,
                                                                      axis=1)
    voice_mask = (frequencies >= 80) & (frequencies <= 4000)
    audible_mask = (frequencies >= 40) & (frequencies <= 7800)
    voice_ratio = np.sum(power[:, voice_mask], axis=1) / np.maximum(
        np.sum(power[:, audible_mask], axis=1), 1e-15)
    return {
        "lufs": lufs,
        "true_peak": true_peak,
        "rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(audio * audio)) + 1e-15)),
        "silence_fraction": silence_fraction,
        "spectral_centroid": float(np.median(centroid)),
        "spectral_flatness": float(np.median(flatness)),
        "voice_band_ratio": float(np.median(voice_ratio)),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def log_distance(value: float, reference: float, scale: float) -> float:
    return abs(math.log(max(value, 1e-9) / max(reference, 1e-9))) / scale


def linear_distance(value: float, reference: float, scale: float) -> float:
    return abs(value - reference) / scale


def component_scores(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    visual_scales = {
        "luma": 25.0,
        "contrast": 15.0,
        "tonal_range": 30.0,
        "sharpness": 0.70,
        "edge_density": 0.35,
    }
    temporal_scales = {
        "motion": 0.70,
        "jerk_ratio": 0.55,
        "flicker": 4.0,
        "freeze_fraction": 0.20,
    }
    audio_scales = {
        "lufs": 6.0,
        "true_peak": 6.0,
        "silence_fraction": 0.25,
        "spectral_centroid": 0.70,
        "spectral_flatness": 0.20,
        "voice_band_ratio": 0.30,
    }

    def weighted_score(weights: dict[str, float], scales: dict[str, float],
                       log_fields: set[str]) -> float:
        distance = 0.0
        for key, weight in weights.items():
            if key in log_fields:
                part = log_distance(candidate[key], reference[key], scales[key])
            else:
                part = linear_distance(candidate[key], reference[key], scales[key])
            distance += weight * min(part, 3.0)
        return 100.0 * math.exp(-distance)

    visual = weighted_score(VIDEO_WEIGHTS, visual_scales,
                            {"sharpness", "edge_density"})
    temporal = weighted_score(TEMPORAL_WEIGHTS, temporal_scales,
                              {"motion", "jerk_ratio"})
    audio = weighted_score(AUDIO_WEIGHTS, audio_scales,
                           {"spectral_centroid", "spectral_flatness",
                            "voice_band_ratio"})

    # Hard technical failures must not be hidden by otherwise similar averages.
    if candidate["black_fraction"] > 0.05:
        visual *= max(0.0, 1.0 - candidate["black_fraction"])
    if candidate["clipped_fraction"] > 1e-4:
        audio *= max(0.70, 1.0 - 10.0 * candidate["clipped_fraction"])
    overall = 0.40 * visual + 0.30 * temporal + 0.30 * audio
    return {"visual": visual, "temporal": temporal, "audio": audio,
            "overall": overall}


def confidence_interval(values: list[float], seed: int = 20260914) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values)
    draws = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pattern", default="34block-step-*")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted(args.root.glob(args.pattern),
                        key=lambda p: int(re.search(r"step-(\d+)", p.name).group(1)))
    reference_files = {p.name: p for p in args.reference.glob("*.mp4")}
    if not reference_files:
        raise SystemExit(f"No MP4s in reference {args.reference}")

    cache: dict[str, dict[str, float]] = {}

    def measure(path: Path) -> dict[str, float]:
        key = str(path)
        if key not in cache:
            cache[key] = {**video_metrics(path), **audio_metrics(path)}
        return cache[key]

    reference = {name: measure(path) for name, path in reference_files.items()}
    per_clip: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for folder in candidates:
        step = int(re.search(r"step-(\d+)", folder.name).group(1))
        candidate_files = {p.name: p for p in folder.glob("*.mp4")}
        common = sorted(reference_files.keys() & candidate_files.keys())
        if len(common) != len(reference_files):
            continue
        scores: list[dict[str, float]] = []
        for name in common:
            metrics = measure(candidate_files[name])
            score = component_scores(metrics, reference[name])
            scores.append(score)
            per_clip.append({"step": step, "prompt": name, **score, **metrics})
        overall_values = [s["overall"] for s in scores]
        low, high = confidence_interval(overall_values, seed=20260914 + step)
        summary.append({
            "step": step,
            "clips": len(scores),
            "overall": float(np.mean(overall_values)),
            "ci95_low": low,
            "ci95_high": high,
            "visual": float(np.mean([s["visual"] for s in scores])),
            "temporal": float(np.mean([s["temporal"] for s in scores])),
            "audio": float(np.mean([s["audio"] for s in scores])),
            "worst_prompt": min(per_clip[-len(scores):], key=lambda x: x["overall"])["prompt"],
            "worst_score": min(overall_values),
        })

    summary.sort(key=lambda row: row["overall"], reverse=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with args.output_prefix.with_suffix(".clips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    payload = {
        "reference": str(args.reference),
        "interpretation": "Technical A/V retention relative to paired reference; not a semantic or human-preference score.",
        "overall_weights": {"visual": 0.40, "temporal": 0.30, "audio": 0.30},
        "summary": summary,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
