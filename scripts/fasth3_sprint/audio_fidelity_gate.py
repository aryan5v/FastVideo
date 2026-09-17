#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fidelity gate for H3 recovery media: candidate vs Base-H3 reference panel.

The exact-speech WER gate (verify_speech_asr.py) passes on audio whose
high-frequency content and transient structure are gone, because Whisper is
robust to band-limiting. This gate measures the descriptors that actually
moved in the 2026-09-08 audit: audio HF energy ratio, spectral centroid,
envelope kurtosis (transients), silence ratio, log-spectral distance, and
video sharpness / motion / contrast. All are no-reference or reference-ratio
metrics computed from the decoded media, so they run on a CPU node.

Usage::

    python scripts/fasth3_sprint/audio_fidelity_gate.py \
        --candidate-dir videos/base42-prompt58k-five/job-6967 \
        --reference-dir videos/threeway-five/job-6954/base-h3 \
        --receipt-out /tmp/fidelity.json

Exit code 0 when every threshold passes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
FRAME_IDX = (0, 20, 40, 60, 80, 100, 120)
H, W = 480, 832
DEFAULT_PROMPTS = ("00_speech_exact_presenter", "01_motorcycle_tracking", "02_mechanical_press",
                   "03_two_shot_transition", "04_glass_water_closeup")
DEFAULT_THRESHOLDS = {
    "hf4_ratio": 0.50,
    "kurtosis_ratio": 0.35,
    "silence_delta": 0.15,
    "lsd_mean": 1.50,
    "motion_ratio": 0.70,
    "contrast_ratio": 0.85,
    "lapvar_ratio": 0.80,
}


def _run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout


def audio_signal(clip: Path) -> np.ndarray:
    raw = _run(["ffmpeg", "-v", "error", "-i", str(clip), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"])
    x = np.frombuffer(raw, dtype=np.float32).copy()
    return x[np.isfinite(x)]


def stereo_signal(clip: Path) -> np.ndarray:
    raw = _run(["ffmpeg", "-v", "error", "-i", str(clip), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"])
    x = np.frombuffer(raw, dtype=np.float32).copy()
    return x[np.isfinite(x)][: x.size // 2 * 2].reshape(-1, 2)


def _stft(x: np.ndarray, nper: int = 512, hop: int = 128) -> tuple[np.ndarray, np.ndarray]:
    windows = np.lib.stride_tricks.sliding_window_view(x, nper)[::hop]
    if windows.shape[0] == 0:
        raise ValueError("Signal too short for STFT")
    frames = windows * np.hanning(nper)
    spec = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(nper, 1.0 / SR)
    return freqs, np.abs(spec) ** 2


def mel_log_spectrum(power: np.ndarray, freqs: np.ndarray, bands: int = 40) -> np.ndarray:
    edges = np.linspace(0.0, 2595.0 * math.log10(1.0 + (SR / 2) / 700.0), bands + 2)
    hz = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    out = np.zeros((power.shape[0], bands), dtype=np.float64)
    for b in range(bands):
        lo, hi = hz[b], hz[b + 2]
        sel = (freqs >= lo) & (freqs < hi)
        out[:, b] = power[:, sel].sum(axis=1) if sel.any() else 0.0
    return np.log(out + 1e-9)


def audio_metrics(clip: Path) -> dict:
    x = audio_signal(clip)
    if x.size < SR // 2:
        return {}
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    freqs, power = _stft(x)
    psd = power.mean(axis=0) + 1e-12
    total = psd.sum()
    centroid = float((freqs * psd).sum() / total)
    hf4 = float(psd[freqs >= 4000].sum() / total)
    hf6 = float(psd[freqs >= 6000].sum() / total)
    flat = float(np.exp(np.log(power + 1e-12).mean(axis=1)).mean() / power.mean(axis=1).mean())
    env = np.sqrt(power.sum(axis=1))
    env = env - env.mean()
    mod = 0.0
    if env.std() > 1e-9:
        spectrum = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
        ef = np.fft.rfftfreq(env.size, d=128.0 / SR)
        band = spectrum[(ef >= 2.0) & (ef <= 8.0)].sum()
        mod = float(band / max(spectrum[ef > 0.5].sum(), 1e-12))
    kurt = float(((env / (env.std() + 1e-12)) ** 4).mean()) if env.std() > 1e-9 else 0.0
    frames = x[: x.size // 400 * 400].reshape(-1, 400)
    db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    stereo = stereo_signal(clip)
    mid = stereo.mean(axis=1)
    side = (stereo[:, 0] - stereo[:, 1]) / 2.0
    side_over_mid = float(np.sqrt(np.mean(side ** 2)) / max(np.sqrt(np.mean(mid ** 2)), 1e-9))
    return {
        "rms_db": 20.0 * math.log10(rms),
        "peak_db": 20.0 * math.log10(peak),
        "centroid_hz": centroid,
        "hf4_ratio": hf4,
        "hf6_ratio": hf6,
        "spectral_flatness": flat,
        "mod2_8_ratio": mod,
        "env_kurtosis": kurt,
        "silence_ratio": float(np.mean(db < -45.0)),
        "side_over_mid": side_over_mid,
        "log_mel": mel_log_spectrum(power, freqs).mean(axis=0).tolist(),
    }


def video_metrics(clip: Path) -> dict:
    lumas = []
    for idx in FRAME_IDX:
        raw = _run([
            "ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f",
            "rawvideo", "-pix_fmt", "gray", "-"
        ])
        if len(raw) != H * W:
            continue
        lumas.append(np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(H, W) / 255.0)
    if not lumas:
        return {}
    stack = np.stack(lumas)
    lap = (stack[:-2, 1:-1, 1:-1] * 4 - stack[:-2, :-2, 1:-1] - stack[:-2, 2:, 1:-1] - stack[:-2, 1:-1, :-2] -
           stack[:-2, 1:-1, 2:])
    diffs = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return {
        "luma_mean": float(stack.mean()),
        "contrast": float(stack.std()),
        "lapvar": float(lap.var()),
        "motion": float(diffs.mean()),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate(candidate: Path, reference: Path, prompts: tuple[str, ...]) -> dict:
    rows = []
    for prompt in prompts:
        cand_clip = candidate / f"{prompt}.mp4"
        ref_clip = reference / f"{prompt}.mp4"
        if not cand_clip.exists() or not ref_clip.exists():
            continue
        ca, ra = audio_metrics(cand_clip), audio_metrics(ref_clip)
        cv, rv = video_metrics(cand_clip), video_metrics(ref_clip)
        if not ca or not ra or not cv or not rv:
            continue
        lsd = float(np.sqrt(np.mean((np.array(ca["log_mel"]) - np.array(ra["log_mel"])) ** 2)))
        rows.append({
            "prompt": prompt,
            "audio": {k: v for k, v in ca.items() if k != "log_mel"},
            "reference_audio": {k: v for k, v in ra.items() if k != "log_mel"},
            "video": cv,
            "reference_video": rv,
            "lsd": lsd,
        })
    if not rows:
        raise SystemExit("No overlapping prompt media between candidate and reference directories")
    ratios = {
        "hf4_ratio": _mean([r["audio"]["hf4_ratio"] / max(r["reference_audio"]["hf4_ratio"], 1e-6) for r in rows]),
        "kurtosis_ratio": _mean(
            [r["audio"]["env_kurtosis"] / max(r["reference_audio"]["env_kurtosis"], 1e-6) for r in rows]),
        "silence_delta": _mean([abs(r["audio"]["silence_ratio"] - r["reference_audio"]["silence_ratio"])
                                for r in rows]),
        "lsd_mean": _mean([r["lsd"] for r in rows]),
        "motion_ratio": _mean([r["video"]["motion"] / max(r["reference_video"]["motion"], 1e-6) for r in rows]),
        "contrast_ratio": _mean([r["video"]["contrast"] / max(r["reference_video"]["contrast"], 1e-6) for r in rows]),
        "lapvar_ratio": _mean([r["video"]["lapvar"] / max(r["reference_video"]["lapvar"], 1e-6) for r in rows]),
    }
    return {"prompts": [r["prompt"] for r in rows], "metrics": ratios, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--prompts", nargs="*", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="JSON file overriding DEFAULT_THRESHOLDS")
    args = parser.parse_args()
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        thresholds.update(json.loads(args.thresholds.read_text()))
    result = evaluate(args.candidate_dir, args.reference_dir, tuple(args.prompts))
    failures = []
    for name, value in result["metrics"].items():
        limit = thresholds[name]
        ok = value >= limit if name.endswith("ratio") else value <= limit
        if not ok:
            failures.append(f"{name}={value:.3f} (limit {limit})")
    result["thresholds"] = thresholds
    result["passed"] = not failures
    result["failures"] = failures
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"metrics": {k: round(v, 4) for k, v in result["metrics"].items()},
                      "passed": result["passed"], "failures": failures}, indent=1))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
