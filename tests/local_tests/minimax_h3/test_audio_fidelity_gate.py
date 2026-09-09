# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the H3 audio/video fidelity gate descriptors."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "audio_fidelity_gate", _REPO_ROOT / "scripts" / "fasth3_sprint" / "audio_fidelity_gate.py")
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def _write_wav(path: Path, signal: np.ndarray, rate: int = 32000) -> None:
    pcm = np.clip(signal, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        stereo = np.repeat(pcm[:, None], 2, axis=1)
        handle.writeframes((stereo * 32767.0).astype("<i2").tobytes())


def test_hf_ratio_and_kurtosis_separate_bright_transient_from_muffled(tmp_path: Path) -> None:
    t = np.arange(32000 * 2, dtype=np.float32) / 32000.0
    bright = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)
    impulse = np.zeros_like(t)
    impulse[::4000] = 0.9
    bright = bright + impulse
    muffled = 0.3 * np.sin(2 * np.pi * 300 * t)

    bright_path = tmp_path / "bright.wav"
    muffled_path = tmp_path / "muffled.wav"
    _write_wav(bright_path, bright)
    _write_wav(muffled_path, muffled)

    bright_metrics = gate.audio_metrics(bright_path)
    muffled_metrics = gate.audio_metrics(muffled_path)

    assert bright_metrics["hf4_ratio"] > 10 * muffled_metrics["hf4_ratio"]
    assert bright_metrics["env_kurtosis"] > 3 * muffled_metrics["env_kurtosis"]
    assert bright_metrics["centroid_hz"] > muffled_metrics["centroid_hz"]


def test_silence_ratio_detects_collapsed_audio(tmp_path: Path) -> None:
    t = np.arange(32000, dtype=np.float32) / 32000.0
    speech_like = 0.4 * np.sin(2 * np.pi * 440 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    near_silent = 1e-4 * np.sin(2 * np.pi * 440 * t)

    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    _write_wav(loud, speech_like)
    _write_wav(quiet, near_silent)

    assert gate.audio_metrics(loud)["silence_ratio"] < 0.5
    assert gate.audio_metrics(quiet)["silence_ratio"] > 0.9


def test_log_spectral_distance_zero_for_identical_and_positive_for_shifted(tmp_path: Path) -> None:
    t = np.arange(32000, dtype=np.float32) / 32000.0
    signal = 0.3 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, signal)
    _write_wav(second, 0.3 * np.sin(2 * np.pi * 500 * t))

    same = gate.audio_metrics(first)
    lsd_same = float(np.sqrt(np.mean((np.array(same["log_mel"]) - np.array(same["log_mel"])) ** 2)))
    lsd_shift = float(
        np.sqrt(np.mean((np.array(gate.audio_metrics(second)["log_mel"]) - np.array(same["log_mel"])) ** 2)))
    assert lsd_same == pytest.approx(0.0)
    assert lsd_shift > 1.0


def test_video_metrics_track_sharpness_and_motion(tmp_path: Path) -> None:
    pytest.importorskip("subprocess")
    static = tmp_path / "static.mp4"
    moving = tmp_path / "moving.mp4"
    for path, shift in ((static, 0), (moving, 12)):
        frames = []
        for f in range(124):
            img = np.zeros((gate.H, gate.W), np.uint8)
            x0 = (shift * f) % (gate.W - 80)
            img[100:300, x0:x0 + 80] = 255
            img[100:300, x0:x0 + 80:4] = 0
            frames.append(img)
        raw = np.stack(frames).tobytes()
        import subprocess
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{gate.W}x{gate.H}", "-r", "24",
            "-i", "-", "-frames:v", "124", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)
        ],
                       input=raw,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    static_metrics = gate.video_metrics(static)
    moving_metrics = gate.video_metrics(moving)
    assert moving_metrics["motion"] > 5 * static_metrics["motion"]
    assert static_metrics["lapvar"] > 0.0
