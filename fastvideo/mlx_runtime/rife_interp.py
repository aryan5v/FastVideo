# SPDX-License-Identifier: Apache-2.0
"""Small MLX RIFE wrapper for frame interpolation experiments.

The backend is the Apple-Silicon-native ``rife-mlx`` package, using the
``mlx-community/RIFE-4.25`` weights. Frames are HWC RGB ``uint8`` arrays.
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from collections.abc import Iterable

import numpy as np


class RIFEBackendError(RuntimeError):
    """Raised when the MLX RIFE backend cannot be loaded or run."""


def _require_hwc_rgb(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Validate and normalize a frame as a contiguous HWC RGB NumPy array.
    
    Parameters:
        frame (np.ndarray): Frame data to validate and normalize.
        index (int): Frame index used in validation error messages.
    
    Returns:
        np.ndarray: A contiguous 8-bit RGB frame.
    
    Raises:
        ValueError: If the frame does not have shape HxWx3.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frame {index} must have shape HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


@lru_cache(maxsize=2)
def load_model(version: str = "4.25", weights_dir: str | None = None):
    """
    Load an MLX-native RIFE model for the specified version.
    
    Parameters:
        version (str): RIFE model version to load.
        weights_dir (str | None): Optional directory containing model weights. If omitted, the default model weights are used.
    
    Returns:
        The loaded RIFE model.
    
    Raises:
        RIFEBackendError: If the backend is unavailable or model loading fails.
    """
    try:
        from rife_mlx.utils.weights import build_model
    except ImportError as exc:
        raise RIFEBackendError("MLX RIFE backend is unavailable. Install it into the active venv with "
                               "`uv pip install --python /Users/aryank/claude-fastvideo/FastVideo/.venv/bin/python "
                               "git+https://github.com/xocialize/rife-mlx.git`.") from exc

    try:
        return build_model(version, weights_dir=weights_dir)
    except Exception as exc:  # noqa: BLE001 - preserve exact backend failure.
        raise RIFEBackendError(f"Failed to load MLX RIFE {version}: {exc}") from exc


def interpolate_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    timestep: float = 0.5,
    *,
    model=None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Interpolate a frame at a specified position between two RGB frames.
    
    Parameters:
        frame_a (np.ndarray): The first HWC RGB frame.
        frame_b (np.ndarray): The second HWC RGB frame.
        timestep (float): The interpolation position between the frames, from 0 to 1.
        model: Optional RIFE model to use for interpolation.
        scale (float): Spatial scaling factor used by the model.
    
    Returns:
        np.ndarray: The interpolated HWC RGB frame.
    
    Raises:
        ValueError: If the timestep is outside the open interval from 0 to 1 or the
            frame shapes do not match.
        RIFEBackendError: If model loading or interpolation fails.
    """
    if not 0.0 < timestep < 1.0:
        raise ValueError(f"timestep must be inside (0, 1), got {timestep}")
    img0 = _require_hwc_rgb(frame_a, 0)
    img1 = _require_hwc_rgb(frame_b, 1)
    if img0.shape != img1.shape:
        raise ValueError(f"frame shapes must match, got {img0.shape} and {img1.shape}")

    if model is None:
        model = load_model()
    try:
        from rife_mlx.pipeline_mlx import interpolate_pair as _interpolate_pair

        return _interpolate_pair(model, img0, img1, timestep=timestep, scale=scale)
    except Exception as exc:  # noqa: BLE001 - preserve exact backend failure.
        raise RIFEBackendError(f"MLX RIFE interpolation failed at timestep={timestep}: {exc}") from exc


def interpolate(
    frames: list[np.ndarray] | Iterable[np.ndarray],
    factor: int = 2,
    *,
    model=None,
    scale: float = 1.0,
) -> list[np.ndarray]:
    """
    Generate an interpolated frame sequence while preserving the original frames.
    
    Parameters:
        frames (list[np.ndarray] | Iterable[np.ndarray]): HWC RGB frames in sequence.
        factor (int): Number of output intervals per input interval.
    
    Returns:
        list[np.ndarray]: Original frames with ``factor - 1`` interpolated frames
        inserted between each adjacent pair.
    """
    frame_list = [_require_hwc_rgb(frame, idx) for idx, frame in enumerate(frames)]
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if len(frame_list) < 2 or factor == 1:
        return [frame.copy() for frame in frame_list]

    first_shape = frame_list[0].shape
    for idx, frame in enumerate(frame_list[1:], start=1):
        if frame.shape != first_shape:
            raise ValueError(f"all frames must have the same shape; frame 0={first_shape}, frame {idx}={frame.shape}")

    if model is None:
        model = load_model()

    out: list[np.ndarray] = []
    for left, right in zip(frame_list[:-1], frame_list[1:], strict=True):
        out.append(left)
        for step in range(1, factor):
            out.append(interpolate_pair(left, right, step / factor, model=model, scale=scale))
    out.append(frame_list[-1])
    return out


def _self_test() -> None:
    frame0 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255
    start = time.perf_counter()
    model = load_model()
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    frames = interpolate([frame0, frame1], factor=2, model=model)
    interp_s = time.perf_counter() - start
    assert len(frames) == 3
    assert frames[1].shape == frame0.shape
    assert frames[1].dtype == np.uint8
    print(f"MLX RIFE self-test passed: load_s={load_s:.3f} interp_s={interp_s:.3f} shape={frames[1].shape}")


def main() -> None:
    """Run the command-line self-test when requested."""
    parser = argparse.ArgumentParser(description="MLX RIFE 4.25 frame interpolation smoke test.")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny two-frame interpolation test.")
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("Nothing to do; pass --self-test")
    _self_test()


if __name__ == "__main__":
    main()
