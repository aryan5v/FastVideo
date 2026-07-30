# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in loading of promoted MotionKernel artifacts."""

from __future__ import annotations

from fastvideo.optimization.artifacts import (
    _find_optimized_kernel,
    _load_artifact,
    get_optimized_kernel,
)


def test_promoted_kernel_is_disabled_without_explicit_directory(monkeypatch, ):
    monkeypatch.delenv("FASTVIDEO_AUTOKERNEL_ARTIFACT_DIR", raising=False)
    assert get_optimized_kernel("wan_gated_residual") is None


def test_promoted_kernel_loads_by_declared_operation(monkeypatch, tmp_path):
    artifact = tmp_path / "kernel_wan_gated_residual_3_optimized.py"
    artifact.write_text(
        "KERNEL_TYPE = 'wan_gated_residual'\n"
        "def kernel_fn(residual, x, gate):\n"
        "    return residual + x + gate\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "FASTVIDEO_AUTOKERNEL_ARTIFACT_DIR",
        str(tmp_path),
    )
    _load_artifact.cache_clear()
    _find_optimized_kernel.cache_clear()
    kernel = get_optimized_kernel("wan_gated_residual")
    assert kernel is not None
    assert kernel(1, 2, 3) == 6


def test_mismatched_artifact_operation_uses_fallback(monkeypatch, tmp_path):
    artifact = tmp_path / "kernel_wan_gated_residual_3_optimized.py"
    artifact.write_text(
        "KERNEL_TYPE = 'different_operation'\n"
        "def kernel_fn(*args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "FASTVIDEO_AUTOKERNEL_ARTIFACT_DIR",
        str(tmp_path),
    )
    _load_artifact.cache_clear()
    _find_optimized_kernel.cache_clear()
    assert get_optimized_kernel("wan_gated_residual") is None
