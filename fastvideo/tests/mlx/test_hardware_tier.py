# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hardware-adaptive MLX model tiering.

Backend-agnostic: injects memory sizes so Metal is never required. Safe to run
under ``mlx[cpu]`` Linux CI and on Apple Silicon Metal boxes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.mlx_runtime.hardware_tier import (
    DEFAULT_ASSUMED_MEMORY_GIB,
    FIVE_B_MODEL_REPO,
    MODEL_1_3B_REPO,
    TIER_LARGE_MLX_CAP_GIB,
    TIER_MEDIUM_MLX_CAP_GIB,
    TIER_MEDIUM_MAX_GIB,
    TIER_SMALL_MAX_GIB,
    TIER_SMALL_MLX_CAP_GIB,
    apply_tier_to_namespace,
    detect_unified_memory_gib,
    recommend_tier,
)


@pytest.mark.parametrize(
    ("memory_gib", "expect_name", "expect_repo", "expect_quant", "expect_cap", "expect_preset", "expect_5b"),
    [
        (8.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb", False),
        (16.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb", False),
        (18.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb", False),
        # Medium/large prefer 5B when FIVE_B_MODEL_REPO is published (Track D Rung 3).
        (18.01, "medium", FIVE_B_MODEL_REPO, "int8", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb", True),
        (24.0, "medium", FIVE_B_MODEL_REPO, "int8", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb", True),
        (32.0, "medium", FIVE_B_MODEL_REPO, "int8", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb", True),
        (40.0, "medium", FIVE_B_MODEL_REPO, "int8", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb", True),
        (40.01, "large", FIVE_B_MODEL_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb", True),
        (48.0, "large", FIVE_B_MODEL_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb", True),
        (64.0, "large", FIVE_B_MODEL_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb", True),
        (128.0, "large", FIVE_B_MODEL_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb", True),
    ],
)
def test_recommend_tier_with_5b_published(
    memory_gib: float,
    expect_name: str,
    expect_repo: str,
    expect_quant: str,
    expect_cap: float,
    expect_preset: str,
    expect_5b: bool,
) -> None:
    assert FIVE_B_MODEL_REPO is not None, "Track D Rung 3 should publish FIVE_B_MODEL_REPO"
    tier = recommend_tier(memory_gib, prefer_5b=True)
    assert tier.name == expect_name
    assert tier.model_repo == expect_repo
    assert tier.quantization == expect_quant
    assert tier.mlx_memory_limit_gib == expect_cap
    assert tier.benchmark_preset == expect_preset
    assert tier.uses_5b is expect_5b


def test_prefer_5b_false_keeps_1_3b() -> None:
    tier = recommend_tier(32.0, prefer_5b=False)
    assert tier.name == "medium"
    assert tier.model_repo == MODEL_1_3B_REPO
    assert tier.quantization == "none"
    assert tier.uses_5b is False


def test_non_positive_memory_falls_back_to_safe_small() -> None:
    tier = recommend_tier(0.0)
    assert tier.name == "small"
    assert tier.model_repo == MODEL_1_3B_REPO
    assert tier.quantization == "int8"


def test_detect_unified_memory_gib_positive_on_this_host() -> None:
    gib = detect_unified_memory_gib()
    assert isinstance(gib, float)
    assert gib > 0.0
    assert gib < 1024.0 * 1024.0


def test_detect_falls_back_to_default_when_probes_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._sysctl_memsize_bytes", lambda: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._mlx_device_memory_bytes", lambda mx_module=None: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._proc_meminfo_bytes", lambda: None)
    assert detect_unified_memory_gib() == DEFAULT_ASSUMED_MEMORY_GIB
    tier = recommend_tier(None)
    assert tier.name == "small"


def test_apply_tier_to_namespace_sets_modes_and_caps() -> None:
    args = SimpleNamespace(
        modes="fp16,bf16",
        decoders="wan-vae",
        mlx_memory_limit_gib=None,
        mlx_disable_cache=False,
    )
    tier = recommend_tier(16.0)
    apply_tier_to_namespace(args, tier)
    assert args.modes == "int8"
    assert args.decoders == "taehv"
    assert args.mlx_memory_limit_gib == TIER_SMALL_MLX_CAP_GIB
    assert args.mlx_disable_cache is True
    assert args.auto_tier_name == "small"
    assert args.auto_tier_model_repo == MODEL_1_3B_REPO


def test_threshold_boundaries() -> None:
    assert recommend_tier(TIER_SMALL_MAX_GIB).name == "small"
    assert recommend_tier(TIER_SMALL_MAX_GIB + 1e-9).name == "medium"
    assert recommend_tier(TIER_MEDIUM_MAX_GIB).name == "medium"
    assert recommend_tier(TIER_MEDIUM_MAX_GIB + 1e-9).name == "large"
