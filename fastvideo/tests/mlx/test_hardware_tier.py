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

# Fake 5B id used only to exercise the prefer_5b path (Track D not landed).
_FAKE_5B = "FastVideo/FakeWan2.2-TI2V-5B-Diffusers-NOT-PORTED"


@pytest.mark.parametrize(
    ("memory_gib", "expect_name", "expect_repo", "expect_quant", "expect_cap", "expect_preset"),
    [
        (8.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb"),
        (16.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb"),
        (18.0, "small", MODEL_1_3B_REPO, "int8", TIER_SMALL_MLX_CAP_GIB, "mac-16gb"),
        # Just above small → medium, 1.3B fp16 fallback while FIVE_B_MODEL_REPO is None.
        (18.01, "medium", MODEL_1_3B_REPO, "none", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb"),
        (24.0, "medium", MODEL_1_3B_REPO, "none", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb"),
        (32.0, "medium", MODEL_1_3B_REPO, "none", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb"),
        (40.0, "medium", MODEL_1_3B_REPO, "none", TIER_MEDIUM_MLX_CAP_GIB, "mac-32gb"),
        # Just above medium → large.
        (40.01, "large", MODEL_1_3B_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb"),
        (48.0, "large", MODEL_1_3B_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb"),
        (64.0, "large", MODEL_1_3B_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb"),
        (128.0, "large", MODEL_1_3B_REPO, "none", TIER_LARGE_MLX_CAP_GIB, "mac-64gb"),
    ],
)
def test_recommend_tier_injected_memory_1_3b_fallback(
    memory_gib: float,
    expect_name: str,
    expect_repo: str,
    expect_quant: str,
    expect_cap: float,
    expect_preset: str,
) -> None:
    """Default path: FIVE_B_MODEL_REPO is unset → always 1.3B, quant by band."""
    assert FIVE_B_MODEL_REPO is None, "test assumes Track D 5B repo is not yet published"
    tier = recommend_tier(memory_gib, prefer_5b=True)
    assert tier.name == expect_name
    assert tier.model_repo == expect_repo
    assert tier.quantization == expect_quant
    assert tier.mlx_memory_limit_gib == expect_cap
    assert tier.benchmark_preset == expect_preset
    assert tier.uses_5b is False
    if expect_name == "small":
        assert tier.decoder == "taehv"
        assert tier.modes == "int8"
        assert tier.decoders == "taehv"
        assert tier.max_memory_gib == TIER_SMALL_MAX_GIB
    elif expect_name == "medium":
        assert tier.decoder == "taehv"
        assert tier.modes == "fp16"
        assert tier.max_memory_gib == TIER_MEDIUM_MAX_GIB
    else:
        assert tier.decoder == "wan-vae"
        assert tier.modes == "fp16"
        assert tier.max_memory_gib is None


@pytest.mark.parametrize(
    ("memory_gib", "expect_name", "expect_quant", "expect_decoder"),
    [
        (16.0, "small", "int8", "taehv"),  # small never selects 5B
        (32.0, "medium", "int8", "taehv"),
        (64.0, "large", "none", "wan-vae"),
    ],
)
def test_recommend_tier_prefer_5b_when_repo_known(
    memory_gib: float,
    expect_name: str,
    expect_quant: str,
    expect_decoder: str,
) -> None:
    tier = recommend_tier(memory_gib, prefer_5b=True, five_b_model_repo=_FAKE_5B)
    assert tier.name == expect_name
    assert tier.quantization == expect_quant
    assert tier.decoder == expect_decoder
    if expect_name == "small":
        assert tier.model_repo == MODEL_1_3B_REPO
        assert tier.uses_5b is False
    else:
        assert tier.model_repo == _FAKE_5B
        assert tier.uses_5b is True


def test_prefer_5b_false_keeps_1_3b_even_with_repo() -> None:
    tier = recommend_tier(32.0, prefer_5b=False, five_b_model_repo=_FAKE_5B)
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
    """On the Mac under test this is ~36; on Linux CI it still returns a positive float."""
    gib = detect_unified_memory_gib()
    assert isinstance(gib, float)
    assert gib > 0.0
    # Sanity: not a nonsense petabyte reading.
    assert gib < 1024.0 * 1024.0


def test_detect_falls_back_to_default_when_probes_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._sysctl_memsize_bytes", lambda: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._mlx_device_memory_bytes", lambda mx_module=None: None)
    monkeypatch.setattr("fastvideo.mlx_runtime.hardware_tier._proc_meminfo_bytes", lambda: None)
    assert detect_unified_memory_gib() == DEFAULT_ASSUMED_MEMORY_GIB
    # Safe default maps to the small tier.
    tier = recommend_tier(None)
    # recommend_tier will re-call detect; still defaulted.
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
    assert args.auto_tier_quantization == "int8"
    assert args.auto_tier_benchmark_preset == "mac-16gb"


def test_threshold_boundaries() -> None:
    """Document the inclusive upper bounds: 18 → small, 40 → medium."""
    assert recommend_tier(TIER_SMALL_MAX_GIB).name == "small"
    assert recommend_tier(TIER_SMALL_MAX_GIB + 1e-9).name == "medium"
    assert recommend_tier(TIER_MEDIUM_MAX_GIB).name == "medium"
    assert recommend_tier(TIER_MEDIUM_MAX_GIB + 1e-9).name == "large"
