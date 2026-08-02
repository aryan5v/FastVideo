# SPDX-License-Identifier: Apache-2.0
"""Normalization of the operator-facing optimization environment flags.

Each of these flags is something an operator types on a command line. Before
this was centralized, every site parsed on/off differently and case-sensitively,
so values that plainly read as "off" left the feature on:
``CUDA_GRAPHS=FALSE`` kept CUDA graph replay enabled and ``TIMING=0`` enabled
timing.
"""

from __future__ import annotations

import importlib

import pytest

from fastvideo import envs


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "False", "no", "NO", "off", "OFF", " off "])
def test_off_values_disable_a_flag(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FASTVIDEO_TEST_FLAG", value)
    assert envs.env_flag_enabled("FASTVIDEO_TEST_FLAG", "1") is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "sync", "shadow"])
def test_other_values_enable_a_flag(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FASTVIDEO_TEST_FLAG", value)
    assert envs.env_flag_enabled("FASTVIDEO_TEST_FLAG", "") is True


def test_default_applies_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_TEST_FLAG", raising=False)
    assert envs.env_flag_enabled("FASTVIDEO_TEST_FLAG", "1") is True
    assert envs.env_flag_enabled("FASTVIDEO_TEST_FLAG", "0") is False


@pytest.mark.parametrize("value", ["FALSE", "no", "off", "0"])
def test_cuda_graphs_flag_is_disabled_by_any_off_spelling(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS", value)
    assert envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS is False


def test_cuda_graphs_flag_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS", raising=False)
    assert envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_CUDA_GRAPHS is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "off", "no", ""])
def test_timing_off_values_disable_timing_and_its_modes(monkeypatch, value: str) -> None:
    """`bool(_SETTING)` enabled timing for any non-empty value, including "0"."""
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING", value)
    timing = importlib.reload(importlib.import_module("fastvideo.optimization.timing"))
    assert timing.ENABLED is False
    assert timing.SYNCHRONIZE is False
    assert timing.SHADOW is False


@pytest.mark.parametrize(
    "value, synchronize, shadow",
    [("1", False, False), ("sync", True, False), ("shadow", True, True)],
)
def test_timing_modes_are_gated_on_enabled(monkeypatch, value, synchronize, shadow) -> None:
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING", value)
    timing = importlib.reload(importlib.import_module("fastvideo.optimization.timing"))
    assert timing.ENABLED is True
    assert timing.SYNCHRONIZE is synchronize
    assert timing.SHADOW is shadow


def test_timing_module_is_restored_for_other_tests(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING", raising=False)
    timing = importlib.reload(importlib.import_module("fastvideo.optimization.timing"))
    assert timing.ENABLED is False
