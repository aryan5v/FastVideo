# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


def _method_with_ratios(min_ratio, max_ratio) -> DMD2Method:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {
        "min_timestep_ratio": min_ratio,
        "max_timestep_ratio": max_ratio,
    })
    object.__setattr__(method, "student", SimpleNamespace(num_train_timesteps=1000))
    return method


def test_dmd2_score_timestep_bounds_apply_ratios() -> None:
    method = _method_with_ratios(0.02, 0.98)
    assert method._parse_score_timestep_bounds() == (20, 980)


def test_dmd2_score_timestep_bounds_default_to_full_range() -> None:
    method = _method_with_ratios(None, None)
    assert method._parse_score_timestep_bounds() == (0, 1000)


@pytest.mark.parametrize(
    ("min_ratio", "max_ratio"),
    [(-0.1, 0.9), (0.8, 0.2), (0.0, 1.1)],
)
def test_dmd2_score_timestep_bounds_reject_invalid_ranges(
    min_ratio: float,
    max_ratio: float,
) -> None:
    method = _method_with_ratios(min_ratio, max_ratio)
    with pytest.raises(ValueError, match="0 <= min <= max <= 1"):
        method._parse_score_timestep_bounds()


@pytest.mark.parametrize("warp_max", [0.0, -0.1, 1.1])
def test_dmd2_score_timestep_warp_max_rejects_invalid_endpoint(warp_max: float) -> None:
    method = _method_with_ratios(0.001, 0.999)
    lo, hi = method._parse_score_timestep_bounds()
    object.__setattr__(method, "_score_min_timestep", lo)
    object.__setattr__(method, "_score_max_timestep", hi)
    method.method_config["score_timestep_warp_max"] = warp_max
    with pytest.raises(ValueError, match="0 < max <= 1"):
        method._parse_score_timestep_warp_max()


def test_dmd2_score_timestep_warp_max_must_cover_upper_bound() -> None:
    method = _method_with_ratios(0.001, 1.0)
    lo, hi = method._parse_score_timestep_bounds()
    object.__setattr__(method, "_score_min_timestep", lo)
    object.__setattr__(method, "_score_max_timestep", hi)
    method.method_config["score_timestep_warp_max"] = 0.999
    with pytest.raises(ValueError, match="must not exceed"):
        method._parse_score_timestep_warp_max()


def test_dmd2_score_timestep_continuous_requires_bool() -> None:
    method = _method_with_ratios(0.001, 0.999)
    method.method_config["score_timestep_continuous"] = 1
    with pytest.raises(ValueError, match="must be a bool"):
        method._parse_score_timestep_continuous()


def _sampler(min_ratio: float, max_ratio: float, shift: float) -> DMD2Method:
    method = _method_with_ratios(min_ratio, max_ratio)
    lo, hi = method._parse_score_timestep_bounds()
    object.__setattr__(method, "_score_min_timestep", lo)
    object.__setattr__(method, "_score_max_timestep", hi)
    object.__setattr__(method, "_score_timestep_shift", shift)
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(0))
    method.student.shift_and_clamp_timestep = lambda t: t
    return method


def test_uniform_sampler_draws_in_bounds_without_boundary_atoms() -> None:
    """shift=1 samples uniformly inside the configured integer bounds."""
    method = _sampler(0.02, 0.98, shift=1.0)
    device = torch.device("cpu")
    draws = torch.cat([method._sample_score_timestep(device) for _ in range(4000)])

    assert int(draws.min()) >= 20
    assert int(draws.max()) <= 980
    assert int((draws == 20).sum()) < 20
    assert int((draws == 980).sum()) < 20


def test_shifted_sampler_respects_bounds() -> None:
    method = _sampler(0.005, 0.98, shift=12.0)
    device = torch.device("cpu")
    draws = torch.cat([method._sample_score_timestep(device) for _ in range(2000)])

    assert int(draws.min()) >= 5
    assert int(draws.max()) <= 980
