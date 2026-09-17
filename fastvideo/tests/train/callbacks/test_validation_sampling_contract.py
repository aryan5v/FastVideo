# SPDX-License-Identifier: Apache-2.0
"""Validation must sample at the operating point training teaches.

Two independent knobs have to survive the training-config -> validation hop:
the few-step denoising ladder and the VSA attention contract (sparsity AND
tile geometry). v8 shipped 2400 steps of validation that missed both — three
forwards on the scheduler's native grid at tile 256, while training ran four
forwards on ``[999, 749, 500, 250]`` at tile 64 — with nothing raising.
"""

from __future__ import annotations

import types

import pytest

from fastvideo.train.callbacks.validation import ValidationCallback

LADDER = [999, 749, 500, 250]


def _callback(**kwargs):
    """A callback instance without touching distributed state."""
    return ValidationCallback(
        pipeline_target="fastvideo.pipelines.basic.minimax_h3."
        "minimax_h3_pipeline.MiniMaxH3Pipeline",
        dataset_file="unused.json",
        **kwargs,
    )


def _method(ladder=LADDER):
    cfg = {} if ladder is None else {"dmd_denoising_steps": list(ladder)}
    return types.SimpleNamespace(method_config=cfg)


def test_ladder_is_inherited_when_unset() -> None:
    cb = _callback(sampling_steps=[4])
    assert cb.sampling_timesteps is None
    cb._adopt_training_sampling_contract(_method())
    assert cb.sampling_timesteps == LADDER


def test_matching_ladder_is_accepted() -> None:
    cb = _callback(sampling_steps=[4], sampling_timesteps=LADDER)
    cb._adopt_training_sampling_contract(_method())
    assert cb.sampling_timesteps == LADDER


def test_diverging_ladder_raises() -> None:
    cb = _callback(sampling_steps=[4], sampling_timesteps=[1000, 667, 333])
    with pytest.raises(ValueError, match="disagrees with the trained ladder"):
        cb._adopt_training_sampling_contract(_method())


@pytest.mark.parametrize("ladder", [None, []])
def test_non_dmd_methods_are_left_alone(ladder) -> None:
    cb = _callback(sampling_steps=[40])
    cb._adopt_training_sampling_contract(_method(ladder))
    assert cb.sampling_timesteps is None


def test_method_without_config_is_tolerated() -> None:
    cb = _callback(sampling_steps=[40])
    cb._adopt_training_sampling_contract(types.SimpleNamespace())
    assert cb.sampling_timesteps is None


def test_attention_contract_accepts_matching_args() -> None:
    tc = types.SimpleNamespace(vsa_sparsity=0.9, vsa_tile_size=64)
    args = types.SimpleNamespace(VSA_sparsity=0.9, VSA_tile_size=64)
    ValidationCallback._assert_attention_contract(args, tc)


def test_attention_contract_catches_tile_size_drift() -> None:
    """The exact v8 failure: sparsity propagated, tile size left at default."""
    tc = types.SimpleNamespace(vsa_sparsity=0.9, vsa_tile_size=64)
    args = types.SimpleNamespace(VSA_sparsity=0.9, VSA_tile_size=256)
    with pytest.raises(ValueError, match="VSA_tile_size=256"):
        ValidationCallback._assert_attention_contract(args, tc)


def test_attention_contract_catches_sparsity_drift() -> None:
    tc = types.SimpleNamespace(vsa_sparsity=0.9, vsa_tile_size=64)
    args = types.SimpleNamespace(VSA_sparsity=0.0, VSA_tile_size=64)
    with pytest.raises(ValueError, match="VSA_sparsity=0.0"):
        ValidationCallback._assert_attention_contract(args, tc)


def test_make_inference_args_propagates_both_vsa_knobs() -> None:
    """The missing line that caused the drift, pinned at its source."""
    from fastvideo.train.utils.moduleloader import make_inference_args
    src = __import__("inspect").getsource(make_inference_args)
    assert "args.VSA_sparsity = tc.vsa_sparsity" in src
    assert "args.VSA_tile_size = tc.vsa_tile_size" in src
