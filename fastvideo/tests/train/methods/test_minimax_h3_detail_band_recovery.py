# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the detail-band recovery controls (audit 2026-09-08)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from fastvideo.train.methods.distribution_matching.minimax_h3_joint_dmd2 import distribution_loss
from fastvideo.train.methods.knowledge_distillation.minimax_h3_base_recovery import MiniMaxH3BaseRecoveryMethod

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _sampler(low_sigma_fraction: float, low_sigma_count: int, seed: int = 7) -> MiniMaxH3BaseRecoveryMethod:
    method = object.__new__(MiniMaxH3BaseRecoveryMethod)
    method._low_sigma_fraction = low_sigma_fraction
    method._low_sigma_count = low_sigma_count
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def shared_choice(upper: int) -> int:
        return int(torch.randint(0, upper, (1, ), generator=generator).item())

    method._shared_choice = shared_choice  # type: ignore[method-assign]
    method.method_config = {"teacher_grid_points": 50}
    method._video_velocity_weight = 1.0
    method._audio_velocity_weight = 1.0
    method._student_state_probability = 0.0
    return method


def test_interval_sampler_defaults_to_uniform_grid() -> None:
    method = _sampler(0.0, 12)
    draws = [method._sample_interval(50) for _ in range(4000)]
    assert all(0 <= interval <= 48 for interval, _ in draws)
    assert all(flag == 0 for _, flag in draws)
    low = sum(1 for interval, _ in draws if interval >= 37) / len(draws)
    assert 0.20 < low < 0.30  # uniform chance of the last 12 intervals is 12/49


def test_interval_sampler_low_sigma_fraction_targets_detail_band() -> None:
    method = _sampler(1.0, 12)
    draws = [method._sample_interval(50) for _ in range(200)]
    assert all(37 <= interval <= 48 for interval, _ in draws)
    assert all(flag == 1 for _, flag in draws)

    mixed = _sampler(0.5, 12, seed=11)
    flags = Counter(flag for _, flag in (mixed._sample_interval(50) for _ in range(2000)))
    assert 0.40 < flags[1] / 2000 < 0.60


def test_control_validation_rejects_out_of_range_knobs() -> None:
    method = _sampler(0.5, 12)
    method._low_sigma_fraction = 1.5
    with pytest.raises(ValueError, match="low_sigma_interval_fraction"):
        method._validate_controls()
    method._low_sigma_fraction = 0.5
    method._low_sigma_count = 0
    with pytest.raises(ValueError, match="low_sigma_interval_count"):
        method._validate_controls()
    method._low_sigma_count = 49
    with pytest.raises(ValueError, match="low_sigma_interval_count"):
        method._validate_controls()
    method._low_sigma_count = 12
    method._audio_velocity_weight = -1.0
    with pytest.raises(ValueError, match="audio_velocity_weight"):
        method._validate_controls()
    method._audio_velocity_weight = 4.0
    method._student_state_probability = 1.5
    with pytest.raises(ValueError, match="student_state_probability"):
        method._validate_controls()
    method._student_state_probability = 0.25
    method._validate_controls()


def test_distribution_loss_stays_finite_when_states_collapse() -> None:
    generated = torch.full((4, ), 2.0)
    real = generated.clone()
    fake = generated + 1.0
    loss = distribution_loss(generated, fake, real)
    assert torch.isfinite(loss)
    # Direction is capped: |fake - real| / floor = 1 / 0.1 = 10, not 1e6.
    assert loss.item() <= 0.5 * 100.0 ** 2


def test_distribution_loss_caps_extreme_direction() -> None:
    generated = torch.full((4, ), 1.0)
    real = generated + 1e-9
    fake = generated + 5.0
    loss = distribution_loss(generated, fake, real, denom_floor_ratio=0.05, grad_cap=100.0)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5 * 100.0 ** 2)


def test_detail_band_config_locks_recipe_and_economics() -> None:
    path = _REPO_ROOT / "examples" / "train" / "configs" / "fasth3_detail_band_recovery.yaml"
    cfg = yaml.safe_load(path.read_text())
    method = cfg["method"]
    assert method["_target_"].endswith("MiniMaxH3BaseRecoveryMethod")
    assert method["low_sigma_interval_fraction"] == 0.5
    assert method["audio_velocity_weight"] >= 4.0
    assert method["audio_seam_weight"] == 2.0
    assert method["video_velocity_weight"] == 1.0
    assert method["denoising_weight"] == 1.0
    assert method["feature_weight"] == 1.0
    assert method["teacher_grid_points"] == 50
    assert method["feature_local_block_indices"] == [7, 11, 13, 14]
    training = cfg["training"]
    assert training["optimizer"]["learning_rate"] == 3e-5
    assert training["dit_precision"] == "fp32"
    assert training["loop"]["max_train_steps"] == 300
    # Checkpoint economics: at most two preserves for a 300-update pilot.
    assert training["checkpoint"]["preserve_every_steps"] >= 150
    assert training["checkpoint"]["checkpoints_total_limit"] <= 3
    student = cfg["models"]["student"]
    assert student["init_from"].endswith("preserved/activation42-step500-job6878/export-500")
    assert student["attention_backend"] == "TORCH_SDPA"


def test_detail_band_34_config_targets_release_candidate_seams() -> None:
    path = _REPO_ROOT / "examples" / "train" / "configs" / "fasth3_detail_band_recovery34.yaml"
    cfg = yaml.safe_load(path.read_text())
    method = cfg["method"]
    # Seams are the local blocks immediately after each gap of the activation-34 map.
    assert method["feature_local_block_indices"] == [4, 5, 6, 7, 8, 9, 10, 15]
    assert method["low_sigma_interval_fraction"] == 0.5
    assert method["audio_velocity_weight"] >= 4.0
    assert method["audio_seam_weight"] == 2.0
    student = cfg["models"]["student"]
    assert student["init_from"].endswith("runs/activation34-prompt58k-recovery/job-6972/export-200")
    assert cfg["training"]["loop"]["max_train_steps"] == 300
    assert cfg["training"]["checkpoint"]["preserve_every_steps"] >= 150
