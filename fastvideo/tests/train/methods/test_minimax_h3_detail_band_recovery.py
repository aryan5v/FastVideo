# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the detail-band recovery controls (audit 2026-09-08)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

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
    method._interval_generator = generator
    method.student = SimpleNamespace(device=torch.device("cpu"))
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


def test_base_recovery_defines_shared_choice() -> None:
    # Regression: interval sampling crashed at the first update (jobs
    # 7000-7002) when the policy RNG was missing from the base class.
    assert callable(getattr(MiniMaxH3BaseRecoveryMethod, "_sample_interval", None))
    assert callable(getattr(MiniMaxH3BaseRecoveryMethod, "on_train_start", None))


def _selector():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "select_h3_block_map", _REPO_ROOT / "scripts" / "fasth3_sprint" / "select_h3_block_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recut_selector_enforces_structure_and_audio_veto() -> None:
    module = _selector()
    blended = [0.1 * i for i in range(50)]  # importance rises with depth
    audio = [0.0] * 50
    audio[44] = 1.0  # audio-critical late block must survive
    block_map = module.select_map(blended, audio, 34)
    removed = [i for i in range(50) if i not in block_map]
    assert len(block_map) == 34
    assert 44 not in removed and 49 not in removed and set(range(4)) & set(removed) == set()
    assert sum(1 for i in removed if i >= 31) >= 4
    run = 0
    for index in range(50):
        run = run + 1 if index in removed else 0
        assert run <= 2


def test_seam_helper_matches_known_maps() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seams_for_block_map", _REPO_ROOT / "scripts" / "fasth3_sprint" / "seams_for_block_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    activation34 = [0, 1, 2, 3, 5, 12, 14, 17, 20, 22, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    assert module.seams_for_block_map(activation34) == [4, 5, 6, 7, 8, 9, 10, 15]
    activation42 = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    assert module.seams_for_block_map(activation42) == [7, 11, 13, 14]


@pytest.mark.parametrize("launcher", [
    "slurm_h3_detail_band_recovery.sbatch",
    "slurm_h3_base34_recut.sbatch",
    "slurm_h3_base34_from42_recut.sbatch",
])
def test_launcher_scripts_parse(launcher: str) -> None:
    # An apostrophe inside an srun bash -lc '...' block silently ends the
    # quoting and runs container commands on the login shell (jobs 7035/7036).
    import shutil
    import subprocess
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    path = _REPO_ROOT / "scripts" / "fasth3_sprint" / launcher
    subprocess.run(["bash", "-n", str(path)], check=True)
