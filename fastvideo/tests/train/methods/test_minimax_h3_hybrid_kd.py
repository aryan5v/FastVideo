# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for FastH3-native hybrid velocity distillation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.callbacks.minimax_h3_hybrid_export import MiniMaxH3HybridExportCallback
from fastvideo.train.methods.knowledge_distillation.minimax_h3_velocity import MiniMaxH3VelocityKDMethod
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    MiniMaxH3Model,
    _apply_trainable_parameter_patterns,
    shift_noise_amount,
)
from fastvideo.models.dits.minimax_h3_hybrid.linear import initialize_hybrid_parameter

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_hybrid_trainable_mask_freezes_preview_backbone() -> None:
    transformer = torch.nn.Module()
    transformer.to_q = torch.nn.Linear(2, 2)
    transformer.linear_attention = torch.nn.Linear(2, 2)
    transformer.to_out_linear = torch.nn.Linear(2, 2)
    transformer.softmax_gate = torch.nn.Linear(2, 1)

    patterns = ("linear_attention", "to_out_linear", "softmax_gate")
    _apply_trainable_parameter_patterns(transformer, patterns)

    trainable = {name for name, parameter in transformer.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(any(pattern in name for pattern in patterns) for name in trainable)
    assert not transformer.to_q.weight.requires_grad


def test_fasth3_grid_samples_only_four_forward_sigmas() -> None:
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model._sigma_grid_points = 5
    model.training_config = SimpleNamespace(distributed=SimpleNamespace(sp_size=1))
    model.sp_group = None
    observed = set()

    for seed in range(32):
        video_sigma, audio_sigma, index = model._sample_noise_amounts(
            torch.Generator(device="cpu").manual_seed(seed),
            torch.device("cpu"),
        )
        base_sigma = torch.tensor([1.0, 0.75, 0.5, 0.25])[index:index + 1]
        torch.testing.assert_close(video_sigma, shift_noise_amount(base_sigma, 12.0))
        torch.testing.assert_close(audio_sigma, shift_noise_amount(base_sigma, 3.0))
        observed.add(index)

    assert observed == {0, 1, 2, 3}


def test_missing_hybrid_initialization_keeps_far_branch_live_and_residual_zero() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    prefix = "transformer_blocks.0.attn"

    norm = initialize_hybrid_parameter(f"{prefix}.linear_attention.norm.weight", (4, ), device, dtype)
    conv = initialize_hybrid_parameter(f"{prefix}.linear_attention.short_conv.k_sp.weight", (4, 1, 5, 5),
                                       device, dtype)
    down = initialize_hybrid_parameter(f"{prefix}.linear_attention.alpha.down.weight", (4, 8), device, dtype)
    out = initialize_hybrid_parameter(f"{prefix}.to_out_linear.weight", (8, 4), device, dtype)
    gate_bias = initialize_hybrid_parameter(f"{prefix}.softmax_gate.up.bias", (2, ), device, dtype)

    assert norm is not None and torch.equal(norm, torch.ones_like(norm))
    assert conv is not None and torch.count_nonzero(conv) == 4
    assert torch.equal(conv[:, 0, 2, 2], torch.ones(4))
    assert down is not None and torch.count_nonzero(down) > 0
    assert out is not None and torch.count_nonzero(out) == 0
    assert gate_bias is not None
    torch.testing.assert_close(torch.sigmoid(gate_bias), torch.full((2, ), 0.99))


def test_missing_hybrid_random_initialization_is_name_deterministic() -> None:
    name = "transformer_blocks.7.attn.linear_attention.output_gate.down.weight"
    first = initialize_hybrid_parameter(name, (4, 8), torch.device("cpu"), torch.float32)
    second = initialize_hybrid_parameter(name, (4, 8), torch.device("cpu"), torch.float32)
    other = initialize_hybrid_parameter(name.replace(".7.", ".8."), (4, 8), torch.device("cpu"), torch.float32)
    assert first is not None and second is not None and other is not None
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other)


class _FakeJointRole:

    def __init__(self, prediction: tuple[torch.Tensor, torch.Tensor], *, prepares: bool = False) -> None:
        self.prediction = prediction
        self.prepares = prepares
        self.attn_kinds: list[str] = []

    def prepare_batch(self, batch, *, generator, latents_source):
        del batch, generator, latents_source
        assert self.prepares
        return TrainingBatch(
            latents=torch.zeros(1, 1, 1, 1, 1),
            noisy_model_input=torch.zeros(1, 1, 1, 1, 1),
            timesteps=torch.zeros(1),
            current_timestep=2,
        )

    def predict_noise(self, noisy_latents, timesteps, batch, *, conditional, attn_kind):
        del noisy_latents, timesteps, batch
        assert conditional is True
        self.attn_kinds.append(attn_kind)
        return self.prediction


def test_velocity_kd_uses_vsa_teacher_and_hybrid_student_on_same_batch() -> None:
    teacher = _FakeJointRole((torch.zeros(1), torch.ones(1)))
    student = _FakeJointRole((torch.full((1, ), 2.0), torch.full((1, ), 3.0)), prepares=True)
    method = MiniMaxH3VelocityKDMethod.__new__(MiniMaxH3VelocityKDMethod)
    torch.nn.Module.__init__(method)
    method.student = student
    method.teacher = teacher
    method.cuda_generator = torch.Generator(device="cpu")

    losses, outputs, metrics = method.single_train_step({}, 0)

    torch.testing.assert_close(losses["video_velocity_mse"], torch.tensor(4.0))
    torch.testing.assert_close(losses["audio_velocity_mse"], torch.tensor(4.0))
    torch.testing.assert_close(losses["total_loss"], torch.tensor(8.0))
    assert teacher.attn_kinds == ["vsa"]
    assert student.attn_kinds == ["dense"]
    assert outputs["_fv_backward"] == (2, None)
    assert metrics["sigma_grid_index"] == 2


def test_overfit_configs_encode_requested_12_and_16_gpu_meshes() -> None:
    cases = {
        16: (8, 2, 8, 2),
        12: (4, 3, 4, 3),
    }
    for gpu_count, (sp_size, replicate, shard, repeats) in cases.items():
        path = _REPO_ROOT / "examples/train/configs" / f"overfit_minimax_h3_hybrid_kd_{gpu_count}gpu.yaml"
        config = yaml.safe_load(path.read_text())
        distributed = config["training"]["distributed"]
        assert distributed["num_gpus"] == gpu_count
        assert distributed["sp_size"] == sp_size
        assert distributed["tp_size"] == 1
        assert distributed["hsdp_replicate_dim"] == replicate
        assert distributed["hsdp_shard_dim"] == shard
        assert distributed["hsdp_replicate_dim"] * distributed["hsdp_shard_dim"] == gpu_count
        assert list(config["training"]["data"]["data_path"].values()) == [repeats]
        assert config["training"]["data"]["train_batch_size"] == 1
        assert config["models"]["student"]["sigma_grid_points"] == 5
        assert config["models"]["teacher"]["attention_backend"] == "VIDEO_SPARSE_ATTN_H3"


def test_hybrid_export_broadcasts_rank0_failure_before_barrier(tmp_path: Path) -> None:
    barrier_calls: list[bool] = []
    broadcast_values: list[int] = []
    module = "fastvideo.train.callbacks.minimax_h3_hybrid_export"

    def fake_broadcast(tensor: torch.Tensor, src: int = 0) -> None:
        del src
        broadcast_values.append(int(tensor.item()))

    student = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(arch_config=SimpleNamespace(
            hybrid_window_radius=1,
            hybrid_window_chunk=5,
            hybrid_anchor_frames="both",
            hybrid_enable_softmax_gate=True,
            hybrid_delta_rule="vdn_solve",
            hybrid_enable_text_state=True,
            hybrid_short_conv_targets=["k", "v"],
        ))),
        trainable_parameter_patterns=("linear_attention", ),
    )
    callback = MiniMaxH3HybridExportCallback(output_dir=str(tmp_path))
    with (
            patch(f"{module}.get_model_state_dict", return_value={}),
            patch(f"{module}.dist.is_available", return_value=True),
            patch(f"{module}.dist.is_initialized", return_value=True),
            patch(f"{module}.dist.get_rank", return_value=0),
            patch(f"{module}.dist.get_backend", return_value="gloo"),
            patch(f"{module}.dist.broadcast", side_effect=fake_broadcast),
            patch(f"{module}.dist.barrier", side_effect=lambda: barrier_calls.append(True)),
    ):
        with pytest.raises(RuntimeError, match="Hybrid export failed on rank 0"):
            callback.on_train_end(SimpleNamespace(student=student), iteration=1)

    assert barrier_calls == [True]
    assert broadcast_values == [0]
