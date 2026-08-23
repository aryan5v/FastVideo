# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax H3 DMD2 distillation.

Covers the packed dual-modality adapter (MiniMaxH3DMDModel) and one full
DMD2Method.single_train_step on a tiny CPU trio: student rollout, critic
flow-matching loss, generator DMD loss, both backwards, both optimizers.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadata
from fastvideo.forward_context import get_forward_context
from fastvideo.pipelines.basic.minimax_h3.packing import audio_latent_num_frames, video_latent_num_frames
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.minimax_h3 import MiniMaxH3DMDModel, MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount
from fastvideo.train.utils.config import load_run_config

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "minimax_h3_dmd2_min.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPERIMENT_CONFIG = (_REPO_ROOT / "examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_nuva_v9_dataforce_vsa64.yaml")
_V10_EXPERIMENT_CONFIG = (_REPO_ROOT / "examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml")
_V10_MAXSHAPE_CONFIG = (_REPO_ROOT / "examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml")
_V10_PREPARE_LAUNCHER = _REPO_ROOT / "examples/train/slurm/prepare_h3_dmd2_v10_slinky.sh"
_H3_SBATCH = _REPO_ROOT / "examples/train/slurm/dmd2_32xgb200.sbatch"
_V10_MAXSHAPE_RUNNER = _REPO_ROOT / "scripts/train/run_h3_v10_maxshape_gate.sh"
_V10_KERNEL_GATE = _REPO_ROOT / "scripts/train/gate_h3_v10_kernel.sh"
_V10_KERNEL_REBUILD = _REPO_ROOT / "scripts/train/rebuild_h3_v10_kernel.sh"
_V10_KERNEL_RECEIPT_HELPER = _REPO_ROOT / "scripts/train/h3_v10_kernel_receipt.py"

# Fixture geometry: video latents [1, 24, 2, 4, 4] and audio latents
# [1, 2, 32, 8]; the packed adapter stores video-major [1, T, C, H, W].
_VIDEO_SHAPE = (1, 2, 24, 4, 4)
_AUDIO_SHAPE = (1, 2, 32, 8)
_PACKED_NUMEL = math.prod(_VIDEO_SHAPE) + math.prod(_AUDIO_SHAPE)


class _TinyJointTransformer(torch.nn.Module):
    """Scale packed H3 rows with one trainable parameter."""

    patch_size = (1, 2, 2)

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.last_encoder_hidden_states: torch.Tensor | None = None
        self.last_attn_metadata = None

    def forward(self, **kwargs):
        self.last_encoder_hidden_states = kwargs["encoder_hidden_states"]
        self.last_attn_metadata = get_forward_context().attn_metadata
        return (
            kwargs["hidden_states"] * self.scale,
            kwargs["audio_hidden_states"] * self.scale,
        )


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    training_config,
    *,
    trainable: bool = True,
    scale: float = 1.0,
) -> MiniMaxH3DMDModel:
    monkeypatch.setattr(MiniMaxH3Model, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3DMDModel.__new__(MiniMaxH3DMDModel)
    model._trainable = trainable
    model.transformer = _TinyJointTransformer(scale)
    model.training_config = training_config
    model.sp_group = None
    model.attention_backend = None
    return model


def _tiny_training_config():
    return SimpleNamespace(
        data=SimpleNamespace(
            num_latent_t=2,
            num_frames=5,
            num_height=64,
            num_width=64,
        ),
        distributed=SimpleNamespace(sp_size=1),
        vsa_sparsity=0.0,
        vsa_tile_size=256,
    )


def _raw_batch(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "vae_latent": torch.randn(1, 24, 2, 4, 4, generator=generator),
        "audio_latent": torch.randn(1, 2, 32, 8, generator=generator),
        "text_embedding": torch.randn(1, 4, 5120, generator=generator),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
    }


def _native_raw_batch(
    width: int,
    height: int,
    num_frames: int,
    *,
    seed: int = 1,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "vae_latent": torch.randn(
            1,
            24,
            video_latent_num_frames(num_frames),
            height // 16,
            width // 16,
            generator=generator,
            dtype=torch.bfloat16,
        ),
        "audio_latent": torch.randn(
            1,
            2,
            32,
            audio_latent_num_frames(num_frames),
            generator=generator,
            dtype=torch.bfloat16,
        ),
        "text_embedding": torch.randn(1, 4, 5120, generator=generator),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        "_shape_bucket_id": f"bucket={width}x{height}-{num_frames}f",
        "info_list": [{
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": 24.0,
            "audio_sample_rate": 32_000,
        }],
    }


def _build_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rollout_mode: str,
    generator_update_interval: int = 1,
) -> DMD2Method:
    config = load_run_config(str(_FIXTURE))
    config.method["rollout_mode"] = rollout_mode
    config.method["generator_update_interval"] = generator_update_interval
    # Distinct role scales keep the critic-vs-teacher DMD gradient non-zero.
    student = _make_model(monkeypatch, config.training, scale=1.0)
    teacher = _make_model(monkeypatch, config.training, trainable=False, scale=0.5)
    critic = _make_model(monkeypatch, config.training, scale=0.25)
    student.init_preprocessors = lambda training_config: None
    method = DMD2Method(
        cfg=config,
        role_models={
            "student": student,
            "teacher": teacher,
            "critic": critic,
        },
    )
    method.cuda_generator = torch.Generator(device="cpu").manual_seed(0)
    return method


# ----------------------------------------------------------------------
# Core gate: one full DMD2 train step on CPU
# ----------------------------------------------------------------------


@pytest.mark.parametrize("rollout_mode", ["data_latent", "simulate"])
def test_dmd2_student_iteration_updates_only_student(
    monkeypatch: pytest.MonkeyPatch,
    rollout_mode: str,
) -> None:
    """Student iterations do not train or step the critic."""
    method = _build_method(monkeypatch, rollout_mode=rollout_mode)
    student = method.student
    teacher = method.teacher
    critic = method.critic

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=0)

    assert metrics["update_student"] == 1.0
    for key in ("total_loss", "generator_loss", "fake_score_loss"):
        assert torch.isfinite(loss_map[key]), key
    assert loss_map["generator_loss"].item() > 0.0
    assert loss_map["fake_score_loss"].item() == 0.0
    torch.testing.assert_close(loss_map["total_loss"], loss_map["generator_loss"])
    assert "generator_pred_video" in method.latent_vis
    assert "real_score_pred_video" in method.latent_vis
    assert "faker_score_pred_video" in method.latent_vis

    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is not None
    assert torch.isfinite(student.transformer.scale.grad)
    assert critic.transformer.scale.grad is None
    assert teacher.transformer.scale.grad is None
    assert method.get_optimizers(0) == [method._student_optimizer]
    assert method.get_lr_schedulers(0) == [method._student_lr_scheduler]
    assert method.get_grad_clip_targets(0) == {"student": student.transformer}

    student_before = student.transformer.scale.detach().clone()
    critic_before = critic.transformer.scale.detach().clone()
    method.optimizers_schedulers_step(0)
    assert student.transformer.scale.detach() != student_before
    torch.testing.assert_close(critic.transformer.scale.detach(), critic_before)


def test_dmd2_critic_iteration_updates_only_critic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off-interval iterations do not train or step the student."""
    method = _build_method(
        monkeypatch,
        rollout_mode="data_latent",
        generator_update_interval=5,
    )

    student = method.student
    critic = method.critic
    loss_map, outputs, metrics = method.single_train_step(_raw_batch(), iteration=1)

    assert metrics["update_student"] == 0.0
    assert loss_map["generator_loss"].item() == 0.0
    assert loss_map["fake_score_loss"].item() > 0.0
    torch.testing.assert_close(loss_map["total_loss"], loss_map["fake_score_loss"])
    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is None
    assert critic.transformer.scale.grad is not None
    assert torch.isfinite(critic.transformer.scale.grad)
    assert method.get_optimizers(1) == [method._critic_optimizer]
    assert method.get_lr_schedulers(1) == [method._critic_lr_scheduler]
    assert method.get_grad_clip_targets(1) == {"critic": critic.transformer}

    student_before = student.transformer.scale.detach().clone()
    critic_before = critic.transformer.scale.detach().clone()
    method.optimizers_schedulers_step(1)
    torch.testing.assert_close(student.transformer.scale.detach(), student_before)
    assert critic.transformer.scale.detach() != critic_before


def test_dmd2_five_step_cadence_and_resume_state(monkeypatch: pytest.MonkeyPatch) -> None:
    method = _build_method(
        monkeypatch,
        rollout_mode="simulate",
        generator_update_interval=5,
    )

    assert [method._should_update_student(i) for i in range(1, 6)] == [
        False,
        False,
        False,
        False,
        True,
    ]
    method.method_config["generator_update_interval"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        method._should_update_student(0)

    method.seed_optimizer_state_for_resume()
    for optimizer in (method._student_optimizer, method._critic_optimizer):
        assert optimizer.state
        assert all("exp_avg" in state for state in optimizer.state.values())

    method.method_config.pop("generator_update_interval")
    assert [method._should_update_student(i) for i in range(1, 6)] == [
        False,
        False,
        False,
        False,
        True,
    ]


# ----------------------------------------------------------------------
# Packed dual-modality adapter units
# ----------------------------------------------------------------------


def test_packed_adapter_roundtrip_and_prepare_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pack/unpack inversion and packed clean latents in the batch."""
    model = _make_model(monkeypatch, _tiny_training_config())
    video = torch.randn(_VIDEO_SHAPE)
    audio = torch.randn(_AUDIO_SHAPE)

    packed = model.pack_latents(video, audio)
    assert packed.shape == (1, _PACKED_NUMEL)
    video_out, audio_out = model.unpack_latents(packed)
    torch.testing.assert_close(video_out, video)
    torch.testing.assert_close(audio_out, audio)

    raw_batch = _raw_batch()
    batch = model.prepare_batch(
        raw_batch,
        generator=torch.Generator().manual_seed(7),
    )
    assert batch.latents.shape == (1, _PACKED_NUMEL)
    video_clean, audio_clean = model.unpack_latents(batch.latents)
    torch.testing.assert_close(
        video_clean,
        raw_batch["vae_latent"].permute(0, 2, 1, 3, 4).to(torch.bfloat16),
    )
    torch.testing.assert_close(audio_clean, batch.audio_latents)


def test_native_layout_is_batch_local_across_successive_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later shape must not mutate how an earlier packed tensor is split."""
    config = _tiny_training_config()
    config.data.native_shape_bucketing = True
    model = _make_model(monkeypatch, config)

    first = model.prepare_batch(
        _native_raw_batch(64, 64, 5),
        generator=torch.Generator().manual_seed(3),
    )
    first_packed = first.latents.clone()
    first_layout = first.minimax_h3_dmd_layout
    second = model.prepare_batch(
        _native_raw_batch(96, 64, 22),
        generator=torch.Generator().manual_seed(4),
    )

    assert first.minimax_h3_dmd_layout is first_layout
    assert first_layout != second.minimax_h3_dmd_layout
    first_video, first_audio = model.unpack_latents(first_packed, layout=first_layout)
    assert first_video.shape == (1, 2, 24, 4, 4)
    assert first_audio.shape == (1, 2, 32, 8)
    second_video, second_audio = model.unpack_latents(second.latents, layout=second.minimax_h3_dmd_layout)
    assert second_video.shape == (1, 7, 24, 4, 6)
    assert second_audio.shape == (1, 2, 32, 37)

    noise = torch.zeros_like(second.latents)
    mixed = model.add_noise_for_batch(second.latents, noise, torch.tensor([500]), second)
    assert mixed.shape == second.latents.shape
    slices = dict(model.modality_slices_for_batch(second))
    assert slices["video"].stop == math.prod(second_video.shape)
    assert slices["audio"].stop == second.latents.shape[1]

    prediction = model.predict_noise(
        second.latents,
        torch.tensor([500]),
        second,
        conditional=True,
    )
    torch.testing.assert_close(prediction, -second.latents)


def test_dmd_losses_follow_successive_native_modality_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Student and critic phases both use the active batch's exact split."""
    method = _build_method(
        monkeypatch,
        rollout_mode="data_latent",
        generator_update_interval=2,
    )
    method.training_config.data.native_shape_bucketing = True
    method.method_config["modality_loss_weights"] = {"video": 0.25, "audio": 2.0}

    student_losses, _, student_metrics = method.single_train_step(
        _native_raw_batch(64, 64, 5),
        iteration=2,
    )
    critic_losses, _, critic_metrics = method.single_train_step(
        _native_raw_batch(96, 64, 22),
        iteration=3,
    )

    assert student_metrics["update_student"] == 1.0
    assert {"generator_loss_video", "generator_loss_audio"} <= student_metrics.keys()
    assert torch.isfinite(student_losses["generator_loss"])
    assert critic_metrics["update_student"] == 0.0
    assert {"fake_score_loss_video", "fake_score_loss_audio"} <= critic_metrics.keys()
    assert torch.isfinite(critic_losses["fake_score_loss"])


@pytest.mark.parametrize(
    ("width", "height", "num_frames"),
    [
        (1344, 768, 124),
        (768, 1344, 362),
        (832, 480, 90),
        (480, 832, 124),
    ],
)
def test_native_validation_accepts_min_max_portrait_and_lowres(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    num_frames: int,
) -> None:
    config = _tiny_training_config()
    config.data.native_shape_bucketing = True
    model = _make_model(monkeypatch, config)
    raw = _native_raw_batch(width, height, num_frames)

    video, audio = model._resolve_clean_latents(raw, "data", torch.bfloat16, torch.device("cpu"))

    assert video.shape == (
        1,
        24,
        video_latent_num_frames(num_frames),
        height // 16,
        width // 16,
    )
    assert audio.shape == (1, 2, 32, audio_latent_num_frames(num_frames))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["info_list"][0].update(width=96), "disagrees with row metadata"),
        (lambda raw: raw["info_list"][0].update(fps=30.0), "24 fps clock"),
        (lambda raw: raw["info_list"][0].update(audio_sample_rate=44_100), "32000 Hz"),
        (lambda raw: raw.update(audio_latent=raw["audio_latent"][..., :-1]), "audio clock"),
        (lambda raw: raw.update(vae_latent=raw["vae_latent"][:, :, :-1]), "vae_latent shape"),
    ],
)
def test_native_validation_rejects_bucket_metadata_and_clock_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    config = _tiny_training_config()
    config.data.native_shape_bucketing = True
    model = _make_model(monkeypatch, config)
    raw = _native_raw_batch(64, 64, 5)
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        model._resolve_clean_latents(raw, "data", torch.bfloat16, torch.device("cpu"))


def test_native_validation_requires_production_canvas_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_training_config()
    config.data.native_shape_bucketing = True
    model = _make_model(monkeypatch, config)
    raw = _native_raw_batch(64, 64, 5)
    raw["_shape_bucket_id"] = "bucket=80x64-5f"
    raw["info_list"][0]["width"] = 80
    raw["vae_latent"] = torch.zeros(1, 24, 2, 4, 5)

    with pytest.raises(ValueError, match="canvas multiple 32"):
        model._resolve_clean_latents(raw, "data", torch.bfloat16, torch.device("cpu"))


def test_legacy_fixed_data_path_still_truncates_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_training_config()
    model = _make_model(monkeypatch, config)
    raw = _raw_batch()
    raw["vae_latent"] = torch.cat((raw["vae_latent"], raw["vae_latent"][:, :, :1]), dim=2)
    raw["audio_latent"] = torch.cat((raw["audio_latent"], raw["audio_latent"][..., :2]), dim=-1)

    video, audio = model._resolve_clean_latents(raw, "data", torch.bfloat16, torch.device("cpu"))

    assert video.shape == (1, 24, 2, 4, 4)
    assert audio.shape == (1, 2, 32, 8)


def test_simulate_zeros_remain_fixed_when_native_data_bucketing_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_training_config()
    config.data.native_shape_bucketing = True
    model = _make_model(monkeypatch, config)

    video, audio = model._resolve_clean_latents({}, "zeros", torch.bfloat16, torch.device("cpu"))

    assert video.shape == (1, 24, 2, 4, 4)
    assert audio.shape == (1, 2, 32, 8)


def test_packed_add_noise_applies_modality_shifts(monkeypatch: pytest.MonkeyPatch) -> None:
    """One shared base timestep must map to two shifted noise amounts."""
    model = _make_model(monkeypatch, _tiny_training_config())
    clean = torch.ones(1, _PACKED_NUMEL)
    noise = torch.zeros(1, _PACKED_NUMEL)

    torch.testing.assert_close(
        model.add_noise(clean, noise, torch.tensor([0])),
        clean,
    )
    torch.testing.assert_close(
        model.add_noise(clean, noise, torch.tensor([1000])),
        noise,
    )

    mixed = model.add_noise(clean, noise, torch.tensor([500]))
    video_mixed, audio_mixed = model.unpack_latents(mixed)
    base = torch.tensor([0.5])
    torch.testing.assert_close(
        video_mixed,
        torch.full(_VIDEO_SHAPE, float(1.0 - shift_noise_amount(base, 12.0))),
    )
    torch.testing.assert_close(
        audio_mixed,
        torch.full(_AUDIO_SHAPE, float(1.0 - shift_noise_amount(base, 3.0))),
    )


def test_packed_predict_noise_plumbs_timesteps_and_tolerates_vsa(monkeypatch: pytest.MonkeyPatch, ) -> None:
    """Explicit method timesteps must rewrite both modality clean-times."""
    model = _make_model(monkeypatch, _tiny_training_config())
    batch = model.prepare_batch(_raw_batch(), generator=torch.Generator().manual_seed(7))
    noisy = torch.randn(1, _PACKED_NUMEL).to(torch.bfloat16)
    timestep = torch.tensor([757], dtype=torch.long)

    # attn_kind="vsa" must silently mean dense (both metadata views are None).
    prediction = model.predict_noise(
        noisy,
        timestep,
        batch,
        conditional=True,
        attn_kind="vsa",
    )

    base = torch.tensor([0.757])
    torch.testing.assert_close(
        batch.timesteps,
        1.0 - shift_noise_amount(base, 12.0),
    )
    torch.testing.assert_close(
        batch.audio_timesteps,
        1.0 - shift_noise_amount(base, 3.0),
    )
    # The unit-scale transformer echoes packed rows, and the H3 wrapper
    # negates them into noise-minus-clean form.
    torch.testing.assert_close(prediction, -noisy)

    x0 = model.predict_x0(noisy, timestep, batch, conditional=True)
    noisy_video, noisy_audio = model.unpack_latents(noisy)
    sigma_video = shift_noise_amount(base, 12.0).to(torch.bfloat16)
    sigma_audio = shift_noise_amount(base, 3.0).to(torch.bfloat16)
    expected = model.pack_latents(
        noisy_video + sigma_video * noisy_video,
        noisy_audio + sigma_audio * noisy_audio,
    )
    torch.testing.assert_close(x0, expected)


def test_uncond_forward_zeroes_text_and_guards_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teacher-CFG unconditional forwards zero text; other policies fail fast."""
    model = _make_model(monkeypatch, _tiny_training_config())
    batch = model.prepare_batch(_raw_batch(), generator=torch.Generator().manual_seed(7))
    noisy = torch.randn(1, _PACKED_NUMEL).to(torch.bfloat16)
    timestep = torch.tensor([500], dtype=torch.long)

    model.predict_noise(
        noisy,
        timestep,
        batch,
        conditional=False,
        cfg_uncond={"text": "zero"},
    )
    assert torch.all(model.transformer.last_encoder_hidden_states == 0)

    model.predict_noise(
        noisy,
        timestep,
        batch,
        conditional=True,
        cfg_uncond={"text": "zero"},
    )
    assert torch.any(model.transformer.last_encoder_hidden_states != 0)

    with pytest.raises(ValueError, match="cfg_uncond"):
        model.predict_noise(noisy, timestep, batch, conditional=False)
    with pytest.raises(ValueError, match="negative-prompt"):
        model.set_requires_negative_conditioning(True)
    model.set_requires_negative_conditioning(False)


# ----------------------------------------------------------------------
# VSA-H3 wiring
# ----------------------------------------------------------------------


def test_prepare_batch_builds_vsa_h3_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VSA-H3 role gets real packed-sequence metadata; dense view stays None."""
    tc = _tiny_training_config()
    tc.vsa_sparsity = 0.35
    tc.vsa_tile_size = 256
    model = _make_model(monkeypatch, tc)
    model.attention_backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3

    batch = model.prepare_batch(_raw_batch(), generator=torch.Generator().manual_seed(7))

    meta = batch.attn_metadata_vsa
    assert isinstance(meta, MiniMaxH3VSAMetadata)
    assert batch.attn_metadata is None
    assert meta.VSA_sparsity == pytest.approx(0.35)
    # Packed layout: 2 text rows | 0 condition rows | 16 stereo audio rows |
    # 8 video rows ([1, 24, 2, 4, 4] latents at patch (1, 2, 2)).
    assert meta.total_seq_length == 26
    assert meta.num_prefix_tiles == 2
    assert meta.num_video_tiles == 1
    assert meta.variable_block_sizes.tolist() == [2, 16, 8]
    assert int(meta.variable_block_sizes.sum()) == meta.total_seq_length


def test_predict_noise_routes_vsa_metadata_by_attn_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Student "vsa" forwards see the VSA metadata; "dense" forwards see None."""
    model = _make_model(monkeypatch, _tiny_training_config())
    model.attention_backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    batch = model.prepare_batch(_raw_batch(), generator=torch.Generator().manual_seed(7))
    noisy = torch.randn(1, _PACKED_NUMEL).to(torch.bfloat16)
    timestep = torch.tensor([757], dtype=torch.long)

    model.predict_noise(noisy, timestep, batch, conditional=True, attn_kind="vsa")
    assert model.transformer.last_attn_metadata is batch.attn_metadata_vsa
    assert isinstance(model.transformer.last_attn_metadata, MiniMaxH3VSAMetadata)

    model.predict_noise(noisy, timestep, batch, conditional=True, attn_kind="dense")
    assert model.transformer.last_attn_metadata is None


def _ctor_training_config() -> SimpleNamespace:
    """The minimum surface MiniMaxH3Model.__init__ reads from TrainingConfig."""
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(uniform_parameter_dtype=False)),
        data=SimpleNamespace(
            train_batch_size=1,
            training_cfg_rate=0.0,
            preprocessed_data_type="t2va",
        ),
        model=SimpleNamespace(enable_gradient_checkpointing_type=None),
    )


def test_per_role_attention_backend_override_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each role's backend reaches the loader; unsupported backends fail fast."""
    captured: dict[str, AttentionBackendEnum | None] = {}

    def _fake_load(**kwargs):
        captured[kwargs["model_path"]] = kwargs["attention_backend"]
        return _TinyJointTransformer()

    monkeypatch.setattr(
        "fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path",
        _fake_load,
    )

    student_config = _ctor_training_config()
    student = MiniMaxH3DMDModel(
        init_from="role/student",
        training_config=student_config,
        trainable=True,
        attention_backend="VIDEO_SPARSE_ATTN_H3",
    )
    teacher = MiniMaxH3DMDModel(
        init_from="role/teacher",
        training_config=_ctor_training_config(),
        trainable=False,
        attention_backend="FLASH_ATTN",
    )

    assert student.attention_backend is AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    assert student_config.pipeline_config.dit_config.uniform_parameter_dtype is False
    assert teacher.attention_backend is AttentionBackendEnum.FLASH_ATTN
    # load_module_from_path turns this request into the construction scope
    # that binds the backend to the transformer's attention layers.
    assert captured["role/student"] is AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    assert captured["role/teacher"] is AttentionBackendEnum.FLASH_ATTN
    assert not any(p.requires_grad for p in teacher.transformer.parameters())

    with pytest.raises(ValueError, match="supports the attention backends"):
        MiniMaxH3DMDModel(
            init_from="role/bad",
            training_config=_ctor_training_config(),
            attention_backend="VIDEO_SPARSE_ATTN",
        )


# ----------------------------------------------------------------------
# Config contracts
# ----------------------------------------------------------------------


def test_h3_dmd2_fixture_resolves_trio_contract() -> None:
    """The fixture must wire the H3 DMD trio through the modular builder path."""
    config = load_run_config(str(_FIXTURE))

    for role in ("student", "teacher", "critic"):
        assert config.models[role]["_target_"] == ("fastvideo.train.models.minimax_h3.MiniMaxH3DMDModel")
    assert config.models["teacher"]["trainable"] is False
    assert config.method["_target_"] == ("fastvideo.train.methods.distribution_matching.dmd2.DMD2Method")
    assert config.training.data.preprocessed_data_type == "t2va"


def test_h3_dmd2_current_config_pins_recipe() -> None:
    """The production config pins the intended H3 DMD2 knobs."""
    config = yaml.safe_load(_EXPERIMENT_CONFIG.read_text())
    method = config["method"]
    training = config["training"]

    for role in ("student", "teacher", "critic"):
        assert config["models"][role]["_target_"] == ("fastvideo.train.models.minimax_h3.MiniMaxH3DMDModel")
    assert config["models"]["teacher"]["trainable"] is False
    assert config["models"]["critic"]["trainable"] is True
    assert method["_target_"] == ("fastvideo.train.methods.distribution_matching.dmd2.DMD2Method")
    assert method["rollout_mode"] == "simulate"
    assert method["rollout_carry"] is True
    assert (method["rollout_carry_slots"] == training["loop"]["gradient_accumulation_steps"])
    # Global batch 128 = 32 DP x accum 4; the carry owns one stream per slot.
    assert training["loop"]["gradient_accumulation_steps"] == 4
    assert method["rollout_sample_type"] == "ode"
    # v9: latent-bearing batches train data-forced (FastGen's data-driven
    # regime); text-only batches keep the carried walk.
    assert method["rollout_data_forcing"] is True
    assert method["generator_update_interval"] == 5
    assert method["real_score_guidance_scale"] == 1.0
    # FastGen h3_new grid: time_shift(linspace(0.999, 0, 5), 12) in base time.
    assert method["dmd_denoising_steps"] == [999, 749, 500, 250]
    assert "warp_denoising_step" not in method
    # f_{1/2.4} == f_{5/12}: FastGen's shifted draw f_5(U) on the shift-12 clock.
    assert method["score_timestep_shift"] == 2.4
    assert method["min_timestep_ratio"] == 0.001
    assert method["max_timestep_ratio"] == 0.999
    assert method["fake_score_loss_space"] == "x0"
    assert method["cfg_uncond"] == {"text": "zero"}
    assert method["fake_score_learning_rate"] == training["optimizer"]["learning_rate"]
    assert method["fake_score_betas"] == [0.9, 0.999]
    assert method["fake_score_lr_scheduler"] == "constant"
    assert training["optimizer"]["betas"] == [0.9, 0.999]
    assert training["dit_precision"] == "fp32"
    assert training["checkpoint"]["output_dir"].endswith("v9_dataforce_vsa64")
    assert training["vsa"] == {"sparsity": 0.9, "tile_size": 64}
    assert (config["models"]["student"]["attention_backend"] == "VIDEO_SPARSE_ATTN_H3")
    assert config["models"]["teacher"]["attention_backend"] == "FLASH_ATTN"
    assert config["models"]["critic"]["attention_backend"] == "FLASH_ATTN"
    assert config["pipeline"]["dit_config"]["uniform_parameter_dtype"] is False
    # Mixed loading is declared t2va (the superset schema); text-only roots
    # yield empty latent columns and route to the carried walk.
    assert training["data"]["preprocessed_data_type"] == "t2va"
    data_paths = training["data"]["data_path"]
    assert any("nuva_t2va" in str(path) for path in data_paths)
    assert any("text_only" in str(path) for path in data_paths)
    assert training["data"]["train_batch_size"] == 1
    assert training["data"]["training_cfg_rate"] == 0.0
    assert config["callbacks"]["grad_clip"]["max_grad_norm"] == 1.0
    # Regional compile of the dense roles; gated on the A/B verdict before
    # launch (see the YAML's PENDING GATE note).
    assert training["model"]["enable_torch_compile"] is True
    # The compile A/B (vsa_gate/compile_ab/VERDICT.md) validated the flip
    # with NO torch_compile_kwargs — the config must not add any.
    assert "torch_compile_kwargs" not in training["model"]


def test_h3_dmd2_v10_config_pins_data_only_native_shape_recipe() -> None:
    """V10 is a fresh 64-GPU, global-batch-64, all-real-latent lineage."""
    config = yaml.safe_load(_V10_EXPERIMENT_CONFIG.read_text())
    method = config["method"]
    training = config["training"]
    distributed = training["distributed"]
    data = training["data"]

    assert method["rollout_mode"] == "data_latent"
    for carry_key in (
            "rollout_carry",
            "rollout_carry_slots",
            "rollout_sample_type",
            "rollout_data_forcing",
    ):
        assert carry_key not in method
    assert method["dmd_denoising_steps"] == [999, 749, 500, 250]
    assert method["fake_score_learning_rate"] == 2.0e-6
    assert training["optimizer"]["learning_rate"] == 2.0e-6

    assert distributed == {
        "num_gpus": 64,
        "sp_size": 1,
        "tp_size": 1,
        "hsdp_replicate_dim": 1,
        "hsdp_shard_dim": 64,
    }
    global_batch = (distributed["num_gpus"] // distributed["sp_size"] * data["train_batch_size"] *
                    training["loop"]["gradient_accumulation_steps"])
    assert global_batch == 64
    assert data["preprocessed_data_type"] == "t2va"
    assert data["native_shape_bucketing"] is True
    assert len(data["data_path"]) == 5
    assert all(path.startswith("/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2/")
               and path.endswith("/data") for path in data["data_path"])

    checkpoint = training["checkpoint"]
    assert checkpoint["output_dir"].endswith("v10_dataonly_mixed_vsa64")
    assert training["loop"]["gradient_accumulation_steps"] == 1
    assert checkpoint["save_inference_checkpoint_on_validation"] is True
    assert checkpoint["inference_checkpoint_role"] == "student"
    assert checkpoint["inference_checkpoint_dtype"] == "bfloat16"
    assert checkpoint["training_state_checkpointing_steps"] == 100
    assert checkpoint["checkpointing_start_step"] == 100
    assert checkpoint["checkpoints_total_limit"] == 3
    assert training["tracker"]["run_name"] == "dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64"
    assert training["model"]["enable_torch_compile"] is True
    assert training["model"]["torch_compile_kwargs"] == {}
    assert training["vsa"] == {"sparsity": 0.9, "tile_size": 64}
    assert config["models"]["student"]["attention_backend"] == "VIDEO_SPARSE_ATTN_H3"
    for role in ("teacher", "critic"):
        assert config["models"][role]["attention_backend"] == "FLASH_ATTN"

    validation = config["callbacks"]["validation"]
    assert validation["dataset_file"].endswith("/validation/heldout64.json")
    assert validation["every_steps"] == 100
    assert validation["run_at_start"] is True
    assert validation["sampling_steps"] == [4]
    assert validation["use_record_dimensions"] is True
    assert validation["max_record_num_frames"] == 345
    assert validation["use_validation_media_conditioning"] is False


def test_h3_dmd2_v10_prepare_launcher_pins_finalized_data_and_execution_clone() -> None:
    """The non-submitting helper gates the dedicated clone and immutable dataset."""
    launcher = _V10_PREPARE_LAUNCHER.read_text()

    assert "/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10" in launcher
    assert ('readonly CONFIG="${REPO}/examples/train/configs/distribution_matching/minimax_h3/'
            'dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml"') in launcher
    assert 'readonly DATA_ROOT="/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2"' in launcher
    assert 'readonly VALIDATION_MANIFEST="${DATA_ROOT}/validation/heldout64.json"' in launcher
    assert "readonly VALIDATION_MAX_RECORD_NUM_FRAMES=345" in launcher
    assert ('readonly OUTPUT_DIR="/mnt/lustre/vlm-wlsaidhi/fastvideo/outputs/'
            'minimax_h3_dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64"') in launcher
    assert "readonly NUM_NODES=16" in launcher
    assert "readonly WORLD_SIZE=64" in launcher
    assert "readonly HSDP_REPLICATE=1" in launcher
    assert "readonly HSDP_SHARD=64" in launcher
    assert "readonly GRADIENT_ACCUMULATION_STEPS=1" in launcher
    assert "readonly GLOBAL_BATCH_SIZE=64" in launcher
    assert "readonly MIN_OUTPUT_FREE_BYTES=" in launcher
    for removed_override in ("CONFIG", "DATA_ROOT", "VALIDATION_MANIFEST", "OUTPUT_DIR"):
        assert f'${{{removed_override}:-' not in launcher
    assert 'readonly REVIEWED_V10_COMMIT="7635a5295b027000a00f6d70789c5cb5886218c3"' in launcher
    assert 'merge-base --is-ancestor "${REVIEWED_V10_COMMIT}" HEAD' in launcher
    assert "actual_data_paths = training[\"data\"][\"data_path\"]" in launcher
    assert 'actual_validation = validation["dataset_file"]' in launcher
    assert "actual_validation_max_record_num_frames" in launcher
    assert 'validation.get("sampling_steps") != [4]' in launcher
    assert "actual_output = training[\"checkpoint\"][\"output_dir\"]" in launcher
    assert "actual_topology != expected_topology" in launcher
    assert "actual_global_batch_size != global_batch_size" in launcher
    assert "kernel receipt source" in launcher
    assert 'readonly MAXSHAPE_AUDIT_ROOT=' in launcher
    assert 'audit_root.glob("job-*/RESULT.json")' in launcher
    assert "fastvideo-h3-v10-maxshape-gate-v1" in launcher
    assert "no successful final-commit 64-GPU 1760x768x362 capacity receipt" in launcher
    assert "available_bytes < MIN_OUTPUT_FREE_BYTES" in launcher
    assert 'require_file "${DATA_ROOT}/READY.json"' in launcher
    assert 'require_file "${source_root}/READY.json"' in launcher
    assert 'require_file "${source_root}/MANIFEST.json"' in launcher
    assert 'require_file "${source_root}/MANIFEST_rows.jsonl"' in launcher
    assert 'require_file "${source_root}/data/map_style_cache/file_info.pkl"' in launcher
    assert "finalize_dataset.py" in launcher
    assert "--verify-only" in launcher
    assert "H3_V10_KERNEL_GATE=1" in launcher
    assert "HSDP_SHARD=%q" in launcher
    assert "--nodes=%q" in launcher
    assert 'git -C "${REPO}" status --porcelain' in launcher
    assert "This helper never calls sbatch" in launcher

    sbatch = _H3_SBATCH.read_text()
    assert "HSDP_REPLICATE * HSDP_SHARD != WORLD_SIZE" in sbatch
    assert 'git -C "${REPO}" status --porcelain' in sbatch
    assert "V10 SOURCE GATE FAILED: execution checkout is dirty" in sbatch
    assert "export HOME=" not in sbatch
    for runtime_variable in (
            "HF_HOME",
            "XDG_CACHE_HOME",
            "TORCH_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "CUDA_CACHE_PATH",
            "NUMBA_CACHE_DIR",
            "WANDB_CONFIG_DIR",
            "WANDB_CACHE_DIR",
            "WANDB_DATA_DIR",
            "NETRC",
    ):
        assert f"export {runtime_variable}=" in sbatch


def test_h3_dmd2_v10_maxshape_gate_matches_production_capacity_contract() -> None:
    config = yaml.safe_load(_V10_MAXSHAPE_CONFIG.read_text())
    training = config["training"]
    distributed = training["distributed"]
    checkpoint = training["checkpoint"]

    assert distributed == {
        "num_gpus": 64,
        "sp_size": 1,
        "tp_size": 1,
        "hsdp_replicate_dim": 1,
        "hsdp_shard_dim": 64,
    }
    assert training["data"]["data_path"].endswith("/v10_maxshape_64g/data")
    assert training["data"]["native_shape_bucketing"] is True
    assert training["data"]["train_batch_size"] == 1
    assert training["loop"] == {"max_train_steps": 2, "gradient_accumulation_steps": 1}
    assert config["method"]["generator_update_interval"] == 2
    assert training["model"]["enable_torch_compile"] is True
    assert training["model"]["torch_compile_kwargs"] == {}
    assert checkpoint["resume_from_checkpoint"] == "latest"
    assert checkpoint["save_inference_checkpoint_on_validation"] is False
    assert checkpoint["training_state_checkpointing_steps"] == 0
    assert "validation" not in config.get("callbacks", {})

    runner = _V10_MAXSHAPE_RUNNER.read_text()
    assert '"${SLURM_JOB_NUM_NODES:-0}" != "16"' in runner
    assert '"${staged}" -ef "${source}"' in runner
    assert "all_64_gpus_sampled" in runner
    assert "critic_grad_finite_positive" in runner
    assert "student_grad_finite_positive" in runner
    assert "dense_teacher_and_critic_compiled" in runner
    assert "vsa_grad_used_triton64" in runner

    sbatch = _H3_SBATCH.read_text()
    assert 'TRAIN_LOG_ROOT="${H3_V10_TRAIN_LOG_ROOT:-${REPO}/examples/train/logs}"' in sbatch


def test_h3_dmd2_v10_kernel_gate_pins_import_order_and_real_gpu_checks() -> None:
    launcher = _V10_PREPARE_LAUNCHER.read_text()
    sbatch = _H3_SBATCH.read_text()
    gate = _V10_KERNEL_GATE.read_text()
    rebuild = _V10_KERNEL_REBUILD.read_text()
    receipt_helper = _V10_KERNEL_RECEIPT_HELPER.read_text()
    expected_pythonpath = (
        "${KERNEL_PREFIX}:${FA4_OVERLAY}:${FA4_CUTLASS_PACKAGES}")

    assert f'V10_PYTHONPATH="{expected_pythonpath}"' in launcher
    assert "H3_V10_KERNEL_GATE=1" in launcher
    assert 'if [ "${H3_V10_KERNEL_GATE}" = "1" ]' in sbatch
    assert "scripts/train/gate_h3_v10_kernel.sh" in sbatch
    assert "python' -m pytest" in sbatch
    assert "test_vsa_triton_backward_scale.py" in gate
    assert "test_forward_matches_reference[64]" in gate
    assert "test_real_sm100a_no_grad_route_receipt" in gate
    assert "timeout --signal=TERM --kill-after=30s 300s" in gate
    assert "FASTVIDEO_KERNEL_V10_RECEIPT.json" in gate
    assert 'if source_commit != execution_commit:' in gate
    assert 'observed_wheel_sha256 != receipt.get("wheel_sha256")' in gate
    assert 'observed_prefix_tree_sha256 != receipt.get("installed_prefix_tree_sha256")' in gate
    assert '"installed_prefix_tree_sha256": installed_prefix_tree_sha256' in rebuild
    assert 'UV="${UV:-${KERNEL_ROOT}/tools/uv}"' in rebuild
    assert "/home/vlm-wlsaidhi/.local/bin/uv" not in rebuild
    assert '"__pycache__" not in relative.parts' in receipt_helper
    assert 'path.suffix != ".pyc"' in receipt_helper
    assert "907f2100e" in rebuild and "56d4a6074" in rebuild
    assert "TORCH_CUDA_ARCH_LIST=10.0a" in rebuild


def test_h3_dmd2_v10_kernel_prefix_receipt_hashes_only_stable_installed_files(tmp_path: Path) -> None:
    from scripts.train.h3_v10_kernel_receipt import RECEIPT_FILENAME, installed_prefix_tree_sha256

    prefix = tmp_path / "prefix"
    package = prefix / "fastvideo_kernel"
    package.mkdir(parents=True)
    installed = package / "kernel.so"
    installed.write_bytes(b"installed-kernel-v1")
    (prefix / "metadata.txt").write_text("metadata-v1", encoding="utf-8")

    receipt = prefix / RECEIPT_FILENAME
    receipt.write_text("receipt-v1", encoding="utf-8")
    bytecode_dir = package / "__pycache__"
    bytecode_dir.mkdir()
    bytecode = bytecode_dir / "module.cpython-312.pyc"
    bytecode.write_bytes(b"bytecode-v1")
    stray_bytecode = package / "generated.pyc"
    stray_bytecode.write_bytes(b"stray-v1")

    original = installed_prefix_tree_sha256(prefix)
    receipt.write_text("receipt-v2", encoding="utf-8")
    bytecode.write_bytes(b"bytecode-v2")
    stray_bytecode.write_bytes(b"stray-v2")
    assert installed_prefix_tree_sha256(prefix) == original

    installed.write_bytes(b"installed-kernel-v2")
    assert installed_prefix_tree_sha256(prefix) != original


def test_validation_dmd_sigmas_match_training_noise_amounts() -> None:
    """``pipeline_config.dmd_denoising_steps`` replays the trained jump points.

    The H3 denoising stage normalizes the method's integer steps to base time
    and lets each scheduler apply its own shift; the resulting clean-times must
    match ``1 - shift_noise_amount(base)`` — the exact noising the packed DMD
    adapter applies during training rollouts — with one forward per step.
    """
    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    steps = [1000, 667, 333]
    base = torch.tensor([step / 1000.0 for step in steps] + [0.0], dtype=torch.float32)
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(sigmas=video.shift_sigmas(base))
    audio.set_timesteps(sigmas=audio.shift_sigmas(base))

    assert video.num_inference_steps == len(steps)
    assert audio.num_inference_steps == len(steps)
    for index, step in enumerate(steps):
        base_step = torch.tensor([step / 1000.0])
        assert video.timesteps[index].item() == pytest.approx(1.0 - shift_noise_amount(base_step, 12.0).item())
        assert audio.timesteps[index].item() == pytest.approx(1.0 - shift_noise_amount(base_step, 3.0).item())


def test_validation_callback_injects_method_denoising_steps() -> None:
    """The callback copies the trained step list onto the validation config."""
    from fastvideo.train.callbacks.validation import ValidationCallback

    callback = ValidationCallback.__new__(ValidationCallback)
    callback.method = SimpleNamespace(method_config={"dmd_denoising_steps": [1000, 667, 333]})

    config = SimpleNamespace(dmd_denoising_steps=None)
    callback._inject_method_denoising_steps(config)
    assert config.dmd_denoising_steps == [1000, 667, 333]

    explicit = SimpleNamespace(dmd_denoising_steps=[1000, 500])
    callback._inject_method_denoising_steps(explicit)
    assert explicit.dmd_denoising_steps == [1000, 500]

    callback.method = SimpleNamespace(method_config={"dmd_denoising_steps": [1000, 757], "warp_denoising_step": True})
    warped = SimpleNamespace(dmd_denoising_steps=None)
    callback._inject_method_denoising_steps(warped)
    assert warped.dmd_denoising_steps is None
