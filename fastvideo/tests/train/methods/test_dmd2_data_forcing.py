# SPDX-License-Identifier: Apache-2.0
"""Per-batch data forcing for carried DMD2 (v9, FastGen data-driven regime).

CPU contract tests for ``rollout_data_forcing``: routing by latent
presence under mixed loading, forced-input noising math (uniform grid-rung
draw, per-modality shifts), walk pausing, uniform first-call seeding, and
knob validation. ``rollout_data_forcing: false`` (the default) keeps the
carried walk byte-identical — ``test_dmd2_rollout_carry.py`` covers that
path unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.tests.train.methods.test_dmd2_rollout_carry import (
    _GRID,
    _LATENT_SHAPE,
    _CarryStudent,
    _make_method,
    _stub_losses,
)
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount

_VIDEO_NUMEL = 6
_AUDIO_NUMEL = 2
assert _VIDEO_NUMEL + _AUDIO_NUMEL == _LATENT_SHAPE[1]


class _ForcingStudent(_CarryStudent):
    """Carry fake that also packs real latents for ``latents_source='data'``."""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_sources: list[str] = []
        self.add_noise_calls: list[dict] = []

    def prepare_batch(self, raw_batch, *, generator, latents_source):
        self.prepare_sources.append(latents_source)
        self.prepare_calls.append(raw_batch)
        if latents_source == "data":
            latents = torch.cat(
                (
                    raw_batch["vae_latent"].reshape(1, -1),
                    raw_batch["audio_latent"].reshape(1, -1),
                ),
                dim=1,
            )
        else:
            latents = torch.zeros(_LATENT_SHAPE)
        batch = SimpleNamespace(
            latents=latents,
            timesteps=torch.tensor([0.0]),
            attn_metadata=None,
            attn_metadata_vsa="vsa-metadata",
            dmd_latent_vis_dict={},
            fake_score_latent_vis_dict={},
        )
        self.last_batch = batch
        return batch

    def add_noise(self, clean, noise, timestep):
        noisy = super().add_noise(clean, noise, timestep)
        self.add_noise_calls.append({
            "clean": clean,
            "noise": noise,
            "timestep": float(timestep.reshape(-1)[0]),
            "noisy": noisy,
        })
        return noisy


def _latent_batch(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "vae_latent": torch.randn(1, _VIDEO_NUMEL, generator=generator),
        "audio_latent": torch.randn(1, _AUDIO_NUMEL, generator=generator),
        "text_embedding": torch.randn(1, 4, generator=generator),
    }


def _text_only_batch() -> dict[str, torch.Tensor]:
    """Mixed loading under the t2va schema: latent columns come through empty."""
    return {
        "vae_latent": torch.zeros(1, 0),
        "audio_latent": torch.zeros(1, 0),
        "text_embedding": torch.ones(1, 4),
    }




def test_data_forcing_defaults_off_and_latent_batches_walk() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1, student=_ForcingStudent())
    _stub_losses(method)
    student = method.student

    assert method._rollout_data_forcing is False
    _, _, metrics = method.single_train_step(_latent_batch(), iteration=0)

    assert student.prepare_sources == ["zeros"]
    assert "data_forced" not in metrics
    assert metrics["rollout_step"] == 0.0




def test_routing_by_latent_presence_under_mixed_loading() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1, student=_ForcingStudent(), data_forcing=True)
    _stub_losses(method)
    student = method.student

    _, _, walk_metrics = method.single_train_step(_text_only_batch(), iteration=0)
    _, _, forced_metrics = method.single_train_step(_latent_batch(), iteration=1)

    assert student.prepare_sources == ["zeros", "data"]
    assert walk_metrics["data_forced"] == 0.0
    assert walk_metrics["rollout_step"] == 0.0
    assert forced_metrics["data_forced"] == 1.0
    assert "rollout_step" not in forced_metrics


def test_half_present_latent_pair_fails_loud() -> None:
    method = _make_method(slots=1, sample_type="ode", student=_ForcingStudent(), data_forcing=True)
    batch = _latent_batch()
    batch["audio_latent"] = torch.zeros(1, 0)
    with pytest.raises(ValueError, match="exactly one of"):
        method.single_train_step(batch, iteration=0)




def test_forced_input_is_real_latents_noised_at_a_grid_rung() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1, student=_ForcingStudent(), data_forcing=True)
    _stub_losses(method)
    student = method.student
    method._carry_slot_seeded[0] = True

    batch = _latent_batch(seed=3)
    packed_real = torch.cat(
        (batch["vae_latent"].reshape(1, -1), batch["audio_latent"].reshape(1, -1)),
        dim=1,
    )
    method.single_train_step(batch, iteration=0)

    forced = student.add_noise_calls[0]
    assert forced["timestep"] in [float(t) for t in _GRID]
    torch.testing.assert_close(forced["clean"], packed_real)
    sigma = forced["timestep"] / 1000.0
    torch.testing.assert_close(forced["noisy"], (1.0 - sigma) * packed_real + sigma * forced["noise"])
    main = student.predict_calls[-1]
    assert main["timestep"] == forced["timestep"]
    assert main["grad_enabled"] is True
    assert main["attn_kind"] == "vsa"
    vis = student.last_batch.dmd_latent_vis_dict
    torch.testing.assert_close(vis["generator_timestep"], torch.tensor([forced["timestep"]]))


def test_forced_rung_draw_covers_the_whole_grid_and_never_zero() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1, student=_ForcingStudent(), data_forcing=True)
    _stub_losses(method)
    method._carry_slot_seeded[0] = True

    for call in range(64):
        method.single_train_step(_latent_batch(seed=call), iteration=call)
    drawn = {call["timestep"] for call in method.student.add_noise_calls}
    assert drawn == {float(t) for t in _GRID}
    assert 0.0 not in drawn




def test_forced_batches_pause_the_walk_and_text_batches_resume_it() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1, student=_ForcingStudent(), data_forcing=True)
    _stub_losses(method)

    method.single_train_step(_text_only_batch(), iteration=0)
    carried = method._carry_slots[0]
    assert carried is not None and carried["rung"] == 1
    state_before = carried["state"].clone()

    method.single_train_step(_latent_batch(seed=5), iteration=1)
    method.single_train_step(_latent_batch(seed=6), iteration=2)
    assert method._carry_slots[0] is carried
    torch.testing.assert_close(carried["state"], state_before)

    _, _, metrics = method.single_train_step(_text_only_batch(), iteration=3)
    assert metrics["rollout_step"] == 1.0
    assert method._carry_slots[0] is not None
    assert method._carry_slots[0]["rung"] == 2


def test_forced_batch_at_walk_boundary_leaves_the_boundary_state() -> None:
    method = _make_method(slots=1, grid=[999, 500], sample_type="ode", interval=1, student=_ForcingStudent(),
                          data_forcing=True)
    _stub_losses(method)

    method.single_train_step(_text_only_batch(), iteration=0)
    method.single_train_step(_text_only_batch(), iteration=1)
    assert method._carry_slots[0] is None

    _, _, forced_metrics = method.single_train_step(_latent_batch(), iteration=2)
    assert forced_metrics["data_forced"] == 1.0
    assert method._carry_slots[0] is None
    _, _, metrics = method.single_train_step(_text_only_batch(), iteration=3)
    assert metrics["rollout_step"] == 0.0




def test_forced_first_call_runs_stagger_prewalk_with_uniform_forward_count() -> None:
    forced_student = _ForcingStudent()
    forced_method = _make_method(slots=1, sample_type="ode", interval=1, student=forced_student, data_forcing=True,
                                 rank=1, world=2)
    _stub_losses(forced_method)
    text_student = _ForcingStudent()
    text_method = _make_method(slots=1, sample_type="ode", interval=1, student=text_student, data_forcing=True, rank=1,
                               world=2)
    _stub_losses(text_method)

    latent = _latent_batch(seed=7)
    forced_method.single_train_step(latent, iteration=0)
    text_method.single_train_step(_text_only_batch(), iteration=0)

    assert len(forced_student.predict_calls) == len(_GRID)
    assert len(text_student.predict_calls) == len(_GRID)

    carried = forced_method._carry_slots[0]
    assert carried is not None
    assert carried["rung"] == (1 * 1 + 0) % len(_GRID)
    assert forced_method._carry_slot_seeded[0] is True
    torch.testing.assert_close(carried["raw_batch"]["vae_latent"], latent["vae_latent"])

    _, _, metrics = forced_method.single_train_step(_text_only_batch(), iteration=1)
    assert metrics["rollout_step"] == float(carried["rung"])
    adopted = forced_student.prepare_calls[-1]
    torch.testing.assert_close(adopted["text_embedding"], latent["text_embedding"])




def test_data_forcing_requires_rollout_carry() -> None:
    with pytest.raises(ValueError, match="rollout_carry: true"):
        _make_method(carry=False, sample_type=None, student=_ForcingStudent(), data_forcing=True)


def test_data_forcing_rejects_non_bool() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {"rollout_data_forcing": "yes"})
    object.__setattr__(method, "_rollout_carry", True)
    with pytest.raises(ValueError, match="must be a bool"):
        method._parse_rollout_data_forcing()


def test_data_forcing_requires_explicit_legacy_mixed_regime_opt_in() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {"rollout_data_forcing": True})
    object.__setattr__(method, "_rollout_carry", True)
    with pytest.raises(ValueError, match="not a FastGen recipe"):
        method._parse_rollout_data_forcing()

    method.method_config["allow_mixed_rollout_regimes"] = True
    assert method._parse_rollout_data_forcing() is True


def test_data_forcing_requires_t2va_schema() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "_rollout_mode", "simulate")
    object.__setattr__(method, "_rollout_data_forcing", True)
    object.__setattr__(
        method,
        "training_config",
        SimpleNamespace(data=SimpleNamespace(preprocessed_data_type="text_only")),
    )
    with pytest.raises(ValueError, match="t2va"):
        method._validate_preprocessed_data_type()
    object.__setattr__(
        method,
        "training_config",
        SimpleNamespace(data=SimpleNamespace(preprocessed_data_type="t2va")),
    )
    method._validate_preprocessed_data_type()


def test_batch_classifier_contract() -> None:
    assert DMD2Method._batch_has_latents(_latent_batch()) is True
    assert DMD2Method._batch_has_latents(_text_only_batch()) is False
    assert DMD2Method._batch_has_latents({"text_embedding": torch.ones(1, 4)}) is False
    with pytest.raises(ValueError, match="exactly one of"):
        DMD2Method._batch_has_latents({
            "vae_latent": torch.ones(1, 3),
            "audio_latent": torch.zeros(1, 0),
        })




def _build_forcing_trio(monkeypatch: pytest.MonkeyPatch, *, interval: int) -> DMD2Method:
    from fastvideo.tests.train.methods.test_minimax_h3_dmd2 import (
        _FIXTURE,
        _make_model,
    )
    from fastvideo.train.utils.config import load_run_config

    config = load_run_config(str(_FIXTURE))
    config.method["rollout_mode"] = "simulate"
    config.method["generator_update_interval"] = interval
    config.method["rollout_carry"] = True
    config.method["rollout_carry_slots"] = 1
    config.method["rollout_sample_type"] = "ode"
    config.method["rollout_data_forcing"] = True
    config.method["allow_mixed_rollout_regimes"] = True
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


def test_forced_noising_per_modality_shift_on_real_h3_trio(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forced input mixes each modality at its own shifted sigma."""
    from fastvideo.tests.train.methods.test_minimax_h3_dmd2 import _raw_batch

    method = _build_forcing_trio(monkeypatch, interval=5)
    student = method.student
    method._carry_slot_seeded[0] = True

    records: list[dict] = []
    original_add_noise = student.add_noise

    def spy_add_noise(clean, noise, timestep):
        noisy = original_add_noise(clean, noise, timestep)
        records.append({
            "clean": clean,
            "noise": noise,
            "timestep": timestep,
            "noisy": noisy,
        })
        return noisy

    monkeypatch.setattr(student, "add_noise", spy_add_noise)

    raw = _raw_batch(seed=1)
    loss_map, outputs, metrics = method.single_train_step(raw, iteration=1)

    forced = records[0]
    expected_packed = student.pack_latents(
        raw["vae_latent"].permute(0, 2, 1, 3, 4).to(torch.bfloat16),
        raw["audio_latent"].to(torch.bfloat16),
    )
    torch.testing.assert_close(forced["clean"], expected_packed)
    rung = int(forced["timestep"].reshape(-1)[0])
    assert rung in method.method_config["dmd_denoising_steps"]

    slices = dict(student.modality_slices())
    base = torch.tensor([rung / 1000.0], dtype=torch.float64)
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        sigma = shift_noise_amount(base, shift)
        expected = ((1.0 - sigma) * forced["clean"][:, slices[name]].to(torch.float64) +
                    sigma * forced["noise"][:, slices[name]].to(torch.float64)).to(torch.bfloat16)
        torch.testing.assert_close(forced["noisy"][:, slices[name]], expected)

    assert metrics["data_forced"] == 1.0
    assert metrics["update_student"] == 0.0
    assert loss_map["fake_score_loss"].item() > 0.0
    assert method._carry_slots[0] is None

    method.backward(loss_map, outputs)
    assert method.critic.transformer.scale.grad is not None
    assert torch.isfinite(method.critic.transformer.scale.grad)


def test_forced_student_step_and_walk_adoption_on_real_h3_trio(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forced first call seeds the walk; the walk later reuses its prompt."""
    from fastvideo.tests.train.methods.test_minimax_h3_dmd2 import _raw_batch

    method = _build_forcing_trio(monkeypatch, interval=1)
    student = method.student

    latent_raw = _raw_batch(seed=1)
    loss_map, outputs, metrics = method.single_train_step(latent_raw, iteration=0)

    assert metrics["data_forced"] == 1.0
    assert metrics["update_student"] == 1.0
    assert torch.isfinite(loss_map["total_loss"])
    assert loss_map["generator_loss"].item() > 0.0
    assert "generator_loss_video" in metrics and "generator_loss_audio" in metrics
    carried = method._carry_slots[0]
    assert carried is not None and carried["rung"] == 0
    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is not None
    assert torch.isfinite(student.transformer.scale.grad)

    text_raw = {
        "vae_latent": torch.zeros(1, 0),
        "audio_latent": torch.zeros(1, 0),
        "text_embedding": _raw_batch(seed=9)["text_embedding"],
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
    }
    _, _, walk_metrics = method.single_train_step(text_raw, iteration=1)
    assert walk_metrics["data_forced"] == 0.0
    assert walk_metrics["rollout_step"] == 0.0
    adopted_text = latent_raw["text_embedding"][:, :2].to(torch.bfloat16)
    torch.testing.assert_close(student.transformer.last_encoder_hidden_states, adopted_text)
