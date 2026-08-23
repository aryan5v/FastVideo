# SPDX-License-Identifier: Apache-2.0
"""Carried backward-simulation rollout for DMD2 (FastGen port).

CPU contract tests for ``rollout_carry``: slot round-robin, rung
progression and clearing, first-ever staggered starts with a uniform
pre-walk, ODE renoise with per-modality shifts, carried conditioning,
and knob validation. ``rollout_carry: false`` (the default) keeps the
existing rollout path byte-identical; the pre-existing DMD2 suites
(``test_minimax_h3_dmd2.py`` et al.) cover that path unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.minimax_h3 import MiniMaxH3DMDModel
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount

_GRID = [999, 749, 500, 250]
_LATENT_SHAPE = (1, 8)


class _CarryStudent:
    """Rectified-flow fake on one packed tensor (sigma = t / 1000)."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.prepare_calls: list[dict] = []
        self.predict_calls: list[dict] = []
        self.last_batch: SimpleNamespace | None = None
        self.last_pred: torch.Tensor | None = None

    def prepare_batch(self, raw_batch, *, generator, latents_source):
        assert latents_source == "zeros"
        self.prepare_calls.append(raw_batch)
        batch = SimpleNamespace(
            latents=torch.zeros(_LATENT_SHAPE),
            timesteps=torch.tensor([0.0]),
            attn_metadata=None,
            attn_metadata_vsa="vsa-metadata",
            dmd_latent_vis_dict={},
            fake_score_latent_vis_dict={},
        )
        self.last_batch = batch
        return batch

    @staticmethod
    def _sigma(timestep: torch.Tensor) -> float:
        return float(timestep.reshape(-1)[0]) / 1000.0

    def predict_x0(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        self.predict_calls.append({
            "timestep": float(timestep.reshape(-1)[0]),
            "grad_enabled": torch.is_grad_enabled(),
            "attn_kind": attn_kind,
        })
        pred = noisy * 0.5
        self.last_pred = pred
        return pred

    def add_noise(self, clean, noise, timestep):
        sigma = self._sigma(timestep)
        return (1.0 - sigma) * clean + sigma * noise

    def extract_eps(self, noisy, clean, timestep):
        sigma = self._sigma(timestep)
        return (noisy - (1.0 - sigma) * clean) / sigma


def _make_method(
    *,
    slots: int = 1,
    grid: list[int] | None = None,
    interval: int = 1,
    sample_type: str | None = "ode",
    carry: bool = True,
    rank: int = 0,
    world: int = 1,
    grad_accum: int | None = None,
    rollout_mode: str = "simulate",
    student: object | None = None,
    data_forcing: bool | None = None,
    native_shape_bucketing: bool = False,
) -> DMD2Method:
    method = object.__new__(DMD2Method)
    config: dict = {
        "rollout_mode": rollout_mode,
        "rollout_carry": carry,
        "dmd_denoising_steps": list(grid or _GRID),
        "generator_update_interval": interval,
    }
    if carry:
        config["rollout_carry_slots"] = slots
        if sample_type is not None:
            config["rollout_sample_type"] = sample_type
    if data_forcing is not None:
        config["rollout_data_forcing"] = data_forcing
    object.__setattr__(method, "method_config", config)
    object.__setattr__(method, "student", student if student is not None else _CarryStudent())
    object.__setattr__(
        method,
        "training_config",
        SimpleNamespace(
            loop=SimpleNamespace(gradient_accumulation_steps=(slots if grad_accum is None else grad_accum)),
            distributed=SimpleNamespace(sp_size=1),
            data=SimpleNamespace(native_shape_bucketing=native_shape_bucketing),
        ),
    )
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(0))
    object.__setattr__(method, "_cfg_uncond", None)
    object.__setattr__(method, "_denoising_step_list", None)
    object.__setattr__(method, "_rollout_mode", method._parse_rollout_mode())
    object.__setattr__(method, "_rollout_carry_rank_world", lambda: (rank, world))
    knobs = method._parse_rollout_carry()
    object.__setattr__(method, "_rollout_carry", knobs[0])
    object.__setattr__(method, "_rollout_carry_slot_count", knobs[1])
    object.__setattr__(method, "_rollout_sample_type", knobs[2])
    method._init_rollout_carry_state()
    object.__setattr__(
        method,
        "_rollout_data_forcing",
        method._parse_rollout_data_forcing(),
    )
    return method


def _stub_losses(method: DMD2Method) -> list[torch.Tensor]:
    """Replace the loss paths with recorders; carry mechanics stay real."""
    critic_preds: list[torch.Tensor] = []
    object.__setattr__(method, "_dmd_loss", lambda pred, batch: (torch.zeros(()), {}))

    def _critic(batch, *, generator_pred_x0=None):
        critic_preds.append(generator_pred_x0)
        return torch.zeros(()), "critic-ctx", {}, {}

    object.__setattr__(method, "_critic_flow_matching_loss", _critic)
    return critic_preds


def _replay_ode_walk(state: torch.Tensor, grid: list[int], hops: int) -> torch.Tensor:
    """Reference walk under the fake student's flow: pred = 0.5 * state."""
    for rung in range(hops):
        sigma = grid[rung] / 1000.0
        pred = state * 0.5
        eps = (state - (1.0 - sigma) * pred) / sigma
        sigma_next = grid[rung + 1] / 1000.0
        state = (1.0 - sigma_next) * pred + sigma_next * eps
    return state


# ----------------------------------------------------------------------
# (1) Slot round-robin and rung progression 0 -> 1 -> 2 -> 3 -> clear
# ----------------------------------------------------------------------


def test_slot_round_robin_and_rung_progression_with_two_slots() -> None:
    method = _make_method(slots=2, sample_type="sde", interval=1)
    _stub_losses(method)
    student = method.student

    rungs = []
    forwards_per_call = []
    for call in range(10):
        before = len(student.predict_calls)
        _, _, metrics = method.single_train_step({"text_embedding": torch.ones(1, 4)}, iteration=call)
        rungs.append(metrics["rollout_step"])
        forwards_per_call.append(len(student.predict_calls) - before)

    # Offsets: slot 0 -> (0*2+0)%4 = 0, slot 1 -> (0*2+1)%4 = 1. Interleaved
    # round-robin walks: slot0 = 0,1,2,3,clear,0 and slot1 = 1,2,3,clear,0,1.
    assert rungs == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0, 1.0]
    # First-ever fill of each slot pre-walks the whole grid (len-1 forwards)
    # exactly once; every other call, including post-clear restarts, pays
    # exactly one generation forward.
    assert forwards_per_call == [4, 4, 1, 1, 1, 1, 1, 1, 1, 1]
    # Slot 1 cleared after rung 3 (call 5), restarted at rung 0 (call 7),
    # advanced to rung 1 (call 9) and carries rung 2; slot 0 cleared at
    # call 6, restarted at call 8, and carries rung 1.
    assert method._carry_slots[0] is not None and method._carry_slots[0]["rung"] == 1
    assert method._carry_slots[1] is not None and method._carry_slots[1]["rung"] == 2


def test_carry_slots_cleared_after_last_rung() -> None:
    method = _make_method(slots=1, grid=[999, 500], sample_type="ode", interval=1)
    _stub_losses(method)
    student = method.student

    counts, rungs = [], []
    for call in range(3):
        before = len(student.predict_calls)
        _, _, metrics = method.single_train_step({"x": torch.ones(1)}, iteration=call)
        rungs.append(metrics["rollout_step"])
        counts.append(len(student.predict_calls) - before)

    # offset (0*1+0)%2 = 0: pre-walk (1 hop) then rung 0; rung 1 finishes the
    # trajectory; the restart begins at rung 0 with NO stagger pre-walk.
    assert rungs == [0.0, 1.0, 0.0]
    assert counts == [2, 1, 1]
    assert method._carry_slots[0] is not None
    assert method._carry_slots[0]["rung"] == 1


# ----------------------------------------------------------------------
# (2) Staggered starts across (rank, slot) streams
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "slot", "expected_offset"),
    [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3), (2, 0, 0), (2, 1, 1)],
)
def test_stagger_offsets_follow_rank_slot_formula(rank: int, slot: int, expected_offset: int) -> None:
    method = _make_method(slots=2, sample_type="ode", rank=rank, world=3)
    student = method.student
    step_list = method._get_denoising_step_list(torch.device("cpu"))
    state = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    batch = SimpleNamespace(dmd_latent_vis_dict={})

    snapshot, rung = method._staggered_start(state, batch, step_list, slot)

    assert rung == expected_offset == (rank * 2 + slot) % len(_GRID)
    # Every rank walks the whole grid regardless of its offset (uniform FSDP
    # collective count) and only keeps the snapshot at its own rung.
    assert len(student.predict_calls) == len(_GRID) - 1
    assert all(not call["grad_enabled"] for call in student.predict_calls)
    assert all(call["attn_kind"] == "vsa" for call in student.predict_calls)
    torch.testing.assert_close(snapshot, _replay_ode_walk(state, _GRID, expected_offset))


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("slot", [0, 1])
def test_native_shape_stagger_is_rank_synchronous(rank: int, slot: int) -> None:
    method = _make_method(
        slots=2,
        sample_type="ode",
        rank=rank,
        world=8,
        native_shape_bucketing=True,
    )
    step_list = method._get_denoising_step_list(torch.device("cpu"))
    state = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    batch = SimpleNamespace(dmd_latent_vis_dict={})

    _, rung = method._staggered_start(state, batch, step_list, slot)

    assert rung == slot


# ----------------------------------------------------------------------
# (3) ODE renoise analytic identity with per-modality shifts (real H3 math)
# ----------------------------------------------------------------------


def _h3_adapter() -> MiniMaxH3DMDModel:
    model = MiniMaxH3DMDModel.__new__(MiniMaxH3DMDModel)
    model.training_config = SimpleNamespace(data=SimpleNamespace(
        num_latent_t=2,
        num_frames=5,
        num_height=64,
        num_width=64,
    ))
    return model


def test_ode_renoise_analytic_identity_per_modality() -> None:
    """From x_t = (1-s_m)x0 + s_m*eps and pred = x0, the advanced state is
    (1-s'_m)x0 + s'_m*eps for both the video-shift and audio-shift slices."""
    model = _h3_adapter()
    slices = dict(model.modality_slices())
    total = slices["audio"].stop
    generator = torch.Generator().manual_seed(11)
    x0 = torch.randn(1, total, generator=generator)
    eps = torch.randn(1, total, generator=generator)
    timestep = torch.tensor([500], dtype=torch.long)
    next_timestep = torch.tensor([250], dtype=torch.long)

    x_t = model.add_noise(x0, eps, timestep)
    implied = model.extract_eps(x_t, x0, timestep)
    torch.testing.assert_close(implied, eps, rtol=1e-5, atol=1e-5)

    advanced = model.add_noise(x0, implied, next_timestep)
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        sigma_next = float(shift_noise_amount(torch.tensor([0.25]), shift))
        expected = (1.0 - sigma_next) * x0[:, slices[name]] + sigma_next * eps[:, slices[name]]
        torch.testing.assert_close(advanced[:, slices[name]], expected, rtol=1e-5, atol=1e-5)


def test_method_renoise_ode_uses_adapter_extract_eps() -> None:
    model = _h3_adapter()
    method = _make_method(slots=1, grid=[500, 250], sample_type="ode", student=model)
    total = dict(model.modality_slices())["audio"].stop
    generator = torch.Generator().manual_seed(13)
    x0 = torch.randn(1, total, generator=generator)
    eps = torch.randn(1, total, generator=generator)
    timestep = torch.tensor([500], dtype=torch.long)
    x_t = model.add_noise(x0, eps, timestep)

    step_list = method._get_denoising_step_list(torch.device("cpu"))
    advanced = method._renoise(x_t, x0, timestep, 1, step_list)
    expected = model.add_noise(x0, eps, torch.tensor([250], dtype=torch.long))
    torch.testing.assert_close(advanced, expected, rtol=1e-5, atol=1e-5)


def test_method_renoise_sde_draws_fresh_noise() -> None:
    method = _make_method(slots=1, grid=[999, 500], sample_type="sde")
    step_list = method._get_denoising_step_list(torch.device("cpu"))
    state = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(5))
    pred = state * 0.5

    advanced = method._renoise(state, pred, torch.tensor([999]), 1, step_list)
    noise = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(advanced, 0.5 * pred + 0.5 * noise)


# ----------------------------------------------------------------------
# (4) Mid-walk calls reuse the carried conditioning
# ----------------------------------------------------------------------


def test_mid_walk_reuses_carried_conditioning_and_ignores_fresh_batches() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1)
    _stub_losses(method)
    student = method.student

    batch_a = {"text_embedding": torch.full((1, 4), 1.0), "info_list": ["prompt-a"]}
    batch_b = {"text_embedding": torch.full((1, 4), 2.0), "info_list": ["prompt-b"]}
    batch_c = {"text_embedding": torch.full((1, 4), 3.0), "info_list": ["prompt-c"]}

    method.single_train_step(batch_a, iteration=0)
    snapshot = student.prepare_calls[0]
    # The adopted batch is a detached device snapshot, not the loader dict.
    assert snapshot is not batch_a
    torch.testing.assert_close(snapshot["text_embedding"], batch_a["text_embedding"])
    assert snapshot["info_list"] == ["prompt-a"]

    # Rungs 1..3: the fresh loader batches are ignored, the trajectory keeps
    # the exact snapshot object it set out with.
    for call in range(1, 4):
        method.single_train_step(batch_b, iteration=call)
        assert student.prepare_calls[call] is snapshot
        torch.testing.assert_close(student.prepare_calls[call]["text_embedding"], batch_a["text_embedding"])

    # Rung 3 finished the walk; the next call adopts the new conditioning.
    method.single_train_step(batch_c, iteration=4)
    torch.testing.assert_close(student.prepare_calls[4]["text_embedding"], batch_c["text_embedding"])
    assert student.prepare_calls[4]["info_list"] == ["prompt-c"]


# ----------------------------------------------------------------------
# Phase wiring: one forward per call, critic consumes the carried pred
# ----------------------------------------------------------------------


def test_student_phase_forward_has_grad_and_sets_ctx_and_vis() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=5)
    _stub_losses(method)
    student = method.student

    _, outputs, metrics = method.single_train_step({"x": torch.ones(1)}, iteration=5)

    assert metrics["update_student"] == 1.0
    assert metrics["rollout_step"] == 0.0
    main = student.predict_calls[-1]
    assert main["grad_enabled"] is True
    assert main["attn_kind"] == "vsa"
    assert main["timestep"] == float(_GRID[0])
    student_ctx = outputs["_fv_backward"]["student_ctx"]
    assert student_ctx[0] is student.last_batch.timesteps
    assert student_ctx[1] == "vsa-metadata"
    assert outputs["_fv_backward"]["critic_ctx"] is None
    vis = student.last_batch.dmd_latent_vis_dict
    torch.testing.assert_close(vis["generator_timestep"], torch.tensor([float(_GRID[0])]))
    torch.testing.assert_close(vis["generator_pred_video"], student.last_pred)
    assert "generator_timestep" in method.latent_vis


def test_critic_phase_forward_is_no_grad_and_feeds_carried_pred() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=5)
    critic_preds = _stub_losses(method)
    student = method.student

    _, outputs, metrics = method.single_train_step({"x": torch.ones(1)}, iteration=1)

    assert metrics["update_student"] == 0.0
    assert metrics["rollout_step"] == 0.0
    main = student.predict_calls[-1]
    assert main["grad_enabled"] is False
    assert main["attn_kind"] == "vsa"
    # The critic is fit on the same simulated state the student trains on:
    # it receives this call's prediction rather than rolling its own.
    assert len(critic_preds) == 1
    assert critic_preds[0] is student.last_pred
    assert outputs["_fv_backward"]["critic_ctx"] == "critic-ctx"
    assert outputs["_fv_backward"]["student_ctx"] is None
    # Both phases advance the trajectory.
    assert method._carry_slots[0] is not None
    assert method._carry_slots[0]["rung"] == 1
    assert "generator_timestep" in student.last_batch.dmd_latent_vis_dict


def test_carried_advance_state_is_detached_and_matches_ode_math() -> None:
    method = _make_method(slots=1, sample_type="ode", interval=1)
    _stub_losses(method)

    method.single_train_step({"x": torch.ones(1)}, iteration=0)
    carried = method._carry_slots[0]
    assert carried is not None
    assert carried["rung"] == 1
    assert not carried["state"].requires_grad
    # offset 0: the main forward ran on the fresh-noise state drawn from the
    # seeded generator; replay one ODE hop with the fake student's flow.
    state0 = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(carried["state"], _replay_ode_walk(state0, _GRID, 1))


# ----------------------------------------------------------------------
# (6) Knob parsing and validation; default off preserves existing behavior
# ----------------------------------------------------------------------


def _parse_only(
    config: dict,
    *,
    grad_accum: int = 1,
    streams: tuple[int, int] = (0, 1),
    native_shape_bucketing: bool = False,
):
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", dict(config))
    object.__setattr__(method, "student", _CarryStudent())
    object.__setattr__(
        method,
        "training_config",
        SimpleNamespace(
            loop=SimpleNamespace(gradient_accumulation_steps=grad_accum),
            distributed=SimpleNamespace(sp_size=1),
            data=SimpleNamespace(native_shape_bucketing=native_shape_bucketing),
        ),
    )
    object.__setattr__(method, "_rollout_mode", method._parse_rollout_mode())
    object.__setattr__(method, "_rollout_carry_rank_world", lambda: streams)
    return method._parse_rollout_carry()


def test_rollout_carry_defaults_off() -> None:
    knobs = _parse_only({
        "rollout_mode": "simulate",
        "dmd_denoising_steps": _GRID,
    })
    assert knobs == (False, 0, "sde")


def test_rollout_carry_defaults_slots_to_grad_accum() -> None:
    knobs = _parse_only(
        {
            "rollout_mode": "simulate",
            "rollout_carry": True,
            "rollout_sample_type": "ode",
            "dmd_denoising_steps": _GRID,
            "generator_update_interval": 5,
        },
        grad_accum=3,
    )
    assert knobs == (True, 3, "ode")


def test_rollout_carry_requires_simulate_mode() -> None:
    with pytest.raises(ValueError, match="rollout_mode: simulate"):
        _parse_only({
            "rollout_mode": "data_latent",
            "rollout_carry": True,
            "dmd_denoising_steps": _GRID,
        })


def test_rollout_carry_slots_must_match_grad_accum() -> None:
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        _parse_only(
            {
                "rollout_mode": "simulate",
                "rollout_carry": True,
                "rollout_carry_slots": 4,
                "dmd_denoising_steps": _GRID,
            },
            grad_accum=2,
        )


def test_carry_knobs_require_rollout_carry_enabled() -> None:
    with pytest.raises(ValueError, match="rollout_carry: true"):
        _parse_only({
            "rollout_mode": "simulate",
            "rollout_carry_slots": 2,
            "dmd_denoising_steps": _GRID,
        })
    with pytest.raises(ValueError, match="rollout_carry: true"):
        _parse_only({
            "rollout_mode": "simulate",
            "rollout_sample_type": "ode",
            "dmd_denoising_steps": _GRID,
        })


def test_rollout_sample_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="ode, sde"):
        _parse_only({
            "rollout_mode": "simulate",
            "rollout_carry": True,
            "rollout_sample_type": "euler",
            "dmd_denoising_steps": _GRID,
        })


def test_ode_requires_student_extract_eps() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(
        method,
        "method_config",
        {
            "rollout_mode": "simulate",
            "rollout_carry": True,
            "rollout_sample_type": "ode",
            "dmd_denoising_steps": _GRID,
        },
    )
    object.__setattr__(method, "student", SimpleNamespace())
    object.__setattr__(
        method,
        "training_config",
        SimpleNamespace(
            loop=SimpleNamespace(gradient_accumulation_steps=1),
            distributed=SimpleNamespace(sp_size=1),
        ),
    )
    object.__setattr__(method, "_rollout_mode", "simulate")
    with pytest.raises(ValueError, match="extract_eps"):
        method._parse_rollout_carry()


def test_coverage_guard_rejects_uncovered_rung_phases() -> None:
    # gcd(4, 2) = 2 phase classes: one stream cannot cover both.
    with pytest.raises(ValueError, match="cannot cover"):
        DMD2Method._validate_rollout_carry_coverage(streams=1, grid_len=4, interval=2)
    DMD2Method._validate_rollout_carry_coverage(streams=2, grid_len=4, interval=2)
    # The H3 recipe (4-rung grid, interval 5) has gcd 1 and always passes.
    DMD2Method._validate_rollout_carry_coverage(streams=1, grid_len=4, interval=5)


def test_native_shape_coverage_does_not_count_rank_staggering() -> None:
    config = {
        "rollout_mode": "simulate",
        "rollout_carry": True,
        "rollout_carry_slots": 1,
        "rollout_sample_type": "ode",
        "dmd_denoising_steps": _GRID,
        "generator_update_interval": 2,
    }
    assert _parse_only(config, streams=(0, 2)) == (True, 1, "ode")
    with pytest.raises(ValueError, match="cannot cover"):
        _parse_only(config, streams=(0, 2), native_shape_bucketing=True)


# ----------------------------------------------------------------------
# Integration: full carried steps on the real H3 CPU trio
# ----------------------------------------------------------------------


def _build_carry_trio(monkeypatch: pytest.MonkeyPatch, *, interval: int) -> DMD2Method:
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


def test_full_carried_student_step_on_real_h3_trio(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end carried step: real prepare_batch, losses, and ODE advance."""
    from fastvideo.tests.train.methods.test_minimax_h3_dmd2 import _raw_batch

    method = _build_carry_trio(monkeypatch, interval=1)
    student = method.student

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(seed=1), iteration=0)

    assert metrics["update_student"] == 1.0
    assert metrics["rollout_step"] == 0.0  # (rank 0 * 1 + slot 0) % 3
    assert torch.isfinite(loss_map["total_loss"])
    assert loss_map["generator_loss"].item() > 0.0
    assert "generator_loss_video" in metrics and "generator_loss_audio" in metrics
    carried = method._carry_slots[0]
    assert carried is not None and carried["rung"] == 1
    assert carried["state"].shape == student.prepare_batch(
        carried["raw_batch"],
        generator=torch.Generator().manual_seed(0),
        latents_source="zeros",
    ).latents.shape
    assert not carried["state"].requires_grad

    method.backward(loss_map, outputs)
    assert student.transformer.scale.grad is not None
    assert torch.isfinite(student.transformer.scale.grad)

    # Mid-walk call: a new prompt arrives but the student must still be
    # conditioned on the trajectory's adopted text.
    adopted = _raw_batch(seed=1)
    adopted_text = adopted["text_embedding"][:, :2].to(torch.bfloat16)
    method.single_train_step(_raw_batch(seed=9), iteration=1)
    torch.testing.assert_close(student.transformer.last_encoder_hidden_states, adopted_text)


def test_full_carried_critic_step_on_real_h3_trio(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastvideo.tests.train.methods.test_minimax_h3_dmd2 import _raw_batch

    method = _build_carry_trio(monkeypatch, interval=5)
    critic = method.critic

    loss_map, outputs, metrics = method.single_train_step(_raw_batch(seed=1), iteration=1)

    assert metrics["update_student"] == 0.0
    assert metrics["rollout_step"] == 0.0
    assert loss_map["generator_loss"].item() == 0.0
    assert loss_map["fake_score_loss"].item() > 0.0
    assert "fake_score_loss_video" in metrics and "fake_score_loss_audio" in metrics
    assert method._carry_slots[0] is not None and method._carry_slots[0]["rung"] == 1

    method.backward(loss_map, outputs)
    assert critic.transformer.scale.grad is not None
    assert torch.isfinite(critic.transformer.scale.grad)
    assert method.student.transformer.scale.grad is None
