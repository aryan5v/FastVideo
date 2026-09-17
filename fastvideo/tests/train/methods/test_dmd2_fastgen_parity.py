# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for DMD2 behavior matched to the FastGen H3 recipe.

The references below intentionally spell out the small pieces of FastGen math
instead of importing a second checkout.  FastGen's rectified-flow schedule uses
float64 arithmetic over ``max_t=0.999`` and casts latent results back only once.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import MiniMaxH3DMDModel

_FASTGEN_MAX_T = 0.999
_TIMESTEP_SCALE = 1000.0


def _fastgen_time_shift(value: torch.Tensor, shift: float) -> torch.Tensor:
    """FastGen ``time_shift`` on the H3 schedule's finite time domain."""
    value = value.to(torch.float64)
    return value * shift * _FASTGEN_MAX_T / (value * (shift - 1.0) + _FASTGEN_MAX_T)


def _h3_adapter() -> MiniMaxH3DMDModel:
    model = MiniMaxH3DMDModel.__new__(MiniMaxH3DMDModel)
    model.training_config = SimpleNamespace(data=SimpleNamespace(
        num_latent_t=2,
        num_frames=5,
        num_height=64,
        num_width=64,
    ))
    return model


def _score_method() -> DMD2Method:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {
        "min_timestep_ratio": 0.001,
        "max_timestep_ratio": 0.999,
        "score_timestep_shift": 2.4,
        "score_timestep_warp_max": _FASTGEN_MAX_T,
        "score_timestep_continuous": True,
    })
    object.__setattr__(method, "student", SimpleNamespace(
        num_train_timesteps=1000,
        shift_and_clamp_timestep=lambda timestep: timestep,
    ))
    lo, hi = method._parse_score_timestep_bounds()
    object.__setattr__(method, "_score_min_timestep", lo)
    object.__setattr__(method, "_score_max_timestep", hi)
    object.__setattr__(method, "_score_timestep_shift", 2.4)
    object.__setattr__(method, "_score_timestep_warp_max", _FASTGEN_MAX_T)
    object.__setattr__(method, "_score_timestep_continuous", True)
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(0))
    return method


@pytest.mark.parametrize("unit_draw", [0.0, 0.37, 1.0])
def test_shifted_score_sampler_draws_uniform_coordinate_before_inverse_warp(
    monkeypatch: pytest.MonkeyPatch,
    unit_draw: float,
) -> None:
    """The random draw is uniform on FastGen's pre-shift ``[.001, .999]``.

    H3 represents the score time on an unshifted base clock.  Applying the
    inverse 2.4 warp there makes its video shift compose to 5 and its audio
    shift compose to 1.25, exactly matching FastGen's video-clock schedule.
    """

    def fixed_rand(size, *, device=None, dtype=None, generator=None):
        del generator
        assert tuple(size) == (1, )
        return torch.full((1, ), unit_draw, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", fixed_rand)
    method = _score_method()
    sampled = method._sample_score_timestep(torch.device("cpu"))

    pre_shift = torch.tensor(
        [0.001 + unit_draw * (0.999 - 0.001)],
        dtype=torch.float64,
    )
    expected_base = _fastgen_time_shift(pre_shift, 1.0 / 2.4) * _TIMESTEP_SCALE
    assert sampled.dtype == torch.float64
    torch.testing.assert_close(sampled, expected_base, rtol=0.0, atol=1e-10)

    sigma_video, sigma_audio = _h3_adapter()._noise_amounts(sampled)
    torch.testing.assert_close(
        sigma_video,
        _fastgen_time_shift(pre_shift, 5.0),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        sigma_audio,
        _fastgen_time_shift(pre_shift, 1.25),
        rtol=0.0,
        atol=1e-12,
    )


def _packed_bfloat16_pair(model: MiniMaxH3DMDModel) -> tuple[torch.Tensor, torch.Tensor]:
    slices = dict(model.modality_slices())
    total = slices["audio"].stop
    clean = torch.linspace(-2.75, 3.125, total, dtype=torch.float32).reshape(1, -1).to(torch.bfloat16)
    noise = torch.linspace(1.875, -3.5, total, dtype=torch.float32).reshape(1, -1).to(torch.bfloat16)
    return clean, noise


def _reference_sigmas(timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base = (timestep.reshape(-1)[:1].to(torch.float64) / _TIMESTEP_SCALE).clamp(0.0, _FASTGEN_MAX_T)
    return _fastgen_time_shift(base, 12.0), _fastgen_time_shift(base, 3.0)


def _reference_mix(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return ((1.0 - sigma) * clean.to(torch.float64) + sigma * noise.to(torch.float64)).to(clean.dtype)


def _reference_unmix(noisy: torch.Tensor, clean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return ((noisy.to(torch.float64) - (1.0 - sigma) * clean.to(torch.float64)) / sigma.clamp_min(1e-6)).to(
        noisy.dtype)


def test_h3_add_noise_matches_fastgen_fp64_then_cast_for_bfloat16() -> None:
    model = _h3_adapter()
    clean, noise = _packed_bfloat16_pair(model)
    timestep = torch.tensor([413.375], dtype=torch.float64)
    sigma_video, sigma_audio = _reference_sigmas(timestep)
    clean_video, clean_audio = model.unpack_latents(clean)
    noise_video, noise_audio = model.unpack_latents(noise)
    expected = model.pack_latents(
        _reference_mix(clean_video, noise_video, sigma_video),
        _reference_mix(clean_audio, noise_audio, sigma_audio),
    )

    actual = model.add_noise(clean, noise, timestep)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_h3_extract_eps_matches_fastgen_fp64_then_cast_at_low_audio_sigma() -> None:
    model = _h3_adapter()
    clean, noisy = _packed_bfloat16_pair(model)
    timestep = _fastgen_time_shift(torch.tensor([0.001], dtype=torch.float64), 1.0 / 2.4) * _TIMESTEP_SCALE
    sigma_video, sigma_audio = _reference_sigmas(timestep)
    noisy_video, noisy_audio = model.unpack_latents(noisy)
    clean_video, clean_audio = model.unpack_latents(clean)
    expected = model.pack_latents(
        _reference_unmix(noisy_video, clean_video, sigma_video),
        _reference_unmix(noisy_audio, clean_audio, sigma_audio),
    )

    actual = model.extract_eps(noisy, clean, timestep)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_h3_predict_x0_helper_matches_fastgen_fp64_then_cast() -> None:
    model = _h3_adapter()
    noisy, pred_noise = _packed_bfloat16_pair(model)
    sigma = torch.tensor([0.3141592653589793], dtype=torch.float64)
    expected = (noisy.to(torch.float64) - sigma * pred_noise.to(torch.float64)).to(torch.bfloat16)

    actual = model._to_x0(noisy, pred_noise, sigma)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class _ScoreStudent:

    def modality_slices(self):
        return (("video", slice(0, 3)), ("audio", slice(3, 5)))

    def add_noise(self, clean, noise, timestep):
        del timestep
        return 0.75 * clean + 0.25 * noise


class _DirectX0Critic:

    def __init__(self) -> None:
        self.value = torch.nn.Parameter(torch.tensor(0.25))
        self.predict_x0_calls = 0

    def predict_x0(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        del timestep, batch, conditional, cfg_uncond, attn_kind
        self.predict_x0_calls += 1
        return self.value * torch.ones_like(noisy)

    def predict_noise(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("global x0 regression must call critic.predict_x0 directly")


def test_critic_x0_objective_is_direct_predict_x0_mse_per_modality() -> None:
    method = object.__new__(DMD2Method)
    critic = _DirectX0Critic()
    object.__setattr__(method, "method_config", {"fake_score_loss_space": "x0"})
    object.__setattr__(method, "student", _ScoreStudent())
    object.__setattr__(method, "critic", critic)
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(11))
    object.__setattr__(method, "_cfg_uncond", None)
    object.__setattr__(method, "_fake_score_loss_space", {"__default__": "x0"})
    object.__setattr__(method, "_sample_score_timestep", lambda device: torch.tensor([500.0], device=device))
    batch = SimpleNamespace(timesteps=None, attn_metadata=None, fake_score_latent_vis_dict=None)
    generated_x0 = torch.tensor([[1.0, -2.0, 4.0, 8.0, -16.0]])

    loss, _, _, metrics = method._critic_flow_matching_loss(batch, generator_pred_x0=generated_x0)

    pred_x0 = torch.full_like(generated_x0, 0.25)
    expected_video = torch.mean((pred_x0[:, :3] - generated_x0[:, :3])**2)
    expected_audio = torch.mean((pred_x0[:, 3:] - generated_x0[:, 3:])**2)
    assert critic.predict_x0_calls == 1
    torch.testing.assert_close(metrics["fake_score_loss_video"], expected_video)
    torch.testing.assert_close(metrics["fake_score_loss_audio"], expected_audio)
    torch.testing.assert_close(loss, expected_video + expected_audio)

    loss.backward()
    assert critic.value.grad is not None
    assert torch.isfinite(critic.value.grad)
