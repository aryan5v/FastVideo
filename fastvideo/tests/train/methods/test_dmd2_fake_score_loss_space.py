# SPDX-License-Identifier: Apache-2.0
"""fake_score_loss_space=x0 must equal sigma_m^2-weighted velocity MSE."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method

_SPLIT = 24
_TOTAL = 32
_SIGMA = {"video": 0.8, "audio": 0.5}
_TIMESTEP = torch.tensor([500])


class _Student:

    def modality_slices(self):
        return (("video", slice(0, _SPLIT)), ("audio", slice(_SPLIT, _TOTAL)))

    def add_noise(self, clean, noise, timestep):
        out = clean.clone()
        for name, sl in self.modality_slices():
            sigma = _SIGMA[name]
            out[:, sl] = (1.0 - sigma) * clean[:, sl] + sigma * noise[:, sl]
        return out


class _Critic:

    def __init__(self):
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def predict_noise(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        return self.scale * torch.ones_like(noisy)


def _method(space: str | None, critic: _Critic) -> DMD2Method:
    method = object.__new__(DMD2Method)
    config = {} if space is None else {"fake_score_loss_space": space}
    object.__setattr__(method, "method_config", config)
    object.__setattr__(method, "student", _Student())
    object.__setattr__(method, "critic", critic)
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(0))
    object.__setattr__(method, "_cfg_uncond", None)
    object.__setattr__(method, "_fake_score_loss_space", method._parse_fake_score_loss_space())
    object.__setattr__(method, "_sample_score_timestep", lambda device: _TIMESTEP)
    return method


def _run(space: str | None, critic: _Critic) -> torch.Tensor:
    method = _method(space, critic)
    batch = SimpleNamespace(timesteps=None, attn_metadata=None, fake_score_latent_vis_dict=None)
    gen = torch.randn(1, _TOTAL, generator=torch.Generator().manual_seed(7))
    loss, _, _, metrics = method._critic_flow_matching_loss(batch, generator_pred_x0=gen)
    assert set(metrics) == {"fake_score_loss_video", "fake_score_loss_audio"}
    return loss


def _expected(space: str) -> float:
    gen = torch.randn(1, _TOTAL, generator=torch.Generator().manual_seed(7))
    noise = torch.randn(gen.shape, generator=torch.Generator().manual_seed(0))
    target = noise - gen
    pred = 0.1 * torch.ones_like(gen)
    expected = 0.0
    for name, sl in _Student().modality_slices():
        mse = torch.mean((pred[:, sl] - target[:, sl])**2).item()
        weight = _SIGMA[name]**2 if space == "x0" else 1.0
        expected += weight * mse
    return expected


def test_default_is_velocity_space() -> None:
    loss = _run(None, _Critic())
    assert loss.item() == pytest.approx(_expected("velocity"), rel=1e-5)


def test_x0_space_weights_by_sigma_squared() -> None:
    loss = _run("x0", _Critic())
    assert loss.item() == pytest.approx(_expected("x0"), rel=1e-4)


def test_x0_space_keeps_critic_gradient() -> None:
    critic = _Critic()
    loss = _run("x0", critic)
    loss.backward()
    assert critic.scale.grad is not None
    assert critic.scale.grad.abs().item() > 0.0


def test_invalid_loss_space_rejected() -> None:
    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {"fake_score_loss_space": "eps"})
    with pytest.raises(ValueError, match="velocity, x0"):
        method._parse_fake_score_loss_space()
