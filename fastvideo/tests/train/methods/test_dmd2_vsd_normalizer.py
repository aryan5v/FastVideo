# SPDX-License-Identifier: Apache-2.0
"""The VSD normalizer must stay finite when |gen - real| degenerates to 0."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method


class _EchoTeacher:
    """Returns the generator's own prediction: |gen - real| == 0 exactly."""

    def __init__(self, gen: torch.Tensor):
        self._gen = gen

    def predict_x0(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        return self._gen.detach().clone()


class _OffsetCritic(_EchoTeacher):

    def predict_x0(self, noisy, timestep, batch, *, conditional, cfg_uncond, attn_kind):
        return self._gen.detach().clone() + 1.0


def test_degenerate_denominator_yields_finite_loss_and_grad() -> None:
    gen = torch.randn(1, 16, generator=torch.Generator().manual_seed(3)).requires_grad_(True)

    method = object.__new__(DMD2Method)
    object.__setattr__(method, "method_config", {})
    object.__setattr__(method, "cuda_generator", torch.Generator().manual_seed(0))
    object.__setattr__(method, "_cfg_uncond", None)
    object.__setattr__(method, "student", SimpleNamespace(add_noise=lambda clean, noise, t: clean))
    object.__setattr__(method, "teacher", _EchoTeacher(gen))
    object.__setattr__(method, "critic", _OffsetCritic(gen))
    object.__setattr__(method, "_sample_score_timestep", lambda device: torch.tensor([500]))

    batch = SimpleNamespace(dmd_latent_vis_dict={})
    loss, metrics = method._dmd_loss(gen, batch)

    assert torch.isfinite(loss)
    assert metrics == {}
    loss.backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()
