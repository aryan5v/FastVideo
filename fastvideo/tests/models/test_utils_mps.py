# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.models.utils import pred_noise_to_pred_video, pred_noise_to_x_bound


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_pred_noise_to_pred_video_uses_fp32_on_mps() -> None:
    scheduler = SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.5], dtype=torch.float64),
        timesteps=torch.tensor([1000.0, 500.0], dtype=torch.float64),
    )
    pred_noise = torch.ones(1, 1, 1, 1, device="mps", dtype=torch.float16)
    latent = torch.full_like(pred_noise, 2.0)
    timestep = torch.tensor([500], device="mps")

    output = pred_noise_to_pred_video(pred_noise, latent, timestep, scheduler)

    assert output.device.type == "mps"
    assert output.dtype == torch.float16
    assert torch.equal(output.cpu(), torch.full((1, 1, 1, 1), 1.5, dtype=torch.float16))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_pred_noise_to_x_bound_uses_cpu_timestep_lookup_on_mps() -> None:
    scheduler = SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64),
        timesteps=torch.tensor([1000.0, 500.0, 0.0], dtype=torch.float64),
    )
    pred_noise = torch.ones(1, 1, 1, 1, device="mps", dtype=torch.float16)
    latent = torch.full_like(pred_noise, 2.0)

    output = pred_noise_to_x_bound(
        pred_noise,
        latent,
        torch.tensor([1000], device="mps"),
        torch.tensor([500], device="mps"),
        scheduler,
    )

    assert output.device.type == "mps"
    assert torch.equal(output.cpu(), torch.full((1, 1, 1, 1), 1.5, dtype=torch.float16))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_flow_match_add_noise_uses_cpu_timestep_lookup_on_mps() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    clean_latent = torch.zeros(2, 1, 1, 1, device="mps", dtype=torch.float16)
    noise = torch.ones_like(clean_latent)

    output = scheduler.add_noise(
        clean_latent,
        noise,
        torch.tensor([757], device="mps", dtype=torch.long),
    )

    assert output.device.type == "mps"
    assert torch.isfinite(output).all().cpu()
