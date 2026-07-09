# SPDX-License-Identifier: Apache-2.0
"""Wan VAE / TAEHV decode tests — MLX TAEHV parity + denormalize helpers.

- TAEHV MLX vs torch is bit-close (atol 1e-5) for z_dim=16 and 48.
- Full AutoencoderKLWan mean/std denormalize is checked against the torch path
  used in ``mlx_wan_prompt_to_video`` (Metal-gated wall-clock optional).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX required for Wan VAE/TAEHV tests")

from fastvideo.mlx_runtime.wan_vae import (  # noqa: E402
    WanVAEConfigView,
    decode_latents_taehv_mlx,
    denormalize_latents_np,
    ensure_taehv_checkpoint,
)

_HAS_METAL = bool(getattr(mx, "metal", None) and mx.metal.is_available())


@pytest.mark.parametrize("z_dim", [16, 48])
def test_taehv_mlx_matches_torch(z_dim: int) -> None:
    import torch
    from fastvideo.third_party.taehv import TAEHV

    env_name = "FASTVIDEO_TAEW2_1_CHECKPOINT" if z_dim == 16 else "FASTVIDEO_TAEW2_2_CHECKPOINT"
    checkpoint = os.environ.get(env_name)
    if not checkpoint:
        pytest.skip(f"{env_name} is not staged; this future-only test never downloads decoder artifacts")
    ckpt = ensure_taehv_checkpoint(z_dim=z_dim, checkpoint_path=Path(checkpoint))
    rng = np.random.default_rng(0)
    h = w = 8
    t = 5
    lat = (rng.standard_normal((1, z_dim, t, h, w)) * 0.5).astype(np.float32)

    model = TAEHV(str(ckpt)).eval()
    with torch.no_grad():
        out_t = model.decode_video(torch.from_numpy(lat).transpose(1, 2), parallel=True, show_progress_bar=False)
    torch_np = out_t[0].permute(0, 2, 3, 1).float().numpy()
    mlx_np = decode_latents_taehv_mlx(lat, z_dim=z_dim)[0]
    tmin = min(torch_np.shape[0], mlx_np.shape[0])
    np.testing.assert_allclose(mlx_np[:tmin], torch_np[:tmin], atol=1e-5, rtol=1e-5)


def test_denormalize_matches_prompt_to_video_formula() -> None:
    """``latents / (1/std) + mean`` used by the hybrid script."""
    cfg = WanVAEConfigView(
        z_dim=2,
        latents_mean=(0.1, -0.2),
        latents_std=(2.0, 0.5),
    )
    lat = np.ones((1, 2, 1, 1, 1), dtype=np.float32)
    out = denormalize_latents_np(lat, cfg)
    # z * std + mean
    np.testing.assert_allclose(out[0, 0, 0, 0, 0], 1.0 * 2.0 + 0.1)
    np.testing.assert_allclose(out[0, 1, 0, 0, 0], 1.0 * 0.5 + (-0.2))


def test_full_vae_config_and_denormalize_formula() -> None:
    """Config load + denormalize formula vs the hybrid script's tensor math (CPU)."""
    import torch

    vae = os.environ.get("FASTVIDEO_WAN22_5B_VAE_DIR")
    if not vae:
        pytest.skip("FASTVIDEO_WAN22_5B_VAE_DIR is not staged; source/revision/checksum remain blocked in PORT_STATUS")
    vae_dir = Path(vae)
    cfg = WanVAEConfigView.from_vae_dir(vae_dir)
    assert cfg.z_dim == 48
    rng = np.random.default_rng(1)
    lat = (rng.standard_normal((1, 48, 3, 8, 8)) * 0.3).astype(np.float32)
    lat_dn = denormalize_latents_np(lat, cfg)
    mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(cfg.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1)
    lat_t = torch.from_numpy(lat)
    ref = (lat_t / inv_std + mean).numpy()
    np.testing.assert_allclose(lat_dn, ref, atol=1e-5, rtol=1e-5)
