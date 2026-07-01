# SPDX-License-Identifier: Apache-2.0
import torch

from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vaes.wanvae import AutoencoderKLWan


def test_wan_tiled_decode_keeps_requested_frame_count(monkeypatch) -> None:
    vae = AutoencoderKLWan.__new__(AutoencoderKLWan)
    object.__setattr__(vae, "blend_num_frames", 8)
    decoded = torch.zeros(1, 3, 33, 1, 1)

    monkeypatch.setattr(ParallelTiledVAE, "tiled_decode", lambda _vae, _z: decoded)

    output = AutoencoderKLWan.tiled_decode(vae, torch.zeros(1, 16, 9, 1, 1))

    assert output.shape[2] == 33
    assert vae.blend_num_frames == 8
