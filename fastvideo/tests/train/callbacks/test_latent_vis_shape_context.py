# SPDX-License-Identifier: Apache-2.0
"""Latent visualization keeps native packed-shape context."""

from types import SimpleNamespace

import torch

from fastvideo.train.callbacks.latent_vis import LatentVisCallback


def test_latent_vis_passes_batch_local_layout_to_decoder(monkeypatch) -> None:
    layout = object()
    latent = torch.ones(1, 9)
    decoded: list[tuple[torch.Tensor, object]] = []

    class Student:

        @staticmethod
        def decode_vis_latents(value, *, layout):
            decoded.append((value, layout))
            return torch.zeros(1, 1, 3, 2, 2, dtype=torch.uint8).numpy()

    class Tracker:

        def video(self, clip, *, fps, format):
            assert fps == 24
            assert format == "mp4"
            return clip

        def log_artifacts(self, artifacts, iteration):
            assert set(artifacts) == {"latent_vis/generator_pred_video"}
            assert iteration == 8

    monkeypatch.setattr(
        "fastvideo.train.callbacks.latent_vis.get_world_group",
        lambda: SimpleNamespace(rank=0),
    )
    callback = LatentVisCallback(every_steps=1, keys=["generator_pred_video"])
    callback.tracker = Tracker()
    method = SimpleNamespace(
        student=Student(),
        latent_vis={
            "generator_pred_video": latent,
            "_fv_latent_layout": layout,
        },
    )

    callback.on_training_step_end(method, {}, iteration=8)

    assert decoded == [(latent, layout)]
