# SPDX-License-Identifier: Apache-2.0
"""FastH3 Preview velocity distillation for native hybrid H3 attention."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.models.base import ModelBase


class MiniMaxH3VelocityKDMethod(FineTuneMethod):
    """Match a hybrid student to a frozen VSA teacher on identical noisy AV inputs."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "teacher" not in role_models:
            raise ValueError("MiniMaxH3VelocityKDMethod requires role 'teacher'")
        self.teacher = role_models["teacher"]
        if self.teacher._trainable:
            raise ValueError("MiniMaxH3VelocityKDMethod requires a frozen teacher")
        if not getattr(self.student.transformer, "hybrid_attention_enabled", False):
            raise ValueError("MiniMaxH3VelocityKDMethod requires a hybrid-attention student")
        if getattr(self.teacher.transformer, "hybrid_attention_enabled", False):
            raise ValueError("MiniMaxH3VelocityKDMethod requires a non-hybrid VSA teacher")

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        """Run frozen Preview and hybrid student at the same FastH3 grid point."""
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        required = {
            "latents": training_batch.latents,
            "noisy_model_input": training_batch.noisy_model_input,
            "timesteps": training_batch.timesteps,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError("prepare_batch() must set " + ", ".join(missing))

        noisy_latents = training_batch.noisy_model_input.permute(0, 2, 1, 3, 4)
        timesteps = training_batch.timesteps
        with torch.no_grad():
            teacher_prediction = self.teacher.predict_noise(
                noisy_latents,
                timesteps,
                training_batch,
                conditional=True,
                attn_kind="vsa",
            )
        student_prediction = self.student.predict_noise(
            noisy_latents,
            timesteps,
            training_batch,
            conditional=True,
            attn_kind="dense",
        )
        if not isinstance(teacher_prediction, tuple) or not isinstance(student_prediction, tuple):
            raise TypeError("MiniMax H3 velocity KD requires joint video/audio predictions")

        video_velocity_mse = F.mse_loss(student_prediction[0].float(), teacher_prediction[0].float())
        audio_velocity_mse = F.mse_loss(student_prediction[1].float(), teacher_prediction[1].float())
        total_loss = video_velocity_mse + audio_velocity_mse
        loss_map = {
            "total_loss": total_loss,
            "velocity_kd_loss": total_loss,
            "video_velocity_mse": video_velocity_mse,
            "audio_velocity_mse": audio_velocity_mse,
        }
        outputs = {
            "_fv_backward": (
                int(training_batch.current_timestep),
                training_batch.attn_metadata,
            )
        }
        metrics: dict[str, LogScalar] = {"sigma_grid_index": int(training_batch.current_timestep)}
        return loss_map, outputs, metrics


__all__ = ["MiniMaxH3VelocityKDMethod"]
