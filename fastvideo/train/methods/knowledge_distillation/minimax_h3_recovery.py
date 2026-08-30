# SPDX-License-Identifier: Apache-2.0
"""Four-call MiniMax-H3 depth-pruning recovery.

The trainable student sees the ordinary joint T2VA flow target and the frozen
50-block dense teacher's velocity on exactly the same noisy video/audio state.
Selected student blocks also match compact statistics from their corresponding
source-teacher blocks. Every video and audio term is normalized independently
so the larger visual tensor cannot hide an audio regression.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


def _normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw MSE, target energy, and energy-normalized MSE."""
    if energy_floor <= 0.0:
        raise ValueError("energy_floor must be > 0")
    raw = F.mse_loss(prediction.float(), target.float())
    energy = target.float().square().mean().detach().clamp_min(energy_floor)
    return raw, energy, raw / energy


def _hidden_summary(hidden: torch.Tensor) -> torch.Tensor:
    """Compress one SP-local packed state without retaining all token rows."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden state must have shape [batch, rows, width], got {tuple(hidden.shape)}")
    mean = hidden.float().mean(dim=1)
    root_mean_square = hidden.float().square().mean(dim=1).clamp_min(1.0e-12).sqrt()
    return torch.cat((mean, root_mean_square), dim=-1)


def _transformer_blocks(model: MiniMaxH3Model) -> Sequence[torch.nn.Module]:
    blocks = getattr(model.transformer, "transformer_blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError("MiniMax H3 transformer must expose transformer_blocks as ModuleList")
    return blocks


def _block_map(model: MiniMaxH3Model) -> tuple[int, ...]:
    blocks = _transformer_blocks(model)
    config = getattr(model.transformer, "config", None)
    arch = getattr(config, "arch_config", config)
    raw = getattr(arch, "block_map", None)
    if raw is None:
        return tuple(range(len(blocks)))
    values = tuple(int(value) for value in raw)
    if len(values) != len(blocks):
        raise ValueError("MiniMax H3 block_map length does not match transformer_blocks")
    return values


@contextmanager
def _capture_hidden_summaries(
    model: MiniMaxH3Model,
    indices: Sequence[int],
) -> Generator[dict[int, torch.Tensor], None, None]:
    """Capture compact differentiable summaries from selected block outputs."""
    blocks = _transformer_blocks(model)
    captures: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(index: int) -> None:
        if not 0 <= index < len(blocks):
            raise ValueError(f"hidden feature block {index} is outside [0, {len(blocks)})")

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"block {index} produced {type(value).__name__}, expected Tensor")
            captures[index] = _hidden_summary(value)

        handles.append(blocks[index].register_forward_hook(hook))

    for block_index in indices:
        register(int(block_index))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


class MiniMaxH3RecoveryMethod(TrainingMethod):
    """Recover a depth-pruned four-call H3 student from a frozen dense teacher."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if set(role_models) != {"student", "teacher"}:
            raise ValueError("MiniMaxH3RecoveryMethod requires exactly student and teacher roles")
        if not isinstance(self.student, MiniMaxH3Model):
            raise TypeError("MiniMaxH3RecoveryMethod requires a MiniMaxH3Model student")
        teacher = role_models["teacher"]
        if not isinstance(teacher, MiniMaxH3Model):
            raise TypeError("MiniMaxH3RecoveryMethod requires a MiniMaxH3Model teacher")
        if not self.student._trainable or teacher._trainable:
            raise ValueError("recovery requires trainable student and frozen teacher")
        if teacher.attention_backend_name != "TORCH_SDPA":
            raise ValueError("MiniMax H3 recovery teacher must use exact dense TORCH_SDPA attention")
        self.teacher = teacher
        self._student_attn_kind: Literal["dense", "vsa"] = self._infer_attn_kind()

        mcfg = self.method_config
        self._energy_floor = float(mcfg.get("modality_energy_floor", 1.0e-3))
        self._denoising_weight = float(mcfg.get("denoising_weight", 1.0))
        self._teacher_velocity_weight = float(mcfg.get("teacher_velocity_weight", 1.0))
        self._feature_weight = float(mcfg.get("feature_weight", 0.01))
        for name, value in (
            ("denoising_weight", self._denoising_weight),
            ("teacher_velocity_weight", self._teacher_velocity_weight),
            ("feature_weight", self._feature_weight),
        ):
            if value < 0.0:
                raise ValueError(f"method.{name} must be non-negative")
        if self._energy_floor <= 0.0:
            raise ValueError("method.modality_energy_floor must be > 0")

        student_map = _block_map(self.student)
        raw_feature_indices = mcfg.get("feature_local_block_indices", [4, 9, 14, 19])
        if not isinstance(raw_feature_indices, list) or not raw_feature_indices:
            raise ValueError("method.feature_local_block_indices must be a non-empty list")
        self._student_feature_indices = tuple(int(index) for index in raw_feature_indices)
        if len(set(self._student_feature_indices)) != len(self._student_feature_indices):
            raise ValueError("method.feature_local_block_indices must be unique")
        try:
            self._teacher_feature_indices = tuple(student_map[index] for index in self._student_feature_indices)
        except IndexError as error:
            raise ValueError("feature_local_block_indices references a missing student block") from error
        teacher_blocks = _transformer_blocks(self.teacher)
        if any(index >= len(teacher_blocks) for index in self._teacher_feature_indices):
            raise ValueError("student block_map references a missing teacher block")

        self.student.init_preprocessors(self.training_config)
        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def on_train_start(self) -> None:
        super().on_train_start()
        self.teacher.on_train_start()

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config
        learning_rate = float(tc.optimizer.learning_rate)
        if learning_rate <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0")
        parameters = [parameter for parameter in self.student.transformer.parameters() if parameter.requires_grad]
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=parameters,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=learning_rate,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        required = {
            "latents": training_batch.latents,
            "audio_latents": training_batch.audio_latents,
            "noisy_model_input": training_batch.noisy_model_input,
            "audio_noisy_model_input": training_batch.audio_noisy_model_input,
            "noise": training_batch.noise,
            "audio_noise": training_batch.audio_noise,
            "timesteps": training_batch.timesteps,
        }
        missing = [name for name, value in required.items() if not isinstance(value, torch.Tensor)]
        if missing:
            raise RuntimeError("MiniMax H3 recovery batch is missing " + ", ".join(missing))

        clean_video = required["latents"]
        clean_audio = required["audio_latents"]
        noisy_video = required["noisy_model_input"].permute(0, 2, 1, 3, 4)
        video_target = required["noise"].permute(0, 2, 1, 3, 4) - clean_video
        audio_target = required["audio_noise"] - clean_audio
        assert isinstance(clean_video, torch.Tensor)
        assert isinstance(clean_audio, torch.Tensor)
        assert isinstance(noisy_video, torch.Tensor)
        assert isinstance(video_target, torch.Tensor)
        assert isinstance(audio_target, torch.Tensor)
        assert isinstance(training_batch.timesteps, torch.Tensor)

        with torch.no_grad(), _capture_hidden_summaries(self.teacher,
                                                        self._teacher_feature_indices) as teacher_features:
            teacher_prediction = self.teacher.predict_noise(
                noisy_video,
                training_batch.timesteps,
                training_batch,
                conditional=True,
                attn_kind="dense",
            )
        with _capture_hidden_summaries(self.student, self._student_feature_indices) as student_features:
            student_prediction = self.student.predict_noise(
                noisy_video,
                training_batch.timesteps,
                training_batch,
                conditional=True,
                attn_kind=self._student_attn_kind,
            )
        if not isinstance(teacher_prediction, tuple) or not isinstance(student_prediction, tuple):
            raise TypeError("MiniMax H3 recovery requires joint video/audio predictions")
        teacher_video, teacher_audio = (value.detach() for value in teacher_prediction)
        student_video, student_audio = student_prediction

        raw_video, video_energy, norm_video = _normalized_mse(
            student_video,
            video_target,
            energy_floor=self._energy_floor,
        )
        raw_audio, audio_energy, norm_audio = _normalized_mse(
            student_audio,
            audio_target,
            energy_floor=self._energy_floor,
        )
        raw_teacher_video, teacher_video_energy, norm_teacher_video = _normalized_mse(
            student_video,
            teacher_video,
            energy_floor=self._energy_floor,
        )
        raw_teacher_audio, teacher_audio_energy, norm_teacher_audio = _normalized_mse(
            student_audio,
            teacher_audio,
            energy_floor=self._energy_floor,
        )

        feature_terms: list[torch.Tensor] = []
        for student_index, teacher_index in zip(
                self._student_feature_indices,
                self._teacher_feature_indices,
                strict=True,
        ):
            if student_index not in student_features or teacher_index not in teacher_features:
                raise RuntimeError("selected hidden-state hook did not execute")
            _, _, normalized = _normalized_mse(
                student_features[student_index],
                teacher_features[teacher_index].detach(),
                energy_floor=self._energy_floor,
            )
            feature_terms.append(normalized)
        feature_loss = torch.stack(feature_terms).mean()

        denoising_loss = norm_video + norm_audio
        teacher_velocity_loss = norm_teacher_video + norm_teacher_audio
        total_loss = (self._denoising_weight * denoising_loss + self._teacher_velocity_weight * teacher_velocity_loss +
                      self._feature_weight * feature_loss)
        loss_map = {
            "total_loss": total_loss,
            "denoising_loss": denoising_loss,
            "teacher_velocity_loss": teacher_velocity_loss,
            "hidden_feature_loss": feature_loss,
            "raw_video_denoising_loss": raw_video,
            "raw_audio_denoising_loss": raw_audio,
            "normalized_video_denoising_loss": norm_video,
            "normalized_audio_denoising_loss": norm_audio,
            "raw_video_teacher_velocity_loss": raw_teacher_video,
            "raw_audio_teacher_velocity_loss": raw_teacher_audio,
            "normalized_video_teacher_velocity_loss": norm_teacher_video,
            "normalized_audio_teacher_velocity_loss": norm_teacher_audio,
            "video_target_energy": video_energy,
            "audio_target_energy": audio_energy,
            "teacher_video_energy": teacher_video_energy,
            "teacher_audio_energy": teacher_audio_energy,
        }
        attn_metadata = (training_batch.attn_metadata_vsa
                         if self._student_attn_kind == "vsa" else training_batch.attn_metadata)
        return loss_map, {"_fv_backward": (training_batch.timesteps, attn_metadata)}, {}

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        context = outputs.get("_fv_backward")
        if context is None:
            raise RuntimeError("MiniMax H3 recovery backward context is missing")
        self.student.backward(
            loss_map["total_loss"],
            context,
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )


__all__ = ["MiniMaxH3RecoveryMethod"]
