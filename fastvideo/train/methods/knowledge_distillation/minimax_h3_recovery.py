# SPDX-License-Identifier: Apache-2.0
"""Four-call MiniMax-H3 depth-pruning recovery.

The trainable student sees the ordinary joint T2VA flow target and the frozen
50-block dense teacher's velocity on exactly the same noisy video/audio state.
Selected student blocks also match compact statistics from their corresponding
source-teacher blocks. Every video and audio term is normalized independently
so the larger visual tensor cannot hide an audio regression.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from collections.abc import Generator, Sequence
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount
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
        self.teacher = teacher
        self._student_attn_kind: Literal["dense", "vsa"] = self._infer_attn_kind()
        self._teacher_attn_kind: Literal["dense", "vsa"] = (
            "vsa" if teacher.attention_backend_name in {"VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"} else "dense")
        match_teacher_backend = bool(self.method_config.get("match_teacher_backend", False))
        if match_teacher_backend:
            if self._teacher_attn_kind != self._student_attn_kind:
                raise ValueError("four-call recovery requires a branch-matched teacher attention backend")
        elif teacher.attention_backend_name != "TORCH_SDPA":
            raise ValueError("MiniMax H3 recovery teacher must use exact dense TORCH_SDPA attention")

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


def _deployment_sigmas(grid_points: int, shift: float, device: torch.device) -> torch.Tensor:
    if grid_points != 5:
        raise ValueError("FastH3 V1 recovery requires the released five-point/four-call schedule")
    base = torch.linspace(1.0, 0.0, grid_points, device=device, dtype=torch.float32)
    return shift_noise_amount(base, shift)


def _euler_update(
    state: torch.Tensor,
    noise_minus_clean: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Apply one H3 Euler interval using the training wrapper's flow sign."""
    delta = (sigma - sigma_next).to(device=state.device, dtype=torch.float32)
    return (state.float() - delta * noise_minus_clean.float()).to(state.dtype)


def _local_parameter_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(parameter, DTensor):
            return parameter.to_local()
    except ImportError:
        pass
    return parameter


class MiniMaxH3FourCallRecoveryMethod(MiniMaxH3RecoveryMethod):
    """Warm-restart recovery against branch-matched FastH3 V1 trajectories."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        method = self.method_config
        self._grid_points = int(method.get("deployment_grid_points", 5))
        self._student_state_probability = float(method.get("student_state_probability", 0.5))
        self._trajectory_weight = float(method.get("trajectory_weight", 1.0))
        self._require_fp32_master = bool(method.get("require_fp32_master", True))
        self._video_interval_weights = self._interval_weights(method.get("video_interval_weights"), "video")
        self._audio_interval_weights = self._interval_weights(method.get("audio_interval_weights"), "audio")
        if not 0.0 <= self._student_state_probability <= 1.0:
            raise ValueError("student_state_probability must be in [0, 1]")
        if self._trajectory_weight <= 0.0:
            raise ValueError("trajectory_weight must be > 0")
        if self._teacher_attn_kind != self._student_attn_kind:
            raise ValueError("four-call recovery requires dense->dense or VSA->VSA")
        if self._require_fp32_master:
            bad = [
                name for name, parameter in self.student.transformer.named_parameters()
                if parameter.requires_grad and parameter.dtype != torch.float32
            ]
            if bad:
                raise ValueError(
                    "four-call recovery requires FP32 trainable parameters; "
                    f"found non-FP32 tensors including {bad[:5]}")

    def _interval_weights(self, raw: Any, modality: str) -> tuple[float, ...]:
        intervals = self._grid_points - 1
        values = (1.0, ) * intervals if raw is None else tuple(float(value) for value in raw)
        if len(values) != intervals or any(value <= 0.0 for value in values):
            raise ValueError(f"{modality}_interval_weights must contain {intervals} positive values")
        mean = sum(values) / len(values)
        return tuple(value / mean for value in values)

    def _shared_choice(self, upper: int) -> int:
        if self.cuda_generator is None:
            raise RuntimeError("Training RNG is not initialized")
        choice = torch.randint(
            0,
            upper,
            (1, ),
            generator=self.cuda_generator,
            device=self.student.device,
            dtype=torch.int64,
        )
        if int(self.training_config.distributed.sp_size) > 1:
            self.student.sp_group.broadcast(choice, src=0)
        return int(choice.item())

    def _set_vsa_interval(self, batch: Any, interval: int) -> None:
        if self._student_attn_kind != "vsa":
            return
        batch.attn_metadata_vsa = self.student._build_vsa_metadata(
            batch.minimax_h3_layout,
            self.student.device,
            current_timestep=interval,
        )

    def _roll_to_interval(
        self,
        batch: Any,
        interval: int,
        video_sigmas: torch.Tensor,
        audio_sigmas: torch.Tensor,
        use_student: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = batch.noise.permute(0, 2, 1, 3, 4)
        audio = batch.audio_noise
        model = self.student if use_student else self.teacher
        attn_kind = self._student_attn_kind if use_student else self._teacher_attn_kind
        with torch.no_grad():
            for step in range(interval):
                self._set_vsa_interval(batch, step)
                video_time = (1.0 - video_sigmas[step]).reshape(1)
                audio_time = (1.0 - audio_sigmas[step]).reshape(1)
                video_flow, audio_flow = model.predict_joint_noise(
                    video,
                    audio,
                    video_time,
                    audio_time,
                    batch,
                    conditional=True,
                    attn_kind=attn_kind,
                )
                video = _euler_update(video, video_flow, video_sigmas[step], video_sigmas[step + 1])
                audio = _euler_update(audio, audio_flow, audio_sigmas[step], audio_sigmas[step + 1])
        return video.detach(), audio.detach()

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        if self.cuda_generator is None:
            raise RuntimeError("Training RNG is not initialized")
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        required = {
            "clean_video": training_batch.latents,
            "clean_audio": training_batch.audio_latents,
            "video_noise": training_batch.noise,
            "audio_noise": training_batch.audio_noise,
        }
        missing = [name for name, value in required.items() if not isinstance(value, torch.Tensor)]
        if missing:
            raise RuntimeError("four-call recovery batch is missing " + ", ".join(missing))

        device = self.student.device
        video_sigmas = _deployment_sigmas(self._grid_points, 12.0, device)
        audio_sigmas = _deployment_sigmas(self._grid_points, 3.0, device)
        interval = self._shared_choice(self._grid_points - 1)
        use_student_state = self._shared_choice(10_000) < round(self._student_state_probability * 10_000)
        state_video, state_audio = self._roll_to_interval(
            training_batch,
            interval,
            video_sigmas,
            audio_sigmas,
            use_student_state,
        )
        video_time = (1.0 - video_sigmas[interval]).reshape(1)
        audio_time = (1.0 - audio_sigmas[interval]).reshape(1)
        training_batch.noisy_model_input = state_video.permute(0, 2, 1, 3, 4)
        training_batch.audio_noisy_model_input = state_audio
        training_batch.timesteps = video_time
        training_batch.audio_timesteps = audio_time
        self._set_vsa_interval(training_batch, interval)

        with torch.no_grad(), _capture_hidden_summaries(self.teacher,
                                                        self._teacher_feature_indices) as teacher_features:
            teacher_video, teacher_audio = self.teacher.predict_joint_noise(
                state_video,
                state_audio,
                video_time,
                audio_time,
                training_batch,
                conditional=True,
                attn_kind=self._teacher_attn_kind,
            )
        with _capture_hidden_summaries(self.student, self._student_feature_indices) as student_features:
            student_video, student_audio = self.student.predict_joint_noise(
                state_video,
                state_audio,
                video_time,
                audio_time,
                training_batch,
                conditional=True,
                attn_kind=self._student_attn_kind,
            )

        teacher_video_update = (_euler_update(
            state_video,
            teacher_video,
            video_sigmas[interval],
            video_sigmas[interval + 1],
        ) - state_video).detach()
        student_video_update = _euler_update(
            state_video,
            student_video,
            video_sigmas[interval],
            video_sigmas[interval + 1],
        ) - state_video
        teacher_audio_update = (_euler_update(
            state_audio,
            teacher_audio,
            audio_sigmas[interval],
            audio_sigmas[interval + 1],
        ) - state_audio).detach()
        student_audio_update = _euler_update(
            state_audio,
            student_audio,
            audio_sigmas[interval],
            audio_sigmas[interval + 1],
        ) - state_audio
        raw_update_video, update_video_energy, norm_update_video = _normalized_mse(
            student_video_update,
            teacher_video_update,
            energy_floor=self._energy_floor,
        )
        raw_update_audio, update_audio_energy, norm_update_audio = _normalized_mse(
            student_audio_update,
            teacher_audio_update,
            energy_floor=self._energy_floor,
        )

        clean_video = required["clean_video"]
        clean_audio = required["clean_audio"]
        video_noise = required["video_noise"]
        audio_noise = required["audio_noise"]
        assert isinstance(clean_video, torch.Tensor)
        assert isinstance(clean_audio, torch.Tensor)
        assert isinstance(video_noise, torch.Tensor)
        assert isinstance(audio_noise, torch.Tensor)
        _, _, norm_real_video = _normalized_mse(
            student_video,
            video_noise.permute(0, 2, 1, 3, 4) - clean_video,
            energy_floor=self._energy_floor,
        )
        _, _, norm_real_audio = _normalized_mse(
            student_audio,
            audio_noise - clean_audio,
            energy_floor=self._energy_floor,
        )

        feature_terms: list[torch.Tensor] = []
        for student_index, teacher_index in zip(
                self._student_feature_indices,
                self._teacher_feature_indices,
                strict=True,
        ):
            if student_index not in student_features or teacher_index not in teacher_features:
                raise RuntimeError("selected four-call hidden-state hook did not execute")
            _, _, normalized = _normalized_mse(
                student_features[student_index],
                teacher_features[teacher_index].detach(),
                energy_floor=self._energy_floor,
            )
            feature_terms.append(normalized)
        feature_loss = torch.stack(feature_terms).mean()
        trajectory_loss = (self._video_interval_weights[interval] * norm_update_video +
                           self._audio_interval_weights[interval] * norm_update_audio)
        denoising_loss = norm_real_video + norm_real_audio
        total_loss = (self._trajectory_weight * trajectory_loss + self._denoising_weight * denoising_loss +
                      self._feature_weight * feature_loss)
        loss_map = {
            "total_loss": total_loss,
            "trajectory_loss": trajectory_loss,
            "trajectory_video_update_loss": norm_update_video,
            "trajectory_audio_update_loss": norm_update_audio,
            "raw_video_update_loss": raw_update_video,
            "raw_audio_update_loss": raw_update_audio,
            "video_update_energy": update_video_energy,
            "audio_update_energy": update_audio_energy,
            "denoising_loss": denoising_loss,
            "hidden_feature_loss": feature_loss,
        }
        metrics: dict[str, LogScalar] = {
            "trajectory/interval": interval,
            "trajectory/student_state": float(use_student_state),
            "trajectory/video_interval_weight": self._video_interval_weights[interval],
            "trajectory/audio_interval_weight": self._audio_interval_weights[interval],
        }
        attn_metadata = (training_batch.attn_metadata_vsa
                         if self._student_attn_kind == "vsa" else training_batch.attn_metadata)
        return loss_map, {"_fv_backward": (video_time, attn_metadata)}, metrics

    def optimizers_schedulers_step(self, iteration: int) -> None:
        probes: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self._require_fp32_master and iteration == 1:
            for parameter in self.student.transformer.parameters():
                if parameter.requires_grad:
                    local = _local_parameter_tensor(parameter).detach().reshape(-1)
                    if local.numel():
                        probes.append((local, local[:min(4096, local.numel())].float().clone()))
                if len(probes) >= 16:
                    break
        super().optimizers_schedulers_step(iteration)
        if not self._require_fp32_master or iteration != 1:
            return

        changed = 0
        elements = 0
        delta_sq = 0.0
        for local, before in probes:
            after = local[:before.numel()].float()
            changed += int(torch.count_nonzero(after != before).item())
            elements += before.numel()
            delta_sq += float(torch.sum((after - before).square()).item())
        state_dtypes = Counter(
            value.dtype
            for state in self._student_optimizer.state.values()
            for name, value in state.items()
            if name in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor))
        if any(dtype != torch.float32 for dtype in state_dtypes):
            raise RuntimeError(f"four-call recovery optimizer state is not FP32: {dict(state_dtypes)}")
        if changed == 0 or delta_sq == 0.0:
            raise RuntimeError("FP32 warm restart produced zero changes across all parameter probes")
        if not dist.is_initialized() or dist.get_rank() == 0:
            assert self.tracker is not None
            self.tracker.log({
                "optimizer/master_parameter_dtype": "float32",
                "optimizer/state_dtype": "float32",
                "optimizer/update_probe_elements": elements,
                "optimizer/update_probe_changed_fraction": changed / elements,
                "optimizer/update_probe_l2": delta_sq**0.5,
            }, iteration)


__all__ = [
    "MiniMaxH3FourCallRecoveryMethod",
    "MiniMaxH3RecoveryMethod",
]
