# SPDX-License-Identifier: Apache-2.0
"""Full-schedule Base H3 recovery with an actual FP32 update receipt."""
from pathlib import Path
import json
import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    MiniMaxH3RecoveryMethod,
    _local_parameter_tensor,
    _normalized_mse,
    _euler_update,
)

from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount


@contextmanager
def _capture_tokens(model: Any, indices: Any) -> Any:
    targets: dict[int, torch.Tensor] = {}
    handles = []
    for index in indices:

        def hook(_module: Any, _inputs: Any, output: torch.Tensor, index: int = index) -> None:
            targets[index] = output.detach()

        handles.append(model.transformer.transformer_blocks[index].register_forward_hook(hook))
    try:
        yield targets
    finally:
        for handle in handles:
            handle.remove()


def _seam_loss(student: torch.Tensor,
               teacher: torch.Tensor,
               layout: Any,
               group: Any,
               floor: float,
               audio_weight: float = 1.0) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("Teacher/student token alignment differs")
    local_rows = student.shape[1]
    start = group.rank_in_group * local_rows
    terms = []
    for indices in (layout.video_indices, layout.audio_indices):
        selected = indices[(indices >= start) & (indices < start + local_rows)].to(student.device) - start
        target = teacher.index_select(1, selected).float()
        prediction = student.index_select(1, selected).float()
        stats = torch.tensor([target.square().sum().item(), target.numel()], device=student.device)
        if group.world_size > 1:
            dist.all_reduce(stats, group=group.device_group)
        if stats[1] == 0:
            raise ValueError("Packed document has no rows for a supervised modality")
        rms = (stats[0] / stats[1]).clamp_min(floor).sqrt()
        # Exclude extreme teacher activations; do not clamp the student gradient.
        mask = target.abs() <= 10 * rms
        kept = torch.stack((mask.sum().float(), (target.square() * mask).sum()))
        if group.world_size > 1:
            dist.all_reduce(kept, group=group.device_group)
        # Excluded outliers must not dominate the normalization denominator.
        energy = (kept[1] / kept[0].clamp_min(1)).clamp_min(floor)
        error = ((prediction - target).square() * mask).sum()
        weight = audio_weight if indices is layout.audio_indices else 1.0
        terms.append(weight * error * group.world_size / kept[0].clamp_min(1) / energy)
    return torch.stack(terms).sum()


class MiniMaxH3BaseRecoveryMethod(MiniMaxH3RecoveryMethod):
    """Keep paired flow/KD training and verify the first optimizer update.

    Detail-band controls (audit 2026-09-08): the shift-12/shift-3 warping of the
    50-point deployment grid leaves only ~2/49 video and ~4/49 audio training
    intervals below sigma 0.2, i.e. the intervals that create edge sharpness and
    audio brightness are barely trained. ``low_sigma_interval_fraction`` re-
    balances the draw without changing the solver. ``video_velocity_weight`` /
    ``audio_velocity_weight`` fix the ~5x loss-magnitude imbalance that let
    video outvote audio at every step, and ``student_state_probability``
    restores exposure to the student's own prefix states once the student is
    coherent (teacher-prefix-only training is pure exposure bias).
    """

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        self._low_sigma_fraction = float(mcfg.get("low_sigma_interval_fraction", 0.0))
        self._low_sigma_count = int(mcfg.get("low_sigma_interval_count", 12))
        self._video_velocity_weight = float(mcfg.get("video_velocity_weight", self._teacher_velocity_weight))
        self._audio_velocity_weight = float(mcfg.get("audio_velocity_weight", self._teacher_velocity_weight))
        self._student_state_probability = float(mcfg.get("student_state_probability", 0.0))
        self._grad_probe_every = int(mcfg.get("modality_grad_probe_every", 0))
        if self._grad_probe_every < 0:
            raise ValueError("method.modality_grad_probe_every must be >= 0")
        self._audio_seam_weight = float(mcfg.get("audio_seam_weight", 1.0))
        if self._audio_seam_weight <= 0.0:
            raise ValueError("method.audio_seam_weight must be > 0")
        self._validate_controls()

    def _validate_controls(self) -> None:
        if not 0.0 <= self._low_sigma_fraction <= 1.0:
            raise ValueError("method.low_sigma_interval_fraction must be in [0, 1]")
        if not 1 <= self._low_sigma_count <= int(self.method_config.get("teacher_grid_points", 50)) - 2:
            raise ValueError("method.low_sigma_interval_count must be in [1, teacher_grid_points - 2]")
        if not 0.0 <= self._student_state_probability <= 1.0:
            raise ValueError("method.student_state_probability must be in [0, 1]")
        for name in ("video_velocity_weight", "audio_velocity_weight"):
            if getattr(self, f"_{name}") < 0.0:
                raise ValueError(f"method.{name} must be non-negative")

    def _sample_interval(self, points: int) -> tuple[int, int]:
        """Return (interval, low_sigma_flag) with SP-consistent RNG."""
        low = self._shared_choice(10_000) < round(self._low_sigma_fraction * 10_000)
        if low:
            offset = self._shared_choice(self._low_sigma_count)
            return points - 1 - self._low_sigma_count + offset, 1
        return self._shared_choice(points - 1), 0

    def _probe_modality_grad_share(self, kv: torch.Tensor, ka: torch.Tensor) -> dict[str, float]:
        """Measure how much of the KD gradient signal each modality actually drives."""
        probe = [p for p in self.student.transformer.parameters() if p.requires_grad and p.numel() >= 1_000_000][:8]
        if not probe:
            return {}
        norms = []
        for term in (kv, ka):
            grads = torch.autograd.grad(term, probe, retain_graph=True, allow_unused=True)
            norms.append(math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None)))
        total = norms[0] + norms[1]
        if total <= 0.0:
            return {}
        return {
            "audio_grad_share": norms[1] / total,
            "video_grad_norm": norms[0],
            "audio_grad_norm": norms[1],
        }

    def single_train_step(self, batch: dict[str, Any], iteration: int) -> Any:
        prompt_only = bool(batch.get("prompt_only", False))
        if prompt_only and not self.method_config.get("allow_prompt_only", False):
            raise ValueError("Prompt-only teacher supervision requires explicit allow_prompt_only")
        tb = self.student.prepare_batch(batch,
                                        generator=self.cuda_generator,
                                        latents_source="zeros" if prompt_only else "data")
        device = self.student.device
        points = int(self.method_config.get("teacher_grid_points", 50))
        if points != 50:
            raise ValueError("Base pilot requires the full 50-grid-point teacher schedule")
        interval, low_sigma = self._sample_interval(points)
        base = torch.linspace(1, 0, points, device=device)
        vs = shift_noise_amount(base, 12.0)
        aus = shift_noise_amount(base, 3.0)
        video = tb.noise.permute(0, 2, 1, 3, 4)
        audio = tb.audio_noise
        use_student_prefix = self._shared_choice(10_000) < round(self._student_state_probability * 10_000)
        # Prefix states are integrated without gradient; the source model is
        # configurable so a coherent student can train on its own trajectory.
        prefix_model = self.student if use_student_prefix else self.teacher
        with torch.no_grad():
            for i in range(interval):
                fv, fa = prefix_model.predict_joint_noise(video,
                                                          audio, (1 - vs[i]).reshape(1), (1 - aus[i]).reshape(1),
                                                          tb,
                                                          conditional=True,
                                                          attn_kind="dense")
                video = _euler_update(video, fv, vs[i], vs[i + 1])
                audio = _euler_update(audio, fa, aus[i], aus[i + 1])
        vt, at = (1 - vs[interval]).reshape(1), (1 - aus[interval]).reshape(1)
        with torch.no_grad(), _capture_tokens(self.teacher, self._teacher_feature_indices) as targets:
            tv, ta = self.teacher.predict_joint_noise(video, audio, vt, at, tb, conditional=True, attn_kind="dense")
        terms: list[torch.Tensor] = []
        handles = []
        for local_index, original_index in zip(self._student_feature_indices,
                                               self._teacher_feature_indices,
                                               strict=True):

            def hook(_module: Any, _inputs: Any, output: torch.Tensor, index: int = original_index) -> None:
                terms.append(
                    _seam_loss(output, targets[index], tb.minimax_h3_layout, self.student.sp_group,
                               self._energy_floor, self._audio_seam_weight))

            handles.append(self.student.transformer.transformer_blocks[local_index].register_forward_hook(hook))
        try:
            sv, sa = self.student.predict_joint_noise(video, audio, vt, at, tb, conditional=True, attn_kind="dense")
        finally:
            for handle in handles:
                handle.remove()
        if len(terms) != len(self._student_feature_indices):
            raise RuntimeError("Tokenwise seam hooks did not all execute")
        feature = torch.stack(terms).mean()
        kv = _normalized_mse(sv, tv.detach(), energy_floor=self._energy_floor)[2]
        ka = _normalized_mse(sa, ta.detach(), energy_floor=self._energy_floor)[2]
        # Placeholder zeros provide geometry only, never a real-data target.
        dv = da = torch.zeros((), device=device)
        if not prompt_only:
            # Separate forward on correctly noised real data, at the same sampled time.
            real_v = (1 - vs[interval]) * tb.latents + vs[interval] * tb.noise.permute(0, 2, 1, 3, 4)
            real_a = (1 - aus[interval]) * tb.audio_latents + aus[interval] * tb.audio_noise
            rv, ra = self.student.predict_joint_noise(real_v, real_a, vt, at, tb, conditional=True, attn_kind="dense")
            dv = _normalized_mse(rv, tb.noise.permute(0, 2, 1, 3, 4) - tb.latents, energy_floor=self._energy_floor)[2]
            da = _normalized_mse(ra, tb.audio_noise - tb.audio_latents, energy_floor=self._energy_floor)[2]
        total = (self._video_velocity_weight * kv + self._audio_velocity_weight * ka +
                 self._feature_weight * feature + self._denoising_weight * (dv + da))
        losses = {
            "total_loss": total,
            "video_velocity_kd": kv,
            "audio_velocity_kd": ka,
            "tokenwise_seam_loss": feature,
            "real_video_flow_loss": dv,
            "real_audio_flow_loss": da
        }
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise RuntimeError("Nonfinite Base recovery loss")
        from fastvideo.train.utils.h3_prompt_coverage import record_prompt_use
        coverage = record_prompt_use(self, batch, iteration)
        probe: dict[str, float] = {}
        if self._grad_probe_every and iteration % self._grad_probe_every == 0:
            probe = self._probe_modality_grad_share(kv, ka)
        return losses, {
            "_fv_backward": (vt, tb.attn_metadata)
        }, {
            "teacher_prefix_interval": interval,
            "low_sigma_interval": low_sigma,
            "student_prefix": int(use_student_prefix),
            "video_velocity_weight": self._video_velocity_weight,
            "audio_velocity_weight": self._audio_velocity_weight,
            "prompt_only": int(prompt_only),
            **probe,
            **coverage
        }

    def optimizers_schedulers_step(self, iteration: int) -> None:
        if getattr(self, "_base_update_verified", False):
            super().optimizers_schedulers_step(iteration)
            return
        probes: list[tuple[torch.Tensor, torch.Tensor]] = []
        for parameter in self.student.transformer.parameters():
            if parameter.requires_grad:
                local = _local_parameter_tensor(parameter).detach().reshape(-1)
                if local.dtype != torch.float32:
                    raise RuntimeError("Base recovery requires FP32 master parameters")
                if local.numel() and len(probes) < 16:
                    probes.append((local, local[:4096].clone()))
        super().optimizers_schedulers_step(iteration)
        changed = 0
        delta_sq = 0.0
        for local, before in probes:
            after = local[:before.numel()]
            if not torch.isfinite(after).all():
                raise RuntimeError("Nonfinite Base recovery parameter update")
            changed += int(torch.count_nonzero(after != before).item())
            delta_sq += float((after - before).square().sum().item())
        states = [
            v for state in self._student_optimizer.state.values() for k, v in state.items()
            if k in {"exp_avg", "exp_avg_sq"}
        ]
        if not states or any(v.dtype != torch.float32 for v in states):
            raise RuntimeError("Base recovery requires populated FP32 Adam moments")
        if changed == 0 or not math.isfinite(delta_sq) or delta_sq <= 0:
            raise RuntimeError("Base recovery did not produce a finite nonzero update")
        rank = dist.get_rank() if dist.is_initialized() else 0
        receipt = {
            "passed": True,
            "iteration": iteration,
            "rank": rank,
            "changed_probe_elements": changed,
            "delta_l2": delta_sq**0.5,
            "learning_rates": [float(g["lr"]) for g in self._student_optimizer.param_groups]
        }
        root = Path(self.training_config.checkpoint.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"base_update_rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        self._base_update_verified = True
