# SPDX-License-Identifier: Apache-2.0
"""Bounded comparison of hard block removal and training with block skipping.

Inspired by the adaptation-before-extraction experiment in FastLightGen
(arXiv:2603.01685). This implementation keeps a frozen branch-matched H3 V1
teacher, a fixed final block set, and the existing joint four-call objective.
It is not a reproduction of that paper or a learned mask-search algorithm.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from fastvideo.distributed import get_world_group, get_world_rank

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    MiniMaxH3FourCallRecoveryMethod,
    _block_map,
    _deployment_sigmas,
    _euler_update,
    _normalized_mse,
    _local_parameter_tensor,
    _transformer_blocks,
)


def sample_execution_mask(
    num_blocks: int,
    retained: tuple[int, ...],
    *,
    drop_probability: float,
    seed: int,
) -> tuple[bool, ...]:
    """Always execute final retained blocks; optionally execute removable ones.

    An independent CPU RNG makes the schedule replayable by iteration and
    avoids shifting the noise/timestep RNG relative to the hard-prune control.
    """
    if (len(retained) < 2 or retained != tuple(sorted(set(retained))) or retained[0] != 0
            or retained[-1] != num_blocks - 1
            or any(type(index) is not int or not 0 <= index < num_blocks for index in retained)):
        raise ValueError("retained blocks must be unique, ordered, in range, and include both endpoints")
    if not math.isfinite(drop_probability) or not 0 <= drop_probability <= 1:
        raise ValueError("drop_probability must be finite and in [0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.rand(num_blocks, generator=generator).tolist()
    kept = set(retained)
    return tuple(index in kept or draw >= drop_probability for index, draw in enumerate(draws))


class MiniMaxH3MaskRecoveryMethod(MiniMaxH3FourCallRecoveryMethod):
    """Use the same final mask and loss to isolate removal-adaptation policy."""

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        self._mask_mode = str(mcfg.get("pruning_mode", "hard"))
        if self._mask_mode not in {"hard", "annealed_skip"}:
            raise ValueError("pruning_mode must be hard or annealed_skip")
        self._retained = tuple(mcfg["retained_blocks"])
        self._num_blocks = len(_transformer_blocks(self.student))
        if _block_map(self.student) != tuple(range(self._num_blocks)):
            raise ValueError("mask pilot requires an unpruned parent with original V1 block indices")
        self._final_mask = sample_execution_mask(self._num_blocks, self._retained, drop_probability=1.0, seed=0)
        if not set(self._student_feature_indices).issubset(self._retained):
            raise ValueError("feature capture indices must be retained in the final static model")
        self._execution_mask = self._final_mask
        self._mask_seed = int(mcfg.get("mask_seed", 20260906))
        self._initial_drop = float(mcfg.get("initial_drop_probability", 0.5))
        sample_execution_mask(self._num_blocks, self._retained, drop_probability=self._initial_drop, seed=0)
        self._anneal_steps = int(mcfg.get("mask_anneal_steps", 100))
        if self._anneal_steps < 1:
            raise ValueError("mask_anneal_steps must be positive")
        if self._denoising_weight != 0.0:
            raise ValueError("mask rollout pilot requires denoising_weight=0; data targets need a separate forward")
        self._validation_every = int(mcfg.get("validation_every", 25))
        self._validation_samples = int(mcfg.get("validation_samples", 2))
        if self._validation_every < 1 or self._validation_samples < 1:
            raise ValueError("validation cadence and sample count must be positive")
        self._validation_data = None
        validation_path = mcfg.get("validation_data_path")
        if validation_path:
            from fastvideo.dataset.minimax_h3_artifact_dataset import MiniMaxH3ArtifactDataset
            self._validation_data = MiniMaxH3ArtifactDataset(validation_path,
                                                             batch_size=1,
                                                             num_sp_groups=1,
                                                             sp_world_size=int(
                                                                 self.training_config.distributed.sp_size),
                                                             global_rank=0,
                                                             seed=0,
                                                             drop_last=False)
            if len(self._validation_data) < self._validation_samples:
                raise ValueError("not enough held-out validation samples")
        self._train_started_at = 0.0

    def on_train_start(self) -> None:
        super().on_train_start()
        self._train_started_at = time.monotonic()

    def _predict_student_joint_noise(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student.predict_joint_noise(*args, block_execution_mask=self._execution_mask, **kwargs)

    def single_train_step(self, batch: dict[str, Any], iteration: int):
        fraction = min(1.0, max(0.0, (iteration - 1) / self._anneal_steps))
        probability = 1.0 if self._mask_mode == "hard" else self._initial_drop + (1 - self._initial_drop) * fraction
        mask = sample_execution_mask(self._num_blocks,
                                     self._retained,
                                     drop_probability=probability,
                                     seed=self._mask_seed + iteration)
        # All SP ranks must take identical branches, including in FSDP collectives.
        tensor_mask = torch.tensor(mask, device=self.student.device, dtype=torch.int64)
        if int(self.training_config.distributed.sp_size) > 1:
            self.student.sp_group.broadcast(tensor_mask, src=0)
        self._execution_mask = tuple(bool(value) for value in tensor_mask.tolist())
        losses, outputs, metrics = super().single_train_step(batch, iteration)
        if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
            raise RuntimeError("non-finite mask-recovery loss; stopping before optimizer update")
        metrics.update({
            "pruning/active_blocks": sum(self._execution_mask),
            "pruning/target_blocks": len(self._retained),
            "pruning/drop_probability": probability,
            "pruning/mask_bits": sum(1 << i for i, enabled in enumerate(self._execution_mask) if enabled),
            "pruning/wall_seconds": time.monotonic() - self._train_started_at,
        })
        return losses, outputs, metrics

    def optimizers_schedulers_step(self, iteration: int) -> None:
        gradients = [
            _local_parameter_tensor(parameter.grad) for parameter in self.student.transformer.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("mask recovery produced no student gradients")
        nonfinite = torch.stack([(~torch.isfinite(gradient)).any() for gradient in gradients]).any().to(torch.int32)
        nonfinite = get_world_group().all_reduce(nonfinite.reshape(1))
        if bool(nonfinite.item()):
            raise RuntimeError("non-finite mask-recovery gradients; stopping before optimizer update")
        super().optimizers_schedulers_step(iteration)

    @torch.no_grad()
    def on_validation_begin(self, iteration: int) -> dict[str, float]:
        """Fixed seeds and final mask at every gate, including before training.

        Compare both closed-loop terminal states and student predictions on
        teacher states. Validation uses its own RNG and never advances the
        training data stream, mask schedule or training noise generator.
        """
        if self._validation_data is None or iteration % self._validation_every:
            return {}
        device = self.student.device
        vs, aus = (_deployment_sigmas(5, shift, device) for shift in (12.0, 3.0))
        totals: dict[str, float] = {}
        old_mask = self._execution_mask
        self._execution_mask = self._final_mask
        started = time.monotonic()
        try:
            for sample_index in range(self._validation_samples):
                rng = torch.Generator(device=device).manual_seed(2026090600 + sample_index)
                batch = self.student.prepare_batch(self._validation_data.__getitems__([sample_index]),
                                                   generator=rng,
                                                   latents_source="data")
                tv = batch.noise.permute(0, 2, 1, 3, 4)
                ta = batch.audio_noise
                sv, sa = tv.clone(), ta.clone()
                for interval in range(4):
                    self._set_vsa_interval(batch, interval)
                    vt, at = (1 - vs[interval]).reshape(1), (1 - aus[interval]).reshape(1)
                    tfv, tfa = self.teacher.predict_joint_noise(tv,
                                                                ta,
                                                                vt,
                                                                at,
                                                                batch,
                                                                conditional=True,
                                                                attn_kind=self._teacher_attn_kind)
                    sfv, sfa = self._predict_student_joint_noise(tv,
                                                                 ta,
                                                                 vt,
                                                                 at,
                                                                 batch,
                                                                 conditional=True,
                                                                 attn_kind=self._student_attn_kind)
                    for modality, prediction, target in (("video", sfv, tfv), ("audio", sfa, tfa)):
                        error = float(_normalized_mse(prediction, target, energy_floor=self._energy_floor)[2])
                        key = f"validation/{modality}_teacher_state_interval{interval}"
                        totals[key] = totals.get(key, 0.0) + error / self._validation_samples
                    if interval:
                        sfv, sfa = self._predict_student_joint_noise(sv,
                                                                     sa,
                                                                     vt,
                                                                     at,
                                                                     batch,
                                                                     conditional=True,
                                                                     attn_kind=self._student_attn_kind)
                    tv = _euler_update(tv, tfv, vs[interval], vs[interval + 1])
                    ta = _euler_update(ta, tfa, aus[interval], aus[interval + 1])
                    sv = _euler_update(sv, sfv, vs[interval], vs[interval + 1])
                    sa = _euler_update(sa, sfa, aus[interval], aus[interval + 1])
                for modality, prediction, target in (("video", sv, tv), ("audio", sa, ta)):
                    error = float(_normalized_mse(prediction, target, energy_floor=self._energy_floor)[2])
                    key = f"validation/{modality}_closed_loop_endpoint"
                    totals[key] = totals.get(key, 0.0) + error / self._validation_samples
        finally:
            self._execution_mask = old_mask
        if not all(math.isfinite(value) for value in totals.values()):
            raise RuntimeError("non-finite held-out pruning validation")
        totals["validation/wall_seconds"] = time.monotonic() - started
        totals["validation/target_blocks"] = float(len(self._retained))
        totals["pruning/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        if get_world_rank() == 0:
            output = Path(self.training_config.checkpoint.output_dir) / "heldout_metrics.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({
                        "iteration": iteration,
                        "mode": self._mask_mode,
                        "retained_blocks": self._retained,
                        **totals
                    }) + "\n")
        return totals
