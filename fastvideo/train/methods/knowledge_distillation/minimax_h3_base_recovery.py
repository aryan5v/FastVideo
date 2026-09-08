# SPDX-License-Identifier: Apache-2.0
"""Full-schedule Base H3 recovery with an actual FP32 update receipt."""
from pathlib import Path
import json
import math
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    MiniMaxH3RecoveryMethod,
    _local_parameter_tensor,
)


class MiniMaxH3BaseRecoveryMethod(MiniMaxH3RecoveryMethod):
    """Keep paired flow/KD training and verify the first optimizer update."""

    def single_train_step(self, batch: dict[str, Any], iteration: int) -> Any:
        result = super().single_train_step(batch, iteration)
        if any(not torch.isfinite(value).all() for value in result[0].values()):
            raise RuntimeError("Nonfinite Base recovery loss")
        return result

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
