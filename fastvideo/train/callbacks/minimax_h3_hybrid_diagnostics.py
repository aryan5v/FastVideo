# SPDX-License-Identifier: Apache-2.0
"""Training health metrics for MiniMax H3 frame-implicit attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from fastvideo.logger import init_logger
from fastvideo.models.dits.minimax_h3_hybrid.linear import BidirectionalLinearBranch, OutputGate
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class MiniMaxH3HybridDiagnosticsCallback(Callback):
    """Log write/retention/gate health and gradient-family norms.

    Forward diagnostics are detached inside each hybrid module. Gradient norms
    are reduced on every rank in a fixed family order before rank zero logs
    them, keeping distributed collective ordering deterministic.
    """

    _GRADIENT_FAMILIES = (
        ("write_log_scale", ("linear_attention.write_log_scale", )),
        ("write_logits", ("linear_attention.beta_proj.", )),
        ("retention", ("linear_attention.alpha.", )),
        ("short_conv", ("linear_attention.short_conv.", )),
        ("linear_norm", ("linear_attention.norm.", )),
        ("output_gate", ("linear_attention.output_gate.", )),
        ("linear_readout", ("to_out_linear.", )),
        ("softmax_gate", ("softmax_gate.", )),
    )

    def __init__(self, *, every_n_steps: int = 10) -> None:
        if int(every_n_steps) <= 0:
            raise ValueError("every_n_steps must be positive")
        self._every_n_steps = int(every_n_steps)

    def _enabled(self, iteration: int) -> bool:
        return iteration == 1 or iteration % self._every_n_steps == 0

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @staticmethod
    def _set_recording(method: TrainingMethod, enabled: bool) -> None:
        for module in method.student.transformer.modules():
            if isinstance(module, BidirectionalLinearBranch | OutputGate):
                module.record_diagnostics = enabled

    def on_train_start(self, method: TrainingMethod, iteration: int = 0) -> None:
        del iteration
        # The first step is always useful, especially for catching bad
        # initial write strength. Later steps are armed one iteration ahead.
        self._set_recording(method, True)

    def on_before_optimizer_step(self, method: TrainingMethod, iteration: int = 0) -> None:
        if not self._enabled(iteration):
            return
        parameters = list(method.student.transformer.named_parameters())
        reference = next((parameter for _, parameter in parameters if parameter.requires_grad), None)
        if reference is None:
            return
        totals = torch.zeros(len(self._GRADIENT_FAMILIES), device=reference.device, dtype=torch.float32)
        for family_index, (_, markers) in enumerate(self._GRADIENT_FAMILIES):
            for name, parameter in parameters:
                if not parameter.requires_grad or not any(marker in name for marker in markers):
                    continue
                grad = parameter.grad
                if grad is None:
                    continue
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                totals[family_index] += local.detach().float().square().sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        if self._rank() != 0:
            return
        tracker = getattr(method, "tracker", None)
        if tracker is None:
            return
        tracker.log(
            {
                f"hybrid_grad_norm/{family}": totals[index].sqrt()
                for index, (family, _) in enumerate(self._GRADIENT_FAMILIES)
            },
            iteration,
        )

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, object],
        iteration: int = 0,
    ) -> None:
        del loss_dict
        if self._enabled(iteration) and self._rank() == 0:
            tracker = getattr(method, "tracker", None)
            if tracker is not None:
                metrics: dict[str, torch.Tensor] = {}
                for module_name, module in method.student.transformer.named_modules():
                    if isinstance(module, BidirectionalLinearBranch | OutputGate):
                        prefix = f"hybrid_health/{module_name}"
                        metrics.update({f"{prefix}/{key}": value for key, value in module.latest_diagnostics.items()})
                if metrics:
                    tracker.log(metrics, iteration)
                else:
                    logger.warning("No MiniMax H3 hybrid forward diagnostics were available at step %d", iteration)
        self._set_recording(method, self._enabled(iteration + 1))


__all__ = ["MiniMaxH3HybridDiagnosticsCallback"]
