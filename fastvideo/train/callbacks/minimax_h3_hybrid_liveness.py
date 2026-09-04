# SPDX-License-Identifier: Apache-2.0
"""Fail fast when the MiniMax H3 hybrid far branch has dead gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class MiniMaxH3HybridGradientLivenessCallback(Callback):
    """Verify the zero-residual hybrid initialization becomes trainable.

    ``to_out_linear`` deliberately starts at zero, so it must receive a
    gradient on the first optimizer step. Once that readout has been updated,
    at least one internal far-branch tensor must receive a gradient on the
    second step. This catches the all-zero initialization failure before a
    multi-node job spends hours learning only the local softmax gate.
    """

    _INTERNAL_MARKERS = (
        "linear_attention.alpha.",
        "linear_attention.beta_proj.",
        "linear_attention.write_log_scale",
        "linear_attention.norm.",
        "linear_attention.output_gate.",
        "linear_attention.short_conv.",
    )

    def __init__(self, *, readout_step: int = 1, internal_step: int = 2) -> None:
        self._readout_step = int(readout_step)
        self._internal_step = int(internal_step)

    @staticmethod
    def _has_nonzero_grad(parameter: torch.nn.Parameter) -> bool:
        grad = parameter.grad
        if grad is None:
            return False
        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        return bool(torch.count_nonzero(local_grad.detach()).item())

    def _count_live(self, method: TrainingMethod, markers: tuple[str, ...]) -> tuple[int, int]:
        total = 0
        live = 0
        for name, parameter in method.student.transformer.named_parameters():
            if not parameter.requires_grad or not any(marker in name for marker in markers):
                continue
            total += 1
            live += int(self._has_nonzero_grad(parameter))
        return live, total

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        markers: tuple[str, ...]
        if iteration == self._readout_step:
            label = "readout"
            markers = ("to_out_linear.", )
        elif iteration == self._internal_step:
            label = "internal"
            markers = self._INTERNAL_MARKERS
        else:
            return

        live, total = self._count_live(method, markers)
        if total == 0:
            raise RuntimeError(f"Hybrid gradient liveness check found no trainable {label} parameters")
        if live == 0:
            raise RuntimeError(
                f"Hybrid gradient liveness check failed at step {iteration}: all {total} {label} tensors have "
                "zero or missing gradients")

        logger.info("Hybrid gradient liveness step %d: %d/%d %s tensors are live", iteration, live, total, label)
        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            tracker.log({f"hybrid_grad_liveness/{label}_live_tensors": live}, iteration)


__all__ = ["MiniMaxH3HybridGradientLivenessCallback"]
