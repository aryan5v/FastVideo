# SPDX-License-Identifier: Apache-2.0
"""EMA (Exponential Moving Average) callback.

Owns the full EMA lifecycle: creation, per-step updates, weight
swapping for validation, and checkpoint state.  All EMA config
lives under ``callbacks.ema`` in the YAML file.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.training.training_utils import EMA_FSDP

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class EMACallback(Callback):
    """Manage EMA shadow weights for the student transformer.

    All configuration lives in the YAML ``callbacks.ema`` section:

    .. code-block:: yaml

        callbacks:
          ema:
            decay: 0.9999
            start_iter: 0

    The callback creates an ``EMA_FSDP`` instance at train start,
    updates it after each optimizer step, and exposes an
    ``ema_context()`` context manager for temporarily swapping
    EMA weights into the live model (used by validation).
    """

    def __init__(
        self,
        *,
        decay: float = 0.9999,
        start_iter: int = 0,
    ) -> None:
        self._decay = float(decay)
        self._start_iter = int(start_iter)
        self._ema_started = False
        self.student_ema: EMA_FSDP | None = None

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        student = getattr(method, "student", None)
        if student is None or student.transformer is None:
            raise ValueError("No student transformer found on method, cannot initialize EMA")

        logger.info(
            "Initializing EMA (local_shard) with "
            "decay=%s from student transformer",
            self._decay,
        )
        self.student_ema = EMA_FSDP(
            student.transformer,
            decay=self._decay,
            mode="local_shard",
        )
        # Kept for checkpoint (de)serialization: converting between local
        # shards and full tensors needs the live params' DTensor placements.
        self._transformer = student.transformer
        logger.info(
            "EMA callback enabled (decay=%s, "
            "start_iter=%d).",
            self._decay,
            self._start_iter,
        )

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if self.student_ema is None:
            return

        if iteration < self._start_iter:
            return
        if not self._ema_started:
            logger.info(
                "Starting EMA updates at iteration %d "
                "(re-initializing shadow from current "
                "model).",
                iteration,
            )
            self.student_ema._init_shadow(method.student.transformer, )
            self._ema_started = True

        self.student_ema.update(method.student.transformer, )

        tracker = getattr(method, "tracker", None)
        if tracker is not None:
            tracker.log(
                {"ema/decay": self.student_ema.decay},
                iteration,
            )

    # ----------------------------------------------------------
    # EMA context manager
    # ----------------------------------------------------------

    @contextlib.contextmanager
    def ema_context(
        self,
        transformer: torch.nn.Module,
    ) -> Generator[torch.nn.Module, None, None]:
        """Temporarily swap EMA weights into *transformer*.

        If EMA is not active, yields the transformer unchanged.
        """
        if (self.student_ema is not None and self._ema_started):
            with self.student_ema.apply_to_model(transformer, ):
                yield transformer
        else:
            yield transformer

    # ----------------------------------------------------------
    # Checkpoint state
    # ----------------------------------------------------------

    # The EMA shadow is a dict of each rank's *local parameter shards* keyed
    # by live module names (including activation-checkpointing wrapper
    # prefixes). Checkpointing the plain shards directly is broken twice
    # over: DCP deduplicates plain tensors as "replicated" so only rank 0's
    # shard survives a multi-GPU save, and a checkpoint written at one world
    # size cannot be loaded at another. Gathering full tensors at save time
    # is also wrong: DTensor.full_tensor() is a collective, and issuing
    # collectives from inside DCP's save path deadlocks (observed as a
    # 10-minute NCCL timeout at the first checkpoint of the run-2 smoke).
    #
    # The correct mechanism is DCP's own: present each shard *as a DTensor*
    # (DTensor.from_local is metadata-only, no communication). DCP then saves
    # every rank's shard and reshards natively on load at any world size —
    # exactly how the model weights under ``roles.*`` are handled.

    _AC_WRAPPER = "._checkpoint_wrapped_module"

    @classmethod
    def _clean_name(cls, name: str) -> str:
        return name.replace(cls._AC_WRAPPER, "")

    @staticmethod
    def _as_dcp_tensor(shard: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        if isinstance(param, DTensor):
            return DTensor.from_local(
                shard.to(device=param.device),
                device_mesh=param.device_mesh,
                placements=param.placements,
            )
        return shard.detach().clone()

    @staticmethod
    def _to_local_cpu(value: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            value = value.to_local()
        return value.detach().float().cpu()

    def state_dict(self) -> dict[str, Any]:
        if self.student_ema is None:
            return {}
        params = {
            self._clean_name(name): param
            for name, param in self._transformer.named_parameters()
        }
        shadow_dcp: dict[str, torch.Tensor] = {}
        for name, shard in self.student_ema.shadow.items():
            clean = self._clean_name(name)
            param = params.get(clean)
            if param is None:
                logger.warning("EMA shadow key %r has no matching parameter; dropping from checkpoint.", name)
                continue
            shadow_dcp[clean] = self._as_dcp_tensor(shard, param)
        return {
            "student_ema_sharded": shadow_dcp,
            "ema_started": self._ema_started,
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        if self.student_ema is not None:
            sharded = state_dict.get("student_ema_sharded")
            if sharded is not None:
                shadow: dict[str, torch.Tensor] = {}
                for name, _param in self._transformer.named_parameters():
                    clean = self._clean_name(name)
                    if clean in sharded:
                        shadow[name] = self._to_local_cpu(sharded[clean])
                self.student_ema.shadow = shadow
            elif state_dict.get("student_ema") is not None:
                # Legacy plain-shard state: world-size-dependent and, on
                # multi-GPU saves, missing every rank but 0. Refuse to load
                # silently-corrupt weights.
                raise ValueError(
                    "This checkpoint holds legacy per-shard EMA state, which is only valid "
                    "on the exact world size that wrote it and loses all non-rank-0 shards "
                    "on multi-GPU saves. The EMA in this checkpoint cannot be trusted; "
                    "resume without the EMA state or re-train with the portable format.")
        self._ema_started = bool(state_dict.get("ema_started", False), )
