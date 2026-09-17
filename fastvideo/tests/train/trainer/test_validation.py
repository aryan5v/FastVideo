# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration tests for Trainer validation hook dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.callbacks.validation import ValidationCallback
from fastvideo.train.trainer import Trainer
from fastvideo.train.utils.training_config import TrainingConfig


class _RecordingValidationCallback(ValidationCallback):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[int] = []

    def _run_validation(self, method: Any, step: int) -> None:  # type: ignore[override]
        del method
        self.run_calls.append(step)


class _DummyTracker:

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class _DummyMethod:

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.train_start_calls = 0
        self.zero_grad_steps: list[int] = []
        self.optimizer_steps: list[int] = []
        self.backward_calls = 0
        self.tracker = None

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def on_train_start(self) -> None:
        self.train_start_calls += 1

    def manages_optimization(self) -> bool:
        return False

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        assert batch["sample"] == "x"
        loss = self.weight * 0.0 + 1.0
        return {"total_loss": loss}, {}, {"iteration_seen": float(iteration)}

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int,
    ) -> None:
        del outputs
        self.backward_calls += 1
        (loss_map["total_loss"] / grad_accum_rounds).backward()

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self.optimizer_steps.append(iteration)

    def optimizers_zero_grad(self, iteration: int) -> None:
        self.zero_grad_steps.append(iteration)
        self.weight.grad = None


class _RecordingCheckpointManager:

    def __init__(self) -> None:
        self.inference_events: list[tuple[int, bool]] = []
        self.training_events: list[int] = []
        self.final_events: list[int] = []

    def maybe_resume(self, *, resume_from_checkpoint: str) -> None:
        assert resume_from_checkpoint == ""
        return None

    def maybe_save_inference(self, step: int, *, validation_scheduled: bool) -> None:
        self.inference_events.append((step, validation_scheduled))

    def maybe_save(self, step: int) -> None:
        self.training_events.append(step)

    def save_final(self, step: int) -> None:
        self.final_events.append(step)


def test_trainer_runs_validation_callback_during_training(monkeypatch, ) -> None:
    tracker = _DummyTracker()
    group = SimpleNamespace(rank=0, local_rank=0, rank_in_group=0, world_size=1)

    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: group)
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: group)
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_world_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_sp_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    cfg = TrainingConfig()
    cfg.tracker.project_name = ""
    cfg.loop.gradient_accumulation_steps = 1
    callback_configs = {
        "validation": {
            "_target_": f"{__name__}._RecordingValidationCallback",
            "pipeline_target": "unused.pipeline.Target",
            "dataset_file": "unused.json",
            "every_steps": 2,
        }
    }
    trainer = Trainer(
        cfg,
        callback_configs=callback_configs,
    )
    method = _DummyMethod()
    checkpoint_manager = _RecordingCheckpointManager()

    trainer.run(
        method,
        dataloader=[{
            "sample": "x"
        }],
        max_steps=3,
        checkpoint_manager=checkpoint_manager,
    )

    validation = trainer.callbacks._callbacks["validation"]
    assert isinstance(validation, _RecordingValidationCallback)
    assert validation.run_calls == [0, 2]
    assert method.train_start_calls == 1
    assert method.backward_calls == 3
    assert method.zero_grad_steps == [0, 1, 2, 3]
    assert method.optimizer_steps == [1, 2, 3]
    assert [step for _, step in tracker.logs] == [1, 2, 3]
    assert tracker.finished is True
    assert checkpoint_manager.inference_events == [
        (0, True),
        (1, False),
        (2, True),
        (3, False),
    ]
    assert checkpoint_manager.training_events == [1, 2, 3]
    assert checkpoint_manager.final_events == [3]
