# SPDX-License-Identifier: Apache-2.0
"""A resumed run must prove its first real optimizer update, even past step one."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    MiniMaxH3FourCallRecoveryMethod,
    MiniMaxH3RecoveryMethod,
)


def test_resume_configured_learning_rate_overrides_loaded_optimizer() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    obj = object.__new__(MiniMaxH3RecoveryMethod)
    obj.training_config = SimpleNamespace(optimizer=SimpleNamespace(learning_rate=3.0e-5))
    obj._student_optimizer = optimizer
    obj._student_lr_scheduler = scheduler
    logs = []
    obj.tracker = SimpleNamespace(log=lambda values, step: logs.append((values, step)))

    obj.on_checkpoint_loaded(200)

    assert optimizer.param_groups[0]["lr"] == 3.0e-5
    assert optimizer.param_groups[0]["initial_lr"] == 3.0e-5
    assert scheduler.base_lrs == [3.0e-5]
    assert logs == [({"optimizer/resumed_learning_rate": 3.0e-5}, 200)]


@pytest.mark.parametrize('learning_rate', [0.0, 0.01])
def test_resume_checks_first_real_update_at_step_201(monkeypatch, learning_rate):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    obj = object.__new__(MiniMaxH3FourCallRecoveryMethod)
    obj._require_fp32_master = True
    obj._optimizer_update_verified = False
    obj.student = SimpleNamespace(transformer=model)
    obj._student_optimizer = optimizer
    logs = []
    obj.tracker = SimpleNamespace(log=lambda values, step: logs.append((values, step)))
    monkeypatch.setattr(MiniMaxH3RecoveryMethod, 'optimizers_schedulers_step',
                        lambda self, iteration: optimizer.step())
    model.weight.grad = torch.ones_like(model.weight)
    if learning_rate == 0:
        with pytest.raises(RuntimeError, match='zero changes'):
            obj.optimizers_schedulers_step(201)
        assert not obj._optimizer_update_verified
    else:
        obj.optimizers_schedulers_step(201)
        assert obj._optimizer_update_verified
        assert logs[0][1] == 201
        assert logs[0][0]['optimizer/update_probe_changed_fraction'] > 0
        obj.optimizers_schedulers_step(202)
        assert len(logs) == 1
