# SPDX-License-Identifier: Apache-2.0
"""A resumed run must prove its first real optimizer update, even past step one."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    MiniMaxH3FourCallRecoveryMethod,
    MiniMaxH3RecoveryMethod,
)


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
