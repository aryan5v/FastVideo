from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.minimax_h3_base_recovery import (
    MiniMaxH3BaseRecoveryMethod,
)


def _method_with_warmup(tmp_path):
    method = object.__new__(MiniMaxH3BaseRecoveryMethod)
    torch.nn.Module.__init__(method)
    transformer = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    method.student = SimpleNamespace(transformer=transformer)
    method.training_config = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
    )
    method._student_optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=1.0e-3,
        weight_decay=0.0,
    )
    method._student_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        method._student_optimizer,
        lr_lambda=lambda step: step / 2,
    )
    return method, transformer


def test_update_probe_defers_during_zero_lr_warmup(tmp_path):
    method, transformer = _method_with_warmup(tmp_path)
    before = transformer.weight.detach().clone()
    transformer.weight.grad = torch.ones_like(transformer.weight)

    method.optimizers_schedulers_step(0)

    torch.testing.assert_close(transformer.weight, before)
    assert not getattr(method, "_base_update_verified", False)
    assert not list(tmp_path.glob("base_update_rank*.json"))

    transformer.weight.grad = torch.ones_like(transformer.weight)
    method.optimizers_schedulers_step(1)

    assert method._base_update_verified
    assert not torch.equal(transformer.weight, before)
    assert (tmp_path / "base_update_rank0.json").is_file()


def test_update_probe_still_rejects_zero_change_at_nonzero_lr(tmp_path):
    method, transformer = _method_with_warmup(tmp_path)
    method._student_optimizer.param_groups[0]["lr"] = 1.0e-3
    transformer.weight.grad = torch.zeros_like(transformer.weight)

    with pytest.raises(RuntimeError, match="finite nonzero update"):
        method.optimizers_schedulers_step(1)
