"""Exercise update guards without importing CUDA-only model kernels."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist


def make_method(tmp_path, update=True, dtype=torch.float32):
    class Base:
        def optimizers_schedulers_step(self, iteration):
            if update:
                self._student_optimizer.step()
    path = Path(__file__).resolve().parents[4] / 'fastvideo/train/methods/knowledge_distillation/minimax_h3_base_recovery.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = dict(MiniMaxH3RecoveryMethod=Base, torch=torch, dist=dist, Path=Path,
                     json=json, math=math, Any=Any, _local_parameter_tensor=lambda p: p)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    method = namespace['MiniMaxH3BaseRecoveryMethod']()
    model = torch.nn.Linear(2, 2, dtype=dtype)
    method.student = SimpleNamespace(transformer=model)
    method.training_config = SimpleNamespace(checkpoint=SimpleNamespace(output_dir=str(tmp_path)))
    method._student_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model(torch.ones(1, 2, dtype=dtype)).sum().backward()
    return method


def test_real_update_writes_receipt(tmp_path):
    method = make_method(tmp_path)
    method.optimizers_schedulers_step(1)
    receipt = json.loads((tmp_path / 'base_update_rank0.json').read_text())
    assert receipt['passed'] and receipt['changed_probe_elements'] > 0
    assert receipt['learning_rates'] == [3e-5]


def test_no_optimizer_step_fails(tmp_path):
    with pytest.raises(RuntimeError, match='populated FP32'):
        make_method(tmp_path, update=False).optimizers_schedulers_step(1)


def test_bf16_master_fails(tmp_path):
    with pytest.raises(RuntimeError, match='FP32 master'):
        make_method(tmp_path, dtype=torch.bfloat16).optimizers_schedulers_step(1)
