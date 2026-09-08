"""Tokenwise masking, modality normalization and SP row alignment contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

p = Path(__file__).resolve().parents[4] / 'fastvideo/train/methods/knowledge_distillation/minimax_h3_base_recovery.py'
tree = ast.parse(p.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_seam_loss')
ns = dict(torch=torch, dist=dist, Any=Any)
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), 'exec'), ns)
loss = ns['_seam_loss']
group = SimpleNamespace(rank_in_group=0, world_size=1)


def test_modality_balance_ignores_text_and_padding():
    teacher = torch.ones(1, 7, 2)
    student = teacher.clone()
    student[:, :4] += 1  # four video rows
    student[:, 4:5] += 2  # one audio row
    student[:, 5:] += 1000  # text / padding ignored
    student.requires_grad_()
    layout = SimpleNamespace(video_indices=torch.arange(4), audio_indices=torch.tensor([4]))
    result = loss(student, teacher, layout, group, .001)
    torch.testing.assert_close(result, torch.tensor(5.))
    result.backward()
    assert torch.count_nonzero(student.grad[:, 5:]) == 0


def test_perfect_match_zero_and_teacher_outlier_excluded():
    teacher = torch.ones(1, 300, 2)
    teacher[:, 0, 0] = 10000
    student = teacher.clone().requires_grad_()
    layout = SimpleNamespace(video_indices=torch.arange(299), audio_indices=torch.tensor([299]))
    assert loss(student, teacher, layout, group, .001).item() == 0
    with torch.no_grad():
        student[:, 0, 0] = -10000
    assert loss(student, teacher, layout, group, .001).item() == 0
