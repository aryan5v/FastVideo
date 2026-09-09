"""Prompt-only updates must never regress towards placeholder zero latents."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]


def run_step(tmp_path, prompt_only, allowed=True):
    path = ROOT / 'fastvideo/train/methods/knowledge_distillation/minimax_h3_base_recovery.py'
    cls = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef))
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'single_train_step')
    @contextmanager
    def capture(model, indices):
        yield {0: torch.ones(1, 2, 1)}
    ns = dict(Any=Any, torch=torch, dist=dist, _capture_tokens=capture,
              shift_noise_amount=lambda x, shift: x,
              _euler_update=lambda x, flow, a, b: x + (b-a)*flow,
              _seam_loss=lambda *args, **kw: args[0].square().mean(),
              _normalized_mse=lambda p,t,**kw: (None,None,(p-t).square().mean()))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    block = torch.nn.Linear(1, 1)
    tb = SimpleNamespace(noise=torch.ones(1,1,1,1,1), audio_noise=torch.ones(1,2,1,1),
                         latents=torch.full((1,1,1,1,1), 17.), audio_latents=torch.full((1,2,1,1), 17.),
                         minimax_h3_layout=None, attn_metadata=None)
    calls = []
    def prepare(batch, **kw):
        calls.append(kw['latents_source'])
        return tb
    def predict(v,a,*args,**kwargs):
        calls.append('student_forward')
        p = block(torch.ones(1,2,1)).mean()
        return v*p, a*p
    group = SimpleNamespace(rank_in_group=0, ranks=[0], device_group=None)
    student = SimpleNamespace(device=torch.device('cpu'), prepare_batch=prepare,
        predict_joint_noise=predict, transformer=SimpleNamespace(transformer_blocks=[block]), sp_group=group)
    method = SimpleNamespace(student=student, teacher=SimpleNamespace(predict_joint_noise=lambda v,a,*x,**kw:(v,a)),
        method_config={'allow_prompt_only': allowed}, cuda_generator=torch.Generator().manual_seed(1),
        _student_feature_indices=[0], _teacher_feature_indices=[0], _energy_floor=.001,
        _teacher_velocity_weight=1., _feature_weight=1., _denoising_weight=1.,
        _video_velocity_weight=1., _audio_velocity_weight=1., _student_state_probability=0.,
        _grad_probe_every=0, _audio_seam_weight=1., _sample_interval=lambda points: (3, 0),
        _shared_choice=lambda upper: 0, _probe_modality_grad_share=lambda kv, ka: {},
        training_config=SimpleNamespace(checkpoint=SimpleNamespace(output_dir=str(tmp_path))))
    batch={'prompt_only': prompt_only, 'info_list':[{'id':'held-example'}]}
    result=ns['single_train_step'](method,batch,1)
    result[0]['total_loss'].backward()
    assert block.weight.grad is not None and torch.isfinite(block.weight.grad).all()
    return result,calls


def test_prompt_uses_teacher_targets_and_one_student_forward(tmp_path):
    result,calls=run_step(tmp_path,True)
    assert calls == ['zeros','student_forward']
    assert result[0]['real_video_flow_loss'].item()==0
    assert result[0]['real_audio_flow_loss'].item()==0
    assert result[2]['unique_prompts_this_dp_group']==1


def test_paired_path_keeps_separate_real_forward(tmp_path):
    result,calls=run_step(tmp_path,False)
    assert calls == ['data','student_forward','student_forward']
    assert result[0]['real_video_flow_loss'].item()>0


def test_prompt_requires_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError,match='explicit allow_prompt_only'):
        run_step(tmp_path,True,allowed=False)
