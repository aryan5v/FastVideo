"""PDD checks use direct dense linear and analytic trajectories as oracles."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from fastvideo.layers.minimax_h3_pdd import integrate_heads, pdd_sigmas, project_heads
from fastvideo.train.methods.trajectory_matching.minimax_h3_pdd import MiniMaxH3PDDMethod


def test_initial_heads_match_parent_and_fusion_matches_explicit_predictions():
    torch.manual_seed(3)
    x = torch.randn(2, 7, 5)
    w = torch.randn(12, 5); b = torch.randn(12)
    repeated = w.repeat(32, 1); biases = b.repeat(32)
    out = project_heads(x, repeated, biases, 4, 12)
    expected = F.linear(x, w, b)
    for head in out.chunk(8, dim=-1):
        torch.testing.assert_close(head, expected)
    # Different heads detect a wrong window or a uniform-average fusion.
    repeated = repeated * torch.arange(1, 33).repeat_interleave(12)[:,None]
    for shift in (12., 3.):
        sigma = pdd_sigmas('cpu', shift); weights = sigma[5:13]-sigma[4:12]
        explicit = project_heads(x, repeated, biases, 4, 12).unflatten(-1,(8,12))
        expected = torch.einsum('...no,n->...o',explicit,(weights/weights.sum()).float())
        fused = project_heads(x,repeated,biases,4,12,weights)
        torch.testing.assert_close(fused,expected,rtol=2e-5,atol=2e-5)


def test_integrator_sign_clock_and_empty_prefix():
    clean=torch.tensor([2.,-5.]); noise=torch.tensor([4.,3.]); heads=(noise-clean).expand(8,2)
    for shift in (12.,3.):
        sigma=pdd_sigmas('cpu',shift)
        assert float(sigma[0]) == pytest.approx(.999)
        state=(1-sigma[0])*clean+sigma[0]*noise
        torch.testing.assert_close(integrate_heads(state,heads,sigma,0,0),state)
        for start in range(0,32,8):
            state=integrate_heads(state,heads,sigma,start,start+8)
        torch.testing.assert_close(state,clean,atol=2e-6,rtol=1e-6)


def test_only_sampled_head_is_supervised_and_prompt_is_carried(tmp_path, monkeypatch):
    class Model:
        def __init__(self):
            self.weight=torch.nn.Parameter(torch.full((32,),.3))
            self.device=torch.device('cpu')
            self.sp_group=SimpleNamespace(rank_in_group=0)
        def prepare_batch(self, raw, **kw):
            return SimpleNamespace(noise=torch.ones(1,1,1,1,1),audio_noise=torch.ones(1,2,1,1),attn_metadata=None)
        def predict_joint_noise(self,v,a,*args,pdd_head_window=None,**kwargs):
            if pdd_head_window is None:
                return v*.8,a*.8
            start,end=pdd_head_window
            return (self.weight[start:end].view(-1,1,1,1,1,1)*v[None],
                    self.weight[start:end].view(-1,1,1,1,1)*a[None])
    import fastvideo.train.methods.trajectory_matching.minimax_h3_pdd as mod
    monkeypatch.setattr(mod,'get_sp_group',lambda:SimpleNamespace(world_size=1))
    m=MiniMaxH3PDDMethod.__new__(MiniMaxH3PDDMethod);torch.nn.Module.__init__(m)
    m.student=Model();m.teacher=Model();m._carry=None;m.cuda_generator=torch.Generator().manual_seed(42)
    m.training_config=SimpleNamespace(checkpoint=SimpleNamespace(output_dir=str(tmp_path)))
    raw={'prompt_only':True,'info_list':[{'id':'test', 'caption':'hello'}]}
    for iteration in range(1,9):
        loss,_,metrics=m.single_train_step(raw,iteration)
        loss['total_loss'].backward()
        changed=torch.nonzero(m.student.weight.grad).flatten().tolist()
        assert changed == [metrics['pdd_supervised_head']]
        assert metrics['pdd_start']==(iteration-1)*4
        m.student.weight.grad=None
    assert m._carry is None


def test_converter_preserves_dtype_and_nonhead_values(tmp_path):
    from safetensors.torch import save_file,load_file
    import json
    script=Path(__file__).resolve().parents[4]/'scripts/checkpoint_conversion/expand_minimax_h3_pdd.py'
    spec=importlib.util.spec_from_file_location('expand_pdd',script); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    src=tmp_path/'src';(src/'transformer').mkdir(parents=True)
    (src/'transformer/config.json').write_text(json.dumps({'num_layers':42}))
    state={key:torch.arange(6,dtype=torch.bfloat16).reshape(2,3) for key in ('proj_out.weight','audio_proj_out.weight')}
    state.update({key:torch.arange(2,dtype=torch.bfloat16) for key in ('proj_out.bias','audio_proj_out.bias')})
    state['body.weight']=torch.randn(3,3)
    save_file(state,src/'transformer/model.safetensors')
    dst=tmp_path/'dst';mod.expand(src,dst)
    result=load_file(dst/'transformer/model.safetensors')
    for key,value in state.items():
        if key=='body.weight': assert torch.equal(value,result[key])
        else:
            assert result[key].dtype==value.dtype
            for head in result[key].chunk(32): assert torch.equal(head,value)
    with pytest.raises(FileExistsError):mod.expand(src,dst)


def test_production_transformer_routes_head_windows_and_fusion(monkeypatch):
    from contextlib import nullcontext
    import copy
    from fastvideo.tests.transformers.test_minimax_h3_execution_mask import TinyPackedH3, inputs
    import fastvideo.models.dits.minimax_h3 as h3
    from fastvideo.layers.minimax_h3_pdd import H3PDDLinear
    monkeypatch.setattr(h3,'nvtx_range',lambda *a:nullcontext())
    parent=TinyPackedH3();parent.pdd_steps=None
    student=copy.deepcopy(parent);student.pdd_steps=32
    # nn.Linear container exercises the actual production H3PDDLinear.forward.
    class Head(torch.nn.Linear):
        forward=H3PDDLinear.forward
    for name in ('proj_out','audio_proj_out'):
        original=getattr(parent,name);head=Head(4,4*32)
        with torch.no_grad():
            head.weight.copy_(original.weight.repeat(32,1));head.bias.copy_(original.bias.repeat(32))
        setattr(student,name,head)
    batch=inputs();expected=parent(**batch)
    actual=student(**batch,pdd_head_window=(8,16))
    fused=student(**batch,pdd_head_window=(8,16),pdd_fuse=True)
    for target,wide,mean in zip(expected,actual,fused):
        for single in wide.chunk(8,dim=-1): torch.testing.assert_close(single,target)
        torch.testing.assert_close(mean,target)
    sum(t.square().sum() for t in actual).backward()
    assert student.proj_out.weight.grad[:8*4].count_nonzero()==0
    assert student.proj_out.weight.grad[8*4:16*4].count_nonzero()>0
    with pytest.raises(ValueError,match='explicit head window'): student(**batch)
