"""Independent analytical checks for joint clocks, DMD gradient and role isolation."""
from types import SimpleNamespace

import pytest
import torch
from fastvideo.train.methods.distribution_matching.minimax_h3_joint_dmd2 import (
    MiniMaxH3JointDMD2Method, warp_sigma, clean_estimate, mix_state, advance_ode, distribution_loss)


def test_joint_score_clock_matches_effective_shifts():
    u = torch.tensor([.001,.2,.7,.999],dtype=torch.float64)
    base=warp_sigma(u,1/2.4,.999)
    assert torch.allclose(warp_sigma(base,12,.999),warp_sigma(u,5,.999),atol=1e-14)
    assert torch.allclose(warp_sigma(base,3,.999),warp_sigma(u,1.25,.999),atol=1e-14)
    assert not torch.equal(warp_sigma(base,12,.999),warp_sigma(base,3,.999))


def test_velocity_sign_clean_reconstruction_and_ode():
    clean=torch.tensor([2.,-3.],dtype=torch.float64)
    noise=torch.tensor([7.,5.],dtype=torch.float64)
    for shift in (12.,3.):
        sigma=warp_sigma(torch.tensor([.749]),shift,1.)
        nxt=warp_sigma(torch.tensor([.5]),shift,1.)
        state=mix_state(clean,noise,sigma)
        recovered=clean_estimate(state,noise-clean,sigma)
        assert torch.allclose(recovered,clean,atol=1e-12)
        assert torch.allclose(advance_ode(state,clean,sigma,nxt),mix_state(clean,noise,nxt),atol=1e-12)


def test_distribution_gradient_direction_and_modality_mean():
    gen=torch.tensor([3.,4.],requires_grad=True)
    real=torch.tensor([1.,2.]);fake=torch.tensor([2.,3.])
    distribution_loss(gen,fake,real).backward()
    assert torch.allclose(gen.grad,torch.full((2,),1/(2+1e-6)/2))
    assert distribution_loss(gen.detach(),real,real)==0
    with pytest.raises(RuntimeError,match='Nonfinite'):
        distribution_loss(gen,torch.full((2,),float('nan')),real)


def test_carried_prompt_and_alternating_optimizer_roles(tmp_path):
    class Model:
        def __init__(self,value):
            self.transformer=torch.nn.Linear(1,1,bias=False)
            self.transformer.weight.data.fill_(value)
            self.device=torch.device('cpu')
            self.sp_group=SimpleNamespace(rank_in_group=0)
        def prepare_batch(self,batch,**kw):
            return SimpleNamespace(noise=torch.ones(1,1,1,1,1),audio_noise=torch.ones(1,2,1,1),attn_metadata=None)
        def predict_joint_noise(self,v,a,*args,**kwargs):
            return v*self.transformer.weight[0,0],a*self.transformer.weight[0,0]
        def backward(self,loss,ctx,**kw):
            loss.backward()
    m=MiniMaxH3JointDMD2Method.__new__(MiniMaxH3JointDMD2Method)
    torch.nn.Module.__init__(m)
    m.student=Model(.3);m.critic=Model(.4);m.teacher=Model(.7)
    m._role_models={'student':m.student,'critic':m.critic,'teacher':m.teacher}
    m.method_config={'generator_update_interval':5}
    m.training_config=SimpleNamespace(checkpoint=SimpleNamespace(output_dir=str(tmp_path)))
    m.cuda_generator=torch.Generator().manual_seed(99);m._carry=None
    m._student_optimizer=torch.optim.SGD(m.student.transformer.parameters(),lr=.01)
    m._critic_optimizer=torch.optim.SGD(m.critic.transformer.parameters(),lr=.01)
    for iteration in range(1,6):
        for role in m._role_models.values():role.transformer.zero_grad(set_to_none=True)
        batch={'prompt_only':True,'info_list':[{'id':str(iteration)}]}
        losses,outputs,metrics=m.single_train_step(batch,iteration)
        assert metrics['rollout_rung']==(iteration-1)%4
        assert metrics['new_prompt_adopted']==int(iteration in (1,5))
        assert outputs['role']==('student' if iteration==5 else 'critic')
        m.backward(losses,outputs)
        active=m._role_models[outputs['role']]
        inactive=m.critic if iteration==5 else m.student
        assert active.transformer.weight.grad is not None
        assert inactive.transformer.weight.grad is None
        assert m.teacher.transformer.weight.grad is None
        m.get_optimizers(iteration)[0].step()
    import json
    ids=[json.loads(l)['id'] for l in (tmp_path/'consumed_prompts_rank0.jsonl').read_text().splitlines()]
    assert ids==['1','1','1','1','5']
