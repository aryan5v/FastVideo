# SPDX-License-Identifier: Apache-2.0
"""Exercise the production packed forward with lightweight residual blocks.

CPU tests isolate skipping, packing, and checkpointed backward. Distributed
attention and FSDP are covered by the separate real-model GPU finite gate.
"""
import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

import fastvideo.models.dits.minimax_h3 as h3
from fastvideo.train.methods.knowledge_distillation.minimax_h3_mask_recovery import sample_execution_mask


class Projection(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value, *args):
        return value + torch.tanh(self.linear(value))


class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(1, 4)

    def forward(self, value):
        return self.fc_in(value[:, None])


class NormOut(nn.Module):
    def forward(self, value, *args):
        return value


class TinyPackedH3(h3.MiniMaxH3Transformer3DModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.proj_in = Projection(4, 4)
        self.audio_proj_in = Projection(4, 4)
        self.proj_out = Projection(4, 4)
        self.audio_proj_out = Projection(4, 4)
        self.time_proj = nn.Identity()
        self.time_embedder = TimeEmbedding()
        self.adaln_basis = None
        self.norm_out = NormOut()
        self.transformer_blocks = nn.ModuleList([Residual() for _ in range(5)])

    def _refined_text(self, value):
        return value

    def _rotary_for(self, positions, dtype):
        return torch.ones(len(positions), 4), torch.zeros(len(positions), 4)


def inputs():
    return dict(hidden_states=torch.randn(1, 2, 4), audio_hidden_states=torch.randn(1, 2, 4),
                encoder_hidden_states=torch.randn(1, 1, 4), timestep=torch.zeros(1),
                timestep_indices=torch.zeros(5, dtype=torch.long), token_tags=torch.tensor([0, 1, 2, 1, 2]),
                position_ids=torch.zeros(5, 3), video_indices=torch.tensor([1, 3]),
                audio_indices=torch.tensor([2, 4]), text_indices=torch.tensor([0]))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_mask_matches_static_removal_in_forward_and_backward(monkeypatch, checkpointed):
    monkeypatch.setattr(h3, 'nvtx_range', lambda *args: nullcontext())
    torch.manual_seed(3)
    model = TinyPackedH3()
    compact = copy.deepcopy(model)
    retained = (0, 2, 4)
    compact.transformer_blocks = nn.ModuleList([compact.transformer_blocks[i] for i in retained])
    if checkpointed:
        model.transformer_blocks = nn.ModuleList([checkpoint_wrapper(block) for block in model.transformer_blocks])
        compact.transformer_blocks = nn.ModuleList([checkpoint_wrapper(block) for block in compact.transformer_blocks])
    batch = inputs()
    expected = compact(**batch)
    actual = model(**batch, block_execution_mask=(True, False, True, False, True))
    for output, target in zip(actual, expected):
        torch.testing.assert_close(output, target, rtol=0, atol=0)
    sum(output.square().sum() for output in actual).backward()
    sum(output.square().sum() for output in expected).backward()
    for index in (1, 3):
        assert all(p.grad is None for p in model.transformer_blocks[index].parameters())
    for compact_index, original_index in enumerate(retained):
        for p, q in zip(model.transformer_blocks[original_index].parameters(),
                        compact.transformer_blocks[compact_index].parameters()):
            assert p.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_all_blocks_matches_default_and_bad_masks_fail(monkeypatch):
    monkeypatch.setattr(h3, 'nvtx_range', lambda *args: nullcontext())
    model, batch = TinyPackedH3(), inputs()
    for actual, expected in zip(model(**batch, block_execution_mask=(True,) * 5), model(**batch)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for mask in ((True,) * 4, (False, True, True, True, True), (True, True, True, True, False), (1,) * 5):
        with pytest.raises(ValueError, match='block_execution_mask'):
            model(**batch, block_execution_mask=mask)


def test_mask_rng_is_replayable_and_does_not_shift_training_rng():
    rng_before = torch.get_rng_state().clone()
    kwargs = dict(num_blocks=50, retained=(0, 10, 20, 30, 40, 49), drop_probability=0.5, seed=27)
    first = sample_execution_mask(**kwargs)
    assert first == sample_execution_mask(**kwargs)
    assert first != sample_execution_mask(**(kwargs | {'seed': 28}))
    assert all(first[i] for i in kwargs['retained'])
    assert sample_execution_mask(**(kwargs | {'drop_probability': 0})) == (True,) * 50
    assert sample_execution_mask(**(kwargs | {'drop_probability': 1})) == tuple(i in kwargs['retained'] for i in range(50))
    assert torch.equal(rng_before, torch.get_rng_state())
