# SPDX-License-Identifier: Apache-2.0
"""Held-out rollout correctness independent of real-model GPU integration."""
from types import SimpleNamespace

import pytest
import torch

import fastvideo.train.methods.knowledge_distillation.minimax_h3_mask_recovery as recovery


class TinyJointModel:
    device = torch.device('cpu')

    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.seen_masks = []

    def prepare_batch(self, value, *, generator, latents_source):
        return SimpleNamespace(noise=torch.randn(1, 2, 1, 1, 1, generator=generator),
                               audio_noise=torch.randn(1, 2, 2, 1, 1, generator=generator))

    def predict_joint_noise(self, video, audio, *args, block_execution_mask=None, **kwargs):
        self.seen_masks.append(block_execution_mask)
        return torch.ones_like(video) * self.multiplier, torch.ones_like(audio) * self.multiplier


class Data:
    def __getitems__(self, indices):
        return {'index': indices[0]}


def method(tmp_path, monkeypatch, multiplier=1):
    monkeypatch.setattr(recovery, 'get_world_rank', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *args: 0)
    obj = object.__new__(recovery.MiniMaxH3MaskRecoveryMethod)
    obj.student, obj.teacher = TinyJointModel(multiplier), TinyJointModel()
    obj._student_attn_kind = obj._teacher_attn_kind = 'dense'
    obj._validation_data = Data()
    obj._validation_every = 25
    obj._validation_samples = 2
    obj._execution_mask = (True, True, True)
    obj._final_mask = (True, False, True)
    obj._retained = (0, 2)
    obj._energy_floor = 1e-3
    obj._mask_mode = 'annealed_skip'
    obj.training_config = SimpleNamespace(checkpoint=SimpleNamespace(output_dir=str(tmp_path)))
    return obj


def test_heldout_uses_final_mask_preserves_rng_and_scores_perfect_teacher_zero(tmp_path, monkeypatch):
    obj = method(tmp_path, monkeypatch)
    rng = torch.get_rng_state().clone()
    metrics = obj.on_validation_begin(0)
    assert obj._execution_mask == (True, True, True)
    assert all(mask == (True, False, True) for mask in obj.student.seen_masks)
    assert all(mask is None for mask in obj.teacher.seen_masks)
    assert torch.equal(rng, torch.get_rng_state())
    errors = [value for name, value in metrics.items() if 'interval' in name or 'endpoint' in name]
    assert len(errors) == 10 and all(value == 0 for value in errors)
    assert (tmp_path / 'heldout_metrics.jsonl').is_file()
    assert obj.on_validation_begin(1) == {}


def test_endpoint_detects_joint_drift_and_final_mask_restores_on_failure(tmp_path, monkeypatch):
    obj = method(tmp_path, monkeypatch, multiplier=2)
    metrics = obj.on_validation_begin(0)
    for modality in ('video', 'audio'):
        assert metrics[f'validation/{modality}_closed_loop_endpoint'] > 0
        for interval in range(4):
            assert metrics[f'validation/{modality}_teacher_state_interval{interval}'] == pytest.approx(1.0)

    def fail(*args, **kwargs):
        raise RuntimeError('injected forward failure')

    monkeypatch.setattr(obj.teacher, 'predict_joint_noise', fail)
    with pytest.raises(RuntimeError, match='injected forward failure'):
        obj.on_validation_begin(25)
    assert obj._execution_mask == (True, True, True)
