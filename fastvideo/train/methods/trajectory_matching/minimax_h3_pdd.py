# SPDX-License-Identifier: Apache-2.0
"""Compact H3 Parallel Decoding Distillation, dense grid32 / block4..8.

Adapted from FastVideo H3 V23 bf16199660d9 (FastGen PDD port). One carried
trajectory, one sampled head / Euler teacher target per update, modality MSE.
Training carries eight calls (stride4); inference fuses eight heads for4calls.
Unlike the reference64-GPU VSA run, this experiment uses matched SDPA/SP4,
accumulation1 and an audited prompt index. No DMD critic or real-latent loss.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.distributed import get_sp_group
from fastvideo.layers.minimax_h3_pdd import integrate_heads, pdd_sigmas
from fastvideo.train.methods.knowledge_distillation.minimax_h3_base_recovery import MiniMaxH3BaseRecoveryMethod
from fastvideo.train.utils.h3_prompt_coverage import record_prompt_use


class MiniMaxH3PDDMethod(MiniMaxH3BaseRecoveryMethod):
    """Reuse checkpoint/optimizer/update guards; replace the recovery objective."""

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if self.student.transformer.pdd_steps != 32 or self.teacher.transformer.pdd_steps is not None:
            raise ValueError("PDD requires widened grid32 student and single-head Base teacher")
        if any(m.attention_backend_name != 'TORCH_SDPA' for m in role_models.values()):
            raise ValueError('The compact PDD experiment requires matched SDPA')
        if int(self.training_config.loop.gradient_accumulation_steps) != 1:
            raise ValueError('Compact PDD supports one carry slot, accumulation1')
        self._carry: Any = None

    def on_checkpoint_loaded(self, iteration: int) -> None:
        super().on_checkpoint_loaded(iteration)
        self._carry = None

    def single_train_step(self, batch: dict[str, Any], iteration: int) -> Any:
        if not batch.get('prompt_only', False):
            raise ValueError('PDD exploration requires prompt-only geometry, never zero-latent targets')
        raw = batch if self._carry is None else self._carry['raw']
        tb = self.student.prepare_batch(raw, generator=self.cuda_generator, latents_source='zeros')
        device = self.student.device
        grids = [pdd_sigmas(device, shift) for shift in (12., 3.)]
        if self._carry is None:
            start = 0
            states = (tb.noise.permute(0, 2, 1, 3, 4).float() * float(grids[0][0]),
                      tb.audio_noise.float() * float(grids[1][0]))
        else:
            start, states = self._carry['start'], self._carry['states']
        end = min(start + 8, 32)
        vt, at = [(1 - grid[start]).reshape(1) for grid in grids]
        heads = self.student.predict_joint_noise(*states,
                                                 vt,
                                                 at,
                                                 tb,
                                                 conditional=True,
                                                 attn_kind='dense',
                                                 pdd_head_window=(start, end))
        k_tensor = torch.randint(start, end, (1, ), device=device, generator=self.cuda_generator)
        group = get_sp_group()
        if group.world_size > 1:
            group.broadcast(k_tensor, src=0)
        k = int(k_tensor)
        with torch.no_grad():
            teacher_states = tuple(
                integrate_heads(x, h, g, start, k) for x, h, g in zip(states, heads, grids, strict=True))
            target = self.teacher.predict_joint_noise(*teacher_states, (1 - grids[0][k]).reshape(1),
                                                      (1 - grids[1][k]).reshape(1),
                                                      tb,
                                                      conditional=True,
                                                      attn_kind='dense')
            next_start = start + 4
            self._carry = None if next_start == 32 else {
                'raw':
                raw,
                'start':
                next_start,
                'states':
                tuple(
                    integrate_heads(x, h, g, start, next_start).detach()
                    for x, h, g in zip(states, heads, grids, strict=True))
            }
        video_loss, audio_loss = [
            F.mse_loss(h[k - start].float(), t.float()) for h, t in zip(heads, target, strict=True)
        ]
        total = video_loss + audio_loss
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite PDD objective')
        return {
            'total_loss': total,
            'pdd_video': video_loss,
            'pdd_audio': audio_loss
        }, {
            '_fv_backward': (vt, tb.attn_metadata)
        }, {
            'pdd_start': start,
            'pdd_supervised_head': k,
            **record_prompt_use(self, raw, iteration)
        }
