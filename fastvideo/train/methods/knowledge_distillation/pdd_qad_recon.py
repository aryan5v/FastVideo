# SPDX-License-Identifier: Apache-2.0
"""Layer-wise reconstruction (JetDiffusion/VSQA Eq. 6) for the FastH3 OmniRef QAD student, streamed per block.

Supervised units R, per transformer block: the attention sub-layer (q/k/v/gate/out NVFP4 projections, QK
RMSNorm, RoPE and the sparse FP4 VSA-128 attention, as deployed) and the FFN sub-layer (NVFP4 fc_in/fc_out).
Each student unit receives the frozen teacher's input to that unit (stop-gradient) and matches the teacher
unit's output, so quantization error is corrected locally and cannot accumulate into a drift away from the
teacher.

Locality makes this cheap: inside one teacher forward, right after teacher block ``b`` runs, student block
``b`` runs on the captured inputs and is backpropagated at once. There is no end-to-end student forward or
backward, no activation recompute, and memory holds one block's activations.

Attention is sparse (VSA-128, the teacher's trained selection), not dense as in the paper's warmup: our
teacher was trained sparse and the student is deployed sparse, so dense attention would supervise a
configuration that never runs.

The loss per unit is the relative squared error ``||s - t||^2 / ||t||^2`` (late FFN outputs reach ~1e6, so an
unnormalized sum would only train those), averaged over R.
"""
from __future__ import annotations

from typing import Any

import torch


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def _rel_sq(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    teacher = teacher.float()
    return (student.float() - teacher).pow(2).mean() / teacher.pow(2).mean().clamp_min(1e-20)


class _Replace:
    """Forward pre-hook replacing a sub-layer's first input with the teacher's, plus an output capture."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def pre(self, module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (self.value, *args[1:])

    def post(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.output = output


class _Capture:

    def __init__(self) -> None:
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def __call__(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.input, self.output = args[0], output


class BlockReconstruction:
    """Stream reconstruction losses for every block during one teacher forward.

    ``scale`` multiplies each block's loss before its backward (e.g. ``1 / (units x rungs x accum)``).
    Use as a context manager around the teacher's ``forward_rung``, with grad disabled for the teacher.
    """

    def __init__(self, teacher: torch.nn.Module, student: torch.nn.Module, scale: float) -> None:
        self.teacher_blocks = list(teacher.transformer_blocks)
        self.student_blocks = list(student.transformer_blocks)
        if len(self.teacher_blocks) != len(self.student_blocks):
            raise ValueError("teacher and student block counts differ")
        self.scale = scale
        self.attn_rel: list[torch.Tensor] = []
        self.ff_rel: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def __enter__(self) -> BlockReconstruction:
        for index, block in enumerate(self.teacher_blocks):
            inner = _unwrap(block)
            attn, ff = _Capture(), _Capture()
            self._handles.append(inner.attn.register_forward_hook(attn))
            self._handles.append(inner.ff.register_forward_hook(ff))
            self._handles.append(block.register_forward_hook(self._after_block(index, attn, ff)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _after_block(self, index: int, attn: _Capture, ff: _Capture):

        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            self._student_block(index, args, attn, ff)
            attn.input = attn.output = ff.input = ff.output = None

        return hook

    def _student_block(self, index: int, block_args: tuple[Any, ...], attn: _Capture, ff: _Capture) -> None:
        block = self.student_blocks[index]
        inner = _unwrap(block)
        replace_attn, replace_ff = _Replace(), _Replace()
        replace_attn.value, replace_ff.value = attn.input, ff.input
        handles = [
            inner.attn.register_forward_pre_hook(replace_attn.pre),
            inner.attn.register_forward_hook(replace_attn.post),
            inner.ff.register_forward_pre_hook(replace_ff.pre),
            inner.ff.register_forward_hook(replace_ff.post)
        ]
        try:
            with torch.enable_grad():
                # The block's own hidden input only feeds norm1 and the residual, whose results are unused; the
                # teacher attention input has the right shape. FSDP needs the call to go through the block.
                block(attn.input, *block_args[1:])
                attn_rel = _rel_sq(replace_attn.output, attn.output)
                ff_rel = _rel_sq(replace_ff.output, ff.output)
                ((attn_rel + ff_rel) * self.scale).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.attn_rel.append(attn_rel.detach())
        self.ff_rel.append(ff_rel.detach())

    def loss(self) -> torch.Tensor:
        """Mean relative error over all units of this forward (detached; already backpropagated)."""
        return (torch.stack(self.attn_rel).sum() + torch.stack(self.ff_rel).sum()) / (2 * len(self.attn_rel))

    def metrics(self, prefix: str) -> dict[str, torch.Tensor]:
        attn, ff = torch.stack(self.attn_rel), torch.stack(self.ff_rel)
        third = max(1, len(attn) // 3)
        return {
            f"{prefix}/attn_rel": attn.mean(),
            f"{prefix}/ff_rel": ff.mean(),
            f"{prefix}/attn_rel_first_third": attn[:third].mean(),
            f"{prefix}/attn_rel_last_third": attn[-third:].mean(),
            f"{prefix}/ff_rel_last_third": ff[-third:].mean()
        }


__all__ = ["BlockReconstruction"]
