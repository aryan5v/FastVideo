# SPDX-License-Identifier: Apache-2.0
"""Hybrid MiniMax H3 attention: local window softmax plus a linear far branch.

Built on the existing H3 QKV / QK-norm / RoPE / ``to_out`` path (including
Sol-Engine fused QK-norm+RoPE and shared FP8 activation quant). Sequence
parallel either all-gathers into this module or, with ``branch_parallel`` and
two ranks, splits softmax vs linear across the pair and all-reduces.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import (
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    model_parallel_is_initialized,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.models.dits.minimax_h3_hybrid.layout import (
    HybridSequenceLayout,
    window_bounds,
    windows_cover_all_frames,
)
from fastvideo.models.dits.minimax_h3_hybrid.linear import BidirectionalLinearBranch, OutputGate
from fastvideo.models.dits.minimax_h3_hybrid.window import window_softmax


def _maybe_prequantized_linear(
    layer: ReplicatedLinear,
    hidden_states: torch.Tensor,
    pre_quantized: tuple[torch.Tensor, torch.Tensor, Any] | None,
) -> torch.Tensor:
    """Reuse one activation quantisation across Q/K/V when the linear wants it."""
    quant_method = getattr(layer, "quant_method", None)
    if pre_quantized is not None and quant_method is not None:
        wants = getattr(quant_method, "wants_prequantized_input", None)
        if callable(wants) and wants():
            return quant_method.apply(layer, hidden_states, pre_quantized=pre_quantized)
    return layer(hidden_states)[0]


class HybridAttention(nn.Module):
    """Drop-in attention body used by ``MiniMaxH3Attention`` when hybrid is on.

    Owns the extra parameters (softmax gate, linear branch, ``to_out_linear``).
    Reuses the parent attention's QKV, norms, RoPE, fused kernels, and ``to_out``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        window_radius: int = 1,
        window_chunk: int = 5,
        anchor_frames: str = "both",
        delta_rule: str = "vdn_solve",
        enable_softmax_gate: bool = True,
        enable_text_state: bool = True,
        short_conv_targets: tuple[str, ...] = ("k", "v"),
        branch_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_radius = window_radius
        self.window_chunk = window_chunk
        self.anchor_frames = anchor_frames
        self.branch_parallel = branch_parallel
        self.enable_softmax_gate = enable_softmax_gate
        self.linear_attention = BidirectionalLinearBranch(
            hidden_size,
            num_heads,
            head_dim,
            delta_rule=delta_rule,
            short_conv_targets=short_conv_targets,
            enable_text_state=enable_text_state,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attention",
        )
        self.to_out_linear = ReplicatedLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out_linear",
        )
        self.softmax_gate = (OutputGate(
            hidden_size,
            num_heads,
            init_value=0.99,
            init="constant",
            quant_config=quant_config,
            prefix=f"{prefix}.softmax_gate",
        ) if enable_softmax_gate else None)

    def project_qkv(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_quantized = None
        quant_method = getattr(attn.to_q, "quant_method", None)
        wants = getattr(quant_method, "wants_prequantized_input", None) if quant_method is not None else None
        if callable(wants) and wants():
            pre_quantized = quant_method.quantize_input(hidden_states.reshape(-1, hidden_states.shape[-1]))
        query = _maybe_prequantized_linear(attn.to_q, hidden_states, pre_quantized)
        key = _maybe_prequantized_linear(attn.to_k, hidden_states, pre_quantized)
        value = _maybe_prequantized_linear(attn.to_v, hidden_states, pre_quantized)
        return (
            query.unflatten(-1, (self.num_heads, self.head_dim)),
            key.unflatten(-1, (self.num_heads, self.head_dim)),
            value.unflatten(-1, (self.num_heads, self.head_dim)),
        )

    def _softmax_output(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layout: HybridSequenceLayout,
        bounds: list[tuple[int, int]],
        full_cover: bool,
    ) -> torch.Tensor:
        if full_cover:
            # Already all-gathered when SP>1; do not re-enter DistributedAttention.
            scale = self.head_dim**-0.5
            heads = F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                scale=scale,
                dropout_p=0.0,
                is_causal=False,
            ).permute(0, 2, 1, 3)
        else:
            heads = window_softmax(
                query[0],
                key[0],
                value[0],
                layout,
                bounds,
                scale=self.head_dim**-0.5,
                anchor_frames=self.anchor_frames,
            ).unsqueeze(0)
        if self.softmax_gate is not None:
            gate = self.softmax_gate(hidden_states[0]).unsqueeze(0)
            heads = heads * gate
        flat = heads.flatten(2, 3).type_as(hidden_states)
        out, _ = attn.to_out(flat)
        return out

    def _linear_output(
        self,
        hidden_states: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: HybridSequenceLayout,
        bounds: list[tuple[int, int]],
    ) -> torch.Tensor:
        video = slice(layout.video_start, layout.video_end)
        text_hidden = text_qkv = None
        if self.linear_attention.enable_text_state and layout.text_end > layout.text_start:
            text = slice(layout.text_start, layout.text_end)
            text_hidden = hidden_states[0, text]
            text_qkv = tuple(tensor[0, text] for tensor in qkv_raw)
        readout = self.linear_attention(
            hidden_states[0, video],
            tuple(tensor[0, video] for tensor in qkv_raw),
            layout,
            bounds,
            skip_ends=self.anchor_frames == "both",
            text_hidden=text_hidden,
            text_qkv=text_qkv,
        )
        projected, _ = self.to_out_linear(readout.type_as(hidden_states))
        contrib = hidden_states.new_zeros(hidden_states.shape)
        contrib[0, video] = projected
        return contrib

    def forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
        layout: HybridSequenceLayout,
        apply_norm_rope,
    ) -> torch.Tensor:
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        working = hidden_states
        working_rope = rotary_emb
        if sp_world_size > 1:
            working = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
            if rotary_emb is not None:
                cos = sequence_model_parallel_all_gather_with_unpad(rotary_emb[0], original_seq_len, dim=0)
                sin = sequence_model_parallel_all_gather_with_unpad(rotary_emb[1], original_seq_len, dim=0)
                working_rope = (cos, sin)

        if working.shape[0] != 1:
            raise ValueError(f"HybridAttention supports batch size 1, got {working.shape[0]}.")

        query_raw, key_raw, value_raw = self.project_qkv(attn, working)
        query, key = apply_norm_rope(query_raw, key_raw, working_rope)
        bounds = window_bounds(layout.num_frames, self.window_radius, self.window_chunk)
        full_cover = windows_cover_all_frames(bounds, layout.num_frames)
        linear_active = not full_cover

        rank = get_sp_parallel_rank() if sp_world_size > 1 else 0
        use_branch_split = self.branch_parallel and sp_world_size == 2 and linear_active
        softmax_rank = (not use_branch_split) or rank == 0
        linear_rank = (not use_branch_split) or rank == 1

        out = working.new_zeros(working.shape)
        if softmax_rank:
            out = self._softmax_output(
                attn,
                working,
                query,
                key,
                value_raw,
                layout,
                bounds,
                full_cover,
            )
        if linear_rank and linear_active:
            out = out + self._linear_output(working, (query_raw, key_raw, value_raw), layout, bounds)

        if use_branch_split:
            out = get_sp_group().all_reduce(out)

        if sp_world_size > 1:
            out, _ = sequence_model_parallel_shard(out, dim=1)
        return out
