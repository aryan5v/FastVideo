# SPDX-License-Identifier: Apache-2.0
"""Hybrid MiniMax H3 attention: local window softmax plus a linear far branch.

Built on the existing H3 QKV / QK-norm / RoPE / ``to_out`` path (including
Sol-Engine fused QK-norm+RoPE and shared FP8 activation quant).

Sequence parallel has two routes:

* **Head-sharded (default when SP divides the head count).** QKV is projected on
  the local sequence shard, exchanged with the Ulysses all-to-all so every rank
  holds the full sequence but only its own head slice, both branches run on that
  slice, and the results are exchanged back.  This is the same organisation
  ``DistributedAttention_VSA`` uses.  The linear branch is per-head independent,
  so this is its only communication.
* **Replicated.** The pre-Ulysses behaviour: all-gather the packed sequence and
  run both branches on every rank.  Retained for SP=1 and for shapes the head
  split cannot cover, so dense/VSA and single-rank behaviour are unchanged.
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
from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.logger import init_logger
from fastvideo.models.dits.minimax_h3_hybrid.layout import (
    HybridSequenceLayout,
    window_bounds,
    windows_cover_all_frames,
)
from fastvideo.models.dits.minimax_h3_hybrid.linear import BidirectionalLinearBranch, OutputGate
from fastvideo.models.dits.minimax_h3_hybrid.parallel import (
    HeadShard,
    all_to_all_heads,
    all_to_all_heads_back,
    head_shard_for,
)
from fastvideo.models.dits.minimax_h3_hybrid.window import window_plan_for, window_softmax

logger = init_logger(__name__)


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
        head_sharded_sp: bool = True,
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
        # Opt-out hatch: set False to force the pre-Ulysses replicated route.
        self.head_sharded_sp = head_sharded_sp
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
        # Receipt for the dispatch tests: records which SP route actually ran.
        self.last_sp_route: str | None = None

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

    def _head_shard(self, sp_world_size: int) -> HeadShard | None:
        """The rank's head slice, or ``None`` when the replicated route applies."""
        if not self.head_sharded_sp or sp_world_size <= 1:
            return None
        # The head-sharded route narrows *parameter weights* along their head
        # axis.  That is only valid for unquantized parameters, so fall back
        # whenever any linear we slice is quantized.  NB ``ReplicatedLinear``
        # always carries a ``quant_method``; unquantized modules carry an
        # ``UnquantizedLinearMethod``, so the test is on the method's type.
        for module in self._head_sliced_linears():
            if _is_quantized(module):
                return None
        return head_shard_for(self.num_heads, sp_world_size, get_sp_parallel_rank())

    def _head_sliced_linears(self) -> list[nn.Module]:
        """Every linear whose weight the head-sharded route narrows."""
        modules: list[nn.Module] = [self.to_out_linear]
        if self.softmax_gate is not None:
            modules.append(self.softmax_gate.up)
            if self.softmax_gate.down is not None:
                modules.append(self.softmax_gate.down)
        linear = self.linear_attention
        modules.append(linear.beta_proj)
        modules.append(linear.output_gate.up)
        if linear.output_gate.down is not None:
            modules.append(linear.output_gate.down)
        return modules

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
        head_shard: HeadShard | None = None,
    ) -> torch.Tensor:
        """Window softmax over this rank's heads. Returns ``[rows, heads, d]``."""
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
            # The rectangle decomposition is request-static: fetch it once per
            # shape and reuse it for every layer instead of rebuilding it here.
            plan = window_plan_for(
                layout,
                self.window_radius,
                self.window_chunk,
                self.anchor_frames,
                query[0],
            )
            heads = window_softmax(
                query[0],
                key[0],
                value[0],
                layout,
                bounds,
                scale=self.head_dim**-0.5,
                anchor_frames=self.anchor_frames,
                plan=plan,
            ).unsqueeze(0)
        if self.softmax_gate is not None:
            gate = self.softmax_gate(hidden_states[0], head_shard=head_shard).unsqueeze(0)
            heads = heads * gate
        return heads

    # ---------------------------------------------------------------- routes
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
        shard = self._head_shard(sp_world_size)
        if shard is None:
            self.last_sp_route = "replicated"
            return self._forward_replicated(
                attn,
                hidden_states,
                rotary_emb,
                original_seq_len,
                layout,
                apply_norm_rope,
                sp_world_size,
            )
        self.last_sp_route = f"head_sharded:{shard.describe()}"
        return self._forward_head_sharded(
            attn,
            hidden_states,
            rotary_emb,
            original_seq_len,
            layout,
            apply_norm_rope,
            shard,
        )

    def _forward_replicated(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
        layout: HybridSequenceLayout,
        apply_norm_rope,
        sp_world_size: int,
    ) -> torch.Tensor:
        """Pre-Ulysses route: gather everything, run both branches everywhere."""
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
            heads = self._softmax_output(
                attn,
                working,
                query,
                key,
                value_raw,
                layout,
                bounds,
                full_cover,
                head_shard=None,
            )
            flat = heads.flatten(2, 3).type_as(working)
            out, _ = attn.to_out(flat)
        if linear_rank and linear_active:
            readout = self._linear_readout(working, (query_raw, key_raw, value_raw), layout, bounds)
            projected, _ = self.to_out_linear(readout.type_as(working))
            contrib = working.new_zeros(working.shape)
            contrib[0, layout.video_start:layout.video_end] = projected
            out = out + contrib

        if use_branch_split:
            out = get_sp_group().all_reduce(out)

        if sp_world_size > 1:
            out, _ = sequence_model_parallel_shard(out, dim=1)
        return out

    def _forward_head_sharded(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
        layout: HybridSequenceLayout,
        apply_norm_rope,
        shard: HeadShard,
    ) -> torch.Tensor:
        """Ulysses route: project locally, exchange heads, run branches sharded.

        Each rank ends up holding the full token axis with ``shard.local_heads``
        heads.  QKV projection, both branches and both output projections all run
        on 1/SP of their former work; the per-head parameter maps (``beta_proj``,
        ``alpha``, the gates, the short conv) are sliced along their contiguous
        head axis instead of being recomputed.
        """
        if hidden_states.shape[0] != 1:
            raise ValueError(f"HybridAttention supports batch size 1, got {hidden_states.shape[0]}.")
        sp_world_size = shard.world_size
        local_heads = shard.local_heads

        # The per-head parameter maps read the *full* token axis.  Their FLOPs are
        # ~1% of the layer, so gathering the activations for them is cheaper than
        # exchanging each projected scalar.
        full = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        full_rope = None
        if rotary_emb is not None:
            cos = sequence_model_parallel_all_gather_with_unpad(rotary_emb[0], original_seq_len, dim=0)
            sin = sequence_model_parallel_all_gather_with_unpad(rotary_emb[1], original_seq_len, dim=0)
            full_rope = (cos, sin)

        # The all-to-all runs over the *padded* local length, so the exchanged
        # token axis is `local_rows * SP` and the unpad is a plain truncation.
        local_rows = hidden_states.shape[1]
        padded_full = local_rows * sp_world_size
        seq_pad = padded_full - original_seq_len

        # QKV on the local shard, then Ulysses to (full sequence, local heads).
        query_raw, key_raw, value_raw = self.project_qkv(attn, hidden_states)
        query_raw = _unpad_seq(all_to_all_heads(query_raw), original_seq_len)
        key_raw = _unpad_seq(all_to_all_heads(key_raw), original_seq_len)
        value_raw = _unpad_seq(all_to_all_heads(value_raw), original_seq_len)

        # QK-norm and RoPE are per head and per token, so applying them after the
        # exchange is equivalent to applying them before it.
        query, key = apply_norm_rope(query_raw, key_raw, full_rope)

        bounds = window_bounds(layout.num_frames, self.window_radius, self.window_chunk)
        full_cover = windows_cover_all_frames(bounds, layout.num_frames)
        linear_active = not full_cover

        softmax_heads = self._softmax_output(
            attn,
            full,
            query,
            key,
            value_raw,
            layout,
            bounds,
            full_cover,
            head_shard=shard,
        )
        # Back to (local sequence, all heads) so to_out runs on the local shard.
        # Re-pad first: the reverse all-to-all expects the padded token axis.
        softmax_local = all_to_all_heads_back(_pad_seq(softmax_heads, seq_pad))
        flat = softmax_local.flatten(2, 3).type_as(hidden_states)
        out, _ = attn.to_out(flat)

        if linear_active:
            # The linear readout covers only the generated-video rows.  Place it
            # into a zeroed padded token space so the reverse exchange carries it
            # back to the right local rows; rows outside video stay zero, and
            # to_out_linear maps a zero row to a zero row.
            readout = self._linear_readout(
                full,
                (query_raw, key_raw, value_raw),
                layout,
                bounds,
                shard=shard,
            )
            readout_full = hidden_states.new_zeros((1, padded_full, local_heads, self.head_dim))
            # The branch returns the readout flattened over heads, ready for
            # to_out_linear; the exchange needs it back in [rows, heads, d].
            readout_full[0, layout.video_start:layout.video_end] = readout.view(-1, local_heads, self.head_dim)
            readout_local = all_to_all_heads_back(readout_full)
            projected, _ = self.to_out_linear(readout_local.flatten(2, 3).type_as(hidden_states))
            out = out + projected
        return out

    # ------------------------------------------------------------- linear side
    def _linear_readout(
        self,
        hidden_states: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: HybridSequenceLayout,
        bounds: list[tuple[int, int]],
        shard: HeadShard | None = None,
    ) -> torch.Tensor:
        """Per-head readout in the token space of ``hidden_states``.

        ``hidden_states`` and ``qkv_raw`` must share a token axis.  Returns
        ``[heads, rows, head_dim]``-ordered ``[rows, heads, head_dim]`` covering
        only the generated video rows; the caller places it back.
        """
        video = slice(layout.video_start, layout.video_end)
        text_hidden = text_qkv = None
        if self.linear_attention.enable_text_state and layout.text_end > layout.text_start:
            text = slice(layout.text_start, layout.text_end)
            text_hidden = hidden_states[0, text]
            text_qkv = tuple(tensor[0, text] for tensor in qkv_raw)
        return self.linear_attention(
            hidden_states[0, video],
            tuple(tensor[0, video] for tensor in qkv_raw),
            layout,
            bounds,
            skip_ends=self.anchor_frames == "both",
            text_hidden=text_hidden,
            text_qkv=text_qkv,
            head_shard=shard,
        )


def _is_quantized(module: nn.Module) -> bool:
    """True when ``module`` is actually quantized (not just carrying a method).

    ``ReplicatedLinear`` always sets ``quant_method``; an unquantized linear
    holds an ``UnquantizedLinearMethod``.  Testing ``quant_method is not None``
    would classify every layer as quantized and silently disable head sharding.
    """
    method = getattr(module, "quant_method", None)
    if method is None:
        return False
    return not isinstance(method, UnquantizedLinearMethod)


def _pad_seq(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Append ``pad`` zero rows on the token axis (dim 1)."""
    if pad <= 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))


def _unpad_seq(x: torch.Tensor, original_seq_len: int) -> torch.Tensor:
    """Truncate the token axis (dim 1) back to the unpadded length."""
    if x.shape[1] == original_seq_len:
        return x
    return x[:, :original_seq_len]
