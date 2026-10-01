# SPDX-License-Identifier: Apache-2.0
"""Inference fast path: VSA-H3 attention on the block-sparse FP4 kernel.

Opt-in with ``FASTVIDEO_H3_VSA_FP4=1`` (no-grad, single sequence-parallel
rank, ``fastvideo-kernel`` built with ``attn_qat_infer``). The selection is
VSA-H3's own: tile pooling, top-k block mask and the gated compression branch
are unchanged; only the block-sparse attention itself runs on SageAttention3's
FP4 kernel (BF16 Triton otherwise), with 64-token tiles carried by quadrant
masks on the kernel's 128x128 blocks.

The attention input is gathered into tile order once per block (one
``hidden_size``-wide pass; pad rows stay zero, so q/k/v pad rows are exactly
zero through the bias-free projections, RMSNorm and RoPE). That replaces the
generic path's concat, four tile scatters and three transposes, and lets q, k
and v share one activation quantization. The output returns to packed order
with one gather before ``to_out``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from fastvideo import envs
from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadata, _build_block_mask, _pool_tiles

_BLOCK = 128

_fp4_api: Any = None


def vsa_fp4_requested() -> bool:
    return envs.FASTVIDEO_H3_VSA_FP4


def _api() -> Any:
    global _fp4_api
    if _fp4_api is None:
        import attn_qat_infer.api as api
        _fp4_api = api
    return _fp4_api


class _TileLayout:
    """Per-step tile-order state shared by every block of one forward."""

    def __init__(self, meta: MiniMaxH3VSAMetadata, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.tile = int(meta.tile_elems)
        self.n_tiles = int(meta.variable_block_sizes.numel())
        self.rows = math.ceil(self.n_tiles * self.tile / _BLOCK) * _BLOCK
        self.untile = meta.untile_combined_index
        self.row_tile = self.untile // self.tile
        self.rotary_src = rotary_emb
        cos, sin = rotary_emb
        self.cos = cos.new_zeros((self.rows, cos.shape[-1])).index_copy_(0, self.untile, cos)
        self.sin = sin.new_zeros((self.rows, sin.shape[-1])).index_copy_(0, self.untile, sin)
        self._buf: torch.Tensor | None = None

    def gather_in(self, x: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, L, C]`` -> tile-ordered ``[B, rows, C]``; pad rows stay zero.

        The buffer is reused across blocks: pad rows are never written and
        every valid row is overwritten, and each block consumes it (q/k/v
        projections) before the next block refills it.
        """
        shape = (x.shape[0], self.rows, x.shape[-1])
        if self._buf is None or self._buf.shape != shape or self._buf.dtype != x.dtype:
            self._buf = x.new_zeros(shape)
        return self._buf.index_copy_(1, self.untile, x)


def _layout_for(meta: MiniMaxH3VSAMetadata, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> _TileLayout:
    layout = getattr(meta, "_h3_fp4_layout", None)
    if layout is None or layout.rotary_src[0] is not rotary_emb[0]:
        layout = _TileLayout(meta, rotary_emb)
        meta._h3_fp4_layout = layout  # type: ignore[attr-defined]
    return layout


def _shared_input_projections(linears: tuple[Any, ...], x: torch.Tensor) -> list[torch.Tensor]:
    """Run projections of one input, quantizing it once when all are NVFP4.

    NVFP4 activations use a unit global scale for every layer, so one
    quantized copy is exactly what each layer would have produced.
    """
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

    methods = [linear.quant_method for linear in linears]
    if not all(type(m) is NVFP4QuantizeMethod and m.wants_prequantized_input() for m in methods):
        return [linear(x)[0] for linear in linears]
    pre = methods[0].quantize_input(x)
    return [m.apply(linear, x, linear.bias, pre_quantized=pre) for m, linear in zip(methods, linears, strict=True)]


def vsa_fp4_attention(attn: Any, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor],
                      meta: MiniMaxH3VSAMetadata, use_fused_rope: bool) -> torch.Tensor:
    """Attention core for ``MiniMaxH3Attention``; returns the pre-``to_out`` ``[B, L, H*D]``."""
    api = _api()
    layout = _layout_for(meta, rotary_emb)
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    x_tiles = layout.gather_in(hidden_states)
    query, key, value = (t.unflatten(-1, (heads, dim))
                         for t in _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), x_tiles))
    if use_fused_rope:
        from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
        cos, sin = layout.cos.to(query.dtype), layout.sin.to(query.dtype)
        query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
        key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
    else:
        rope = (layout.cos, layout.sin)
        query = attn._apply_rotary_emb(attn.norm_q(query), rope)
        key = attn._apply_rotary_emb(attn.norm_k(key), rope)

    vbs = meta.variable_block_sizes
    logical = layout.n_tiles * layout.tile
    q_pooled = _pool_tiles(query[:, :logical], vbs, layout.tile)
    k_pooled = _pool_tiles(key[:, :logical], vbs, layout.tile)
    scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (dim**0.5)
    sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
    mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, sparsity, meta.exempt)
    q2k_idx, q2k_num, kv_valid, q2k_quad = api.vsa_tile_mask_to_fp4_blocks(mask, layout.tile, vbs)
    out = api.sageattn_blackwell_sparse_bshd(query, key, value, q2k_idx, q2k_num, kv_valid, q2k_quad)
    out = out.transpose(1, 2).index_select(1, layout.untile)  # [B, L, H, D], packed order

    if attn.to_gate_compress is not None and attn._gate_active():
        gate, _ = attn.to_gate_compress(hidden_states)
        v_pooled = _pool_tiles(value[:, :logical], vbs, layout.tile)
        out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled).permute(0, 2, 1, 3).to(out.dtype)
        out = out.addcmul_(out_c.index_select(1, layout.row_tile), gate.unflatten(-1, (heads, dim)))
    return out.flatten(2, 3)
