# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 tile-128 fine attention on the FA4-FP4 sm_100/sm_103 kernels (FastH3 OmniRef "mixed" numerics).

``FA4VSAFineAttention`` plugs into ``MiniMaxH3VSAImpl.fine_attention_override``
exactly like the QAD emulator (``fp4_vsa_qat.FP4VSAFineAttention``): VSA-H3's
selection, mask and gated compression branch run unchanged in BF16; only the
block-sparse attention over the selected tiles is replaced. Numerics per mode:

* ``mixmx`` (deployed): NVFP4 Q/K for the video query tiles, MXFP8 Q/K for the
  dense-prefix query tiles (text, reference, audio;
  ``MiniMaxH3VSAMetadata.num_prefix_tiles``), BF16 P.V, K smoothing. The QAD
  emulator's ``prefix_precision="mxfp8"`` is the same computation.
* ``qk``: NVFP4 Q/K for every query tile, BF16 P.V, K smoothing.
* ``bf16``: BF16 Q/K (FA4's dense kernel with the same block lists); a layout check.

Quantization is FlashInfer's: ``nvfp4_quantize(x, 1.0)`` (E4M3 scale ``amax / 6``
per 16 channels, bit-identical to the emulator's ``nvfp4_fake_quant_lastdim``)
and ``mxfp8_quantize`` (E8M0 scale ``2**ceil(log2(amax / 448))`` per 32).

Sparsity: FA4 schedules 256-row Q blocks (two 128-row stages) over 128-key
blocks. Each pair of VSA query tiles runs as one Q block over the union of the
two tiles' key lists. Key blocks that both tiles selected and that hold 128
real tokens are "full" blocks; every other listed block runs through a
``mask_mod`` that keeps exactly the per-Q128 selection and the valid keys of
partial tiles (``(kv % 128) < limit[q_tile, k_tile]``). Partial tiles are masked
in place, not repacked, so the softmax sees the same key set as the emulator.

The two query groups are separate kernel calls (MXFP8 prefix pairs, NVFP4 video
pairs); a pair that straddles the boundary runs in both and each tile keeps the
output of its own group.

Requires the FA4-FP4 ``flash_attn.cute`` package (``hao-ai-lab/flash-attention-fp4``
``fp4_B300@0633a0ef``) with ``scripts/benchmarks/fasth3_omniref_b200/flash_attention_fp4_omniref.patch``
applied: cutlass-dsl 4.8 compatibility (so it imports next to FlashInfer), the FP4
kernel's block-sparse paths, and the column-limit mask this module relies on.
"""
from __future__ import annotations

import functools
import math
from typing import Any

import torch

TILE = 128
PAIR = 2 * TILE
MODES: dict[str, tuple[str, str]] = {
    # mode -> (prefix query tiles, video query tiles)
    "mixmx": ("mxfp8", "nvfp4"),
    "qk": ("nvfp4", "nvfp4"),
    "bf16": ("bf16", "bf16"),
}


@functools.cache
def _fa4() -> tuple[Any, Any, Any]:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    return _flash_attn_fwd, BlockSparseTensorsTorch, _vsa_mask_mod()


def _vsa_mask_mod() -> Any:
    """``mask_mod`` keeping key ``n`` for query ``m`` iff ``n % 128 < limit[b, h, m // 128, n // 128]``."""
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def vsa_tile_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        limit = aux_tensors[0]
        tile = utils.scalar_to_ssa(TILE, cutlass.Int32)
        q_tile = m_idx // tile
        k_tile = n_idx // tile
        keep = utils.scalar_to_ssa(limit[batch[0], head[0], q_tile[0], k_tile[0]], cutlass.Int32)
        return (n_idx - k_tile * tile) < keep

    # Our FA4 build applies this mask as one column limit per (query tile, key block) (``mask.py``); the
    # per-element body above is the reference semantics and the fallback on builds without that path.
    vsa_tile_mask.vsa_col_limit = True
    return vsa_tile_mask


def _quantize(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``[B, S, H, D]`` BF16 -> FA4 token-major operand and its swizzled scale factors (groups along D)."""
    if fmt == "bf16":
        return x, None
    from flashinfer.quantization import SfLayout, mxfp8_quantize, nvfp4_quantize

    b, s, h, d = x.shape
    flat = x.reshape(b * s, h * d)
    if fmt == "nvfp4":
        vec = 16
        unit = torch.ones(1, device=x.device, dtype=torch.float32)
        data, sf = nvfp4_quantize(flat, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        data = data.view(torch.uint8).reshape(b, s, h, d // 2).view(torch.float4_e2m1fn_x2)
        sf = sf.view(torch.float8_e4m3fn)
    elif fmt == "mxfp8":
        vec = 32
        data, sf = mxfp8_quantize(flat, True)
        if data.dtype == torch.uint8:
            data = data.view(torch.float8_e4m3fn)
        data = data.reshape(b, s, h, d)
        sf = sf.view(torch.float8_e8m0fnu)
    else:
        raise ValueError(f"unknown FA4 operand format {fmt!r}")
    rest_k = (d // vec) // 4
    sf = sf.reshape(b, s // TILE, h, rest_k, 32, 4, 4)
    sf = sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return data, sf


def _pair_lists(limit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q256 block lists from per-tile key limits ``[B, H, 2M, T]`` (0 = not listed).

    Returns ``(mask_cnt, mask_idx, full_cnt, full_idx)``: ``[B, H, M]`` counts and ``[B, H, M, T]`` int32 key
    block indices packed left. Full blocks are listed by both tiles with 128 valid keys; the rest need the mask.
    """
    b, h, two_m, n = limit.shape
    pair = limit.view(b, h, two_m // 2, 2, n)
    union = (pair > 0).any(dim=3)
    full = (pair == TILE).all(dim=3)
    partial = union & ~full
    order = torch.arange(n, device=limit.device, dtype=torch.int32)

    def pack(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = selected.sum(dim=-1, dtype=torch.int32)
        keys = torch.where(selected, order, n)
        idx = keys.sort(dim=-1).values.clamp_max(n - 1).to(torch.int32)
        return count.contiguous(), idx.contiguous()

    mask_cnt, mask_idx = pack(partial)
    full_cnt, full_idx = pack(full)
    return mask_cnt, mask_idx, full_cnt, full_idx


def _groups(mode: str, n_tiles: int, num_prefix_tiles: int) -> list[tuple[str, int, int, int, int]]:
    """``(format, first tile, end tile, first owned tile, end owned tile)`` per kernel call; calls start on a pair."""
    prefix_fmt, video_fmt = MODES[mode]
    prefix = min(max(int(num_prefix_tiles), 0), n_tiles)
    if prefix_fmt == video_fmt or prefix in (0, n_tiles):
        return [(video_fmt if prefix < n_tiles else prefix_fmt, 0, n_tiles, 0, n_tiles)]
    # Video first: the prefix call then overwrites the straddling pair, and the one video tile in it is restored.
    return [(video_fmt, prefix - prefix % 2, n_tiles, prefix, n_tiles),
            (prefix_fmt, 0, min(n_tiles, prefix + prefix % 2), 0, prefix)]


def fa4_vsa_attention(query: torch.Tensor,
                      key: torch.Tensor,
                      value: torch.Tensor,
                      tile_mask: torch.Tensor,
                      variable_block_sizes: torch.Tensor,
                      num_prefix_tiles: int,
                      mode: str = "mixmx",
                      smooth_k: bool = True) -> torch.Tensor:
    """Block-sparse VSA-128 attention on FA4; ``[B, S, H, D]`` BF16 in (tile-ordered) and out.

    ``tile_mask`` is VSA-H3's ``[B, H, T, T]`` selection, ``variable_block_sizes`` the ``[T]`` real tokens per tile.
    """
    if mode not in MODES:
        raise ValueError(f"FA4 VSA mode must be one of {sorted(MODES)}, got {mode!r}")
    flash_attn_fwd, block_sparse_tensors, mask_mod = _fa4()
    b, s, h, d = query.shape
    n_tiles = tile_mask.shape[-1]
    if s != n_tiles * TILE:
        raise ValueError(f"tiled length {s} != {n_tiles} tiles x {TILE}")
    if smooth_k:
        # Same op as the emulator (per-head token mean over every tiled row, pad rows included, in BF16).
        key = (key.transpose(1, 2) - key.transpose(1, 2).mean(dim=-2, keepdim=True)).transpose(1, 2)
    key, value = key.contiguous(), value.contiguous()
    vbs = variable_block_sizes.to(device=query.device, dtype=torch.int32)
    limit = (tile_mask & (vbs > 0)) * vbs  # [B, H, T, T] int32: 0 = not listed, else the key tile's real tokens
    limit = limit.to(torch.int32)
    if n_tiles % 2:  # the last Q256 block holds one real tile; its partner rows are out of range
        limit = torch.nn.functional.pad(limit, (0, 0, 0, 1))
    lists = _pair_lists(limit)
    out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(d)
    keys: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for fmt, t0, t1, _own0, own1 in _groups(mode, n_tiles, num_prefix_tiles):
        if fmt not in keys:
            keys[fmt] = _quantize(key, fmt)
        k_op, sf_k = keys[fmt]
        q_op, sf_q = _quantize(query[:, t0 * TILE:t1 * TILE], fmt)
        p0, p1 = t0 // 2, (t1 + 1) // 2
        mask_cnt, mask_idx, full_cnt, full_idx = (t[:, :, p0:p1] for t in lists)
        sparse = block_sparse_tensors(mask_block_cnt=mask_cnt.contiguous(),
                                      mask_block_idx=mask_idx.contiguous(),
                                      full_block_cnt=full_cnt.contiguous(),
                                      full_block_idx=full_idx.contiguous(),
                                      block_size=(PAIR, TILE))
        if own1 < t1:  # the prefix call also covers the straddling pair's video tile: keep that tile's rows
            saved = out[:, own1 * TILE:t1 * TILE].clone()
        target = out[:, t0 * TILE:t1 * TILE]
        result = flash_attn_fwd(q_op,
                                k_op,
                                value,
                                softmax_scale=scale,
                                mask_mod=mask_mod,
                                block_sparse_tensors=sparse,
                                aux_tensors=[limit[:, :, t0:t0 + 2 * (p1 - p0)].contiguous()],
                                out=target if target.is_contiguous() else None,
                                mSFQ=sf_q,
                                mSFK=sf_k)
        if not target.is_contiguous():
            target.copy_(result[0] if isinstance(result, tuple) else result)
        if own1 < t1:
            out[:, own1 * TILE:t1 * TILE] = saved
    return out


class FA4VSAFineAttention:
    """``fine_attention_override`` callable: ``(q, k, v, mask, metadata) -> [B, S, H, D]``."""

    def __init__(self, mode: str = "mixmx", smooth_k: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"FA4 VSA mode must be one of {sorted(MODES)}, got {mode!r}")
        self.mode = mode
        self.smooth_k = smooth_k

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor,
                 metadata: Any) -> torch.Tensor:
        if int(metadata.tile_elems) != TILE:
            raise ValueError(f"FA4 VSA models 128-token tiles only, got {metadata.tile_elems}")
        return fa4_vsa_attention(query,
                                 key,
                                 value,
                                 mask,
                                 metadata.variable_block_sizes,
                                 int(getattr(metadata, "num_prefix_tiles", 0) or 0),
                                 mode=self.mode,
                                 smooth_k=self.smooth_k)


def install_fa4_vsa_attention(transformer: torch.nn.Module, mode: str | None) -> int:
    """Route every VSA-H3 layer's fine attention through FA4 (``None`` restores the default kernels)."""
    from fastvideo.attention.backends.fp4_vsa_qat import vsa_impls

    override = FA4VSAFineAttention(mode) if mode is not None else None
    impls = vsa_impls(transformer)
    for impl in impls:
        impl.fine_attention_override = override
    return len(impls)


__all__ = ["FA4VSAFineAttention", "MODES", "fa4_vsa_attention", "install_fa4_vsa_attention"]
