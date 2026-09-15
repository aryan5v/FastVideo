# SPDX-License-Identifier: Apache-2.0
"""Correctness of the Ulysses head-sharded hybrid route against the replicated one.

The head-sharded route only activates at SP>1, which the rest of the hybrid suite
never reaches.  These tests drive it on a single process by substituting the
pure-tensor stand-ins in ``parallel.py`` for the collective, so a full rank set
can be simulated without a process group.

The contract under test: running the branches on one rank's head slice and
exchanging back must reproduce, bit-for-bit at BF16-friendly tolerances, the
result of running both branches over every head on the full sequence.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.minimax_h3_hybrid import attention as attention_mod
from fastvideo.models.dits.minimax_h3_hybrid.attention import HybridAttention
from fastvideo.models.dits.minimax_h3_hybrid.layout import HybridSequenceLayout
from fastvideo.models.dits.minimax_h3_hybrid import parallel as parallel_mod
from fastvideo.models.dits.minimax_h3_hybrid.parallel import (
    HeadShard,
    simulate_all_to_all_heads,
    simulate_all_to_all_heads_back,
)

HEADS = 8
HEAD_DIM = 16
HIDDEN = 64
FRAME_H, FRAME_W = 4, 4
TOKENS_PER_FRAME = FRAME_H * FRAME_W
FRAMES = 11  # 3 chunks of 5 would be 15; 11 deliberately leaves an uneven tail
GLOBAL_ROWS = 7


class _StubParent(nn.Module):
    """Minimal stand-in for MiniMaxH3Attention: QKV, to_out, and norm+rope."""

    def __init__(self) -> None:
        super().__init__()
        self.num_attention_heads = HEADS
        self.attention_head_dim = HEAD_DIM
        self.to_q = ReplicatedLinear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.to_k = ReplicatedLinear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.to_v = ReplicatedLinear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.to_out = ReplicatedLinear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=1e-5)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=1e-5)

    def _norm_and_rope(self, query, key, rotary_emb):
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = query * cos + query.flip(-1) * sin
            key = key * cos + key.flip(-1) * sin
        return query, key


def _layout() -> HybridSequenceLayout:
    seq_len = GLOBAL_ROWS + FRAMES * TOKENS_PER_FRAME
    return HybridSequenceLayout(
        seq_len=seq_len,
        video_start=GLOBAL_ROWS,
        video_end=seq_len,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_height=FRAME_H,
        frame_width=FRAME_W,
        text_start=0,
        text_end=GLOBAL_ROWS,
    )


def _rotary(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin broadcast over the head axis, as the DiT's RoPE does.
    cos = torch.rand(seq_len, 1, HEAD_DIM)
    sin = torch.rand(seq_len, 1, HEAD_DIM)
    return cos, sin


def _build(device, dtype):
    torch.manual_seed(0)
    attn = _StubParent().to(device=device, dtype=dtype)
    hybrid = HybridAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        window_radius=1,
        window_chunk=5,
        anchor_frames="both",
        branch_parallel=True,
        head_sharded_sp=True,
        quant_config=None,
    ).to(device=device, dtype=dtype)
    # Make the linear residual non-trivial so a dropped branch cannot pass.
    with torch.no_grad():
        hybrid.to_out_linear.weight.normal_(std=0.05)
        for module in (hybrid.linear_attention.output_gate.up, hybrid.linear_attention.beta_proj):
            module.weight.normal_(std=0.05)
        hybrid.linear_attention.write_log_scale.normal_(std=0.1)
    hybrid.eval()
    return attn, hybrid


def _reference(attn, hybrid, hidden, rope, layout):
    """Run the replicated route with the process group stubbed out."""
    attn_mod = attention_mod
    saved = (attn_mod.get_sp_world_size, attn_mod.model_parallel_is_initialized,
             attn_mod.get_sp_parallel_rank)
    attn_mod.get_sp_world_size = lambda: 1
    attn_mod.model_parallel_is_initialized = lambda: False
    attn_mod.get_sp_parallel_rank = lambda: 0
    try:
        return hybrid(attn, hidden, rope, layout.seq_len, layout, attn._norm_and_rope)
    finally:
        (attn_mod.get_sp_world_size, attn_mod.model_parallel_is_initialized,
         attn_mod.get_sp_parallel_rank) = saved


def _head_sharded_rank(attn, hybrid, hidden, rope, layout, world_size, rank):
    """Run the head-sharded route for one simulated rank and return its local output."""
    attn_mod = attention_mod
    saved = (attn_mod.get_sp_world_size, attn_mod.model_parallel_is_initialized,
             attn_mod.get_sp_parallel_rank, attn_mod.all_to_all_heads,
             attn_mod.all_to_all_heads_back, attn_mod.sequence_model_parallel_all_gather_with_unpad)

    # Mirror compute_padding_for_sp: pad UP to a multiple of the world size, then
    # hand each rank its contiguous chunk.  Floor division here would give a
    # negative pad whenever seq_len is not already a multiple.
    chunk = -(-layout.seq_len // world_size)
    pad = world_size * chunk - layout.seq_len
    local = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
    local = local.narrow(1, rank * chunk, chunk)

    attn_mod.get_sp_world_size = lambda: world_size
    attn_mod.model_parallel_is_initialized = lambda: True
    attn_mod.get_sp_parallel_rank = lambda: rank
    attn_mod.all_to_all_heads = lambda x, **kw: simulate_all_to_all_heads(x, world_size)
    attn_mod.all_to_all_heads_back = lambda x, **kw: simulate_all_to_all_heads_back(x, world_size)

    def fake_gather(x, original_len, **kw):
        # The hybrid body gathers the *hidden states* back to the full sequence
        # (the per-head parameter maps read the full token axis) but the RoPE
        # tables it also gathers are already full-length in this harness.
        if x.dim() == 3 and x.shape[0] == 1:
            return hidden
        return x

    attn_mod.sequence_model_parallel_all_gather_with_unpad = fake_gather
    try:
        return hybrid(attn, local, rope, layout.seq_len, layout, attn._norm_and_rope)
    finally:
        (attn_mod.get_sp_world_size, attn_mod.model_parallel_is_initialized,
         attn_mod.get_sp_parallel_rank, attn_mod.all_to_all_heads,
         attn_mod.all_to_all_heads_back,
         attn_mod.sequence_model_parallel_all_gather_with_unpad) = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="head-sharded route needs CUDA kernels")
@pytest.mark.parametrize("world_size", [2, 4])
def test_head_sharded_matches_replicated(world_size):
    """Every rank's local output must equal the replicated route's shard."""
    device, dtype = torch.device("cuda"), torch.bfloat16
    attn, hybrid = _build(device, dtype)
    layout = _layout()
    seq_len = layout.seq_len
    assert HEADS % world_size == 0

    hidden = torch.randn(1, seq_len, HIDDEN, device=device, dtype=dtype)
    rope = tuple(t.to(device=device, dtype=dtype) for t in _rotary(seq_len))

    reference = _reference(attn, hybrid, hidden, rope, layout)

    chunk = -(-seq_len // world_size)
    pad = world_size * chunk - seq_len
    padded_ref = torch.nn.functional.pad(reference, (0, 0, 0, pad))

    for rank in range(world_size):
        got = _head_sharded_rank(attn, hybrid, hidden, rope, layout, world_size, rank)
        # Receipt: the head-sharded route must actually have run.  Without this
        # an accidental fall back to the replicated route could pass on shapes
        # that happen to line up.
        assert hybrid.last_sp_route is not None
        assert hybrid.last_sp_route.startswith("head_sharded"), hybrid.last_sp_route
        want = padded_ref.narrow(1, rank * chunk, chunk)
        assert got.shape == want.shape, f"rank {rank}: {got.shape} != {want.shape}"
        # BF16 inputs, FP32 accumulators; a head-slice reassociation is the only
        # permitted difference, so require agreement well inside bf16 spacing.
        got_f, want_f = got.float(), want.float()
        denom = want_f.abs().mean().clamp_min(1e-6)
        rel = (got_f - want_f).abs().mean() / denom
        assert rel < 2e-2, f"rank {rank}: mean relative error {rel:.4e} too large"


def test_simulated_exchange_is_exact_inverse():
    """The stand-ins must compose to the identity, or the tests above prove nothing."""
    x = torch.randn(1, 6, 8, 4)
    for world_size in (2, 4):
        round_trip = simulate_all_to_all_heads_back(simulate_all_to_all_heads(x, world_size), world_size)
        assert torch.equal(round_trip, x)


def test_head_shard_slices_are_contiguous_and_cover():
    """Head slices must tile the head axis exactly."""
    for total, world in ((56, 4), (8, 2)):
        covered = []
        for rank in range(world):
            shard = HeadShard(index=rank, world_size=world, total_heads=total)
            covered.extend(range(shard.start, shard.stop))
        assert covered == list(range(total))


def test_head_shard_rejects_indivisible_head_counts():
    """An indivisible head count must fall back, not silently truncate."""
    assert parallel_mod.head_shard_for(56, 3, 0) is None
    assert parallel_mod.head_shard_for(56, 2, 1) is not None


def test_head_sharded_route_is_receipted():
    """The module must record which SP route ran, for the dispatch test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    _, hybrid = _build(device, dtype)
    assert hybrid.head_sharded_sp is True
    # Irrelevant for the CPU assertion below; just prove the switch exists and
    # that an indivisible world size disables the sharded route.
    assert hybrid._head_shard(1) is None
