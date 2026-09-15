# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel head sharding for MiniMax H3 hybrid attention.

The hybrid attention body originally all-gathered the packed sequence into every
rank and then ran *both* branches on the full sequence, sharding the result only
at the very end.  At SP=4 that repeats the QKV projection, the window softmax and
the whole linear branch four times, and only one rank's quarter of each result
survives the final ``shard``.

This module implements the Ulysses alternative already used by the VSA path
(``DistributedAttention_VSA.forward``): QKV is projected on the *local* sequence
shard, exchanged so that every rank holds the full sequence but only its own
contiguous slice of heads, and exchanged back after the branches.  Because the
hybrid linear branch is per-head independent, the exchange is the *only*
communication it needs.

Everything here is a pure tensor operation on an explicit ``HeadShard`` so the
sharding can be unit-tested on one process without a distributed group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeadShard:
    """A contiguous slice of attention heads owned by one sequence-parallel rank.

    Heads are contiguous in the last-but-one dimension of every per-head tensor
    in this package (``[..., heads, head_dim]``), and every per-head parameter --
    ``write_log_scale``, ``beta_proj``, ``alpha.up``, ``output_gate.up``, the
    short-conv channel stacks -- lays its head axis out the same way.  Slicing is
    therefore uniform: ``narrow`` on the head axis.
    """

    index: int
    world_size: int
    total_heads: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.total_heads <= 0:
            raise ValueError(f"total_heads must be positive, got {self.total_heads}")
        if not 0 <= self.index < self.world_size:
            raise ValueError(f"index {self.index} outside world_size {self.world_size}")
        if self.total_heads % self.world_size:
            raise ValueError(
                f"total_heads {self.total_heads} is not divisible by world_size {self.world_size}; "
                "fall back to the all-gather route.")

    @property
    def local_heads(self) -> int:
        return self.total_heads // self.world_size

    @property
    def start(self) -> int:
        return self.index * self.local_heads

    @property
    def stop(self) -> int:
        return self.start + self.local_heads

    def narrow(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """View of ``tensor`` restricted to this rank's heads along ``dim``."""
        return tensor.narrow(dim, self.start, self.local_heads)

    def channels(self, tensor: torch.Tensor, dim: int, head_dim: int) -> torch.Tensor:
        """Same as :meth:`narrow` for a fused ``heads * head_dim`` channel axis."""
        return tensor.narrow(dim, self.start * head_dim, self.local_heads * head_dim)

    def describe(self) -> str:
        return f"heads[{self.start}:{self.stop}]/{self.total_heads} of {self.world_size} ranks"


def head_shard_for(num_heads: int, sp_world_size: int, sp_rank: int) -> HeadShard | None:
    """Return the rank's head shard, or ``None`` when sharding is not applicable."""
    if sp_world_size <= 1 or num_heads % sp_world_size:
        return None
    return HeadShard(index=sp_rank, world_size=sp_world_size, total_heads=num_heads)


def all_to_all_heads(x: torch.Tensor, scatter_heads: int = 2, gather_seq: int = 1) -> torch.Tensor:
    """Sequence-sharded, all-heads -> full-sequence, head-sharded.

    Input ``[B, S_local, H, d]`` becomes ``[B, S_local * W, H / W, d]`` where each
    rank keeps a contiguous head slice.  Wraps the project's Ulysses primitive so
    the collective has a single call site.
    """
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D

    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=scatter_heads, gather_dim=gather_seq)


def all_to_all_heads_back(x: torch.Tensor, scatter_seq: int = 1, gather_heads: int = 2) -> torch.Tensor:
    """Inverse of :func:`all_to_all_heads`."""
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D

    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=scatter_seq, gather_dim=gather_heads)


def simulate_all_to_all_heads(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Single-process stand-in for :func:`all_to_all_heads`.

    Semantics of ``DistributedAutograd.AllToAll4D`` on ``[B, S, H, d]`` with
    ``scatter_dim=2, gather_dim=1``: rank ``r`` sends head-chunk ``r`` to rank
    ``r`` and concatenates the chunks it receives along the sequence axis.  The
    result for every rank is the same tensor, so a single process can produce it
    directly.  Used by the correctness tests.
    """
    batch, seq, heads, head_dim = x.shape
    if heads % world_size:
        raise ValueError(f"heads {heads} not divisible by world_size {world_size}")
    chunk = heads // world_size
    # [B, S, W, chunk, d] -> [B, W, S, chunk, d] -> [B, W*S, chunk, d]
    parts = x.reshape(batch, seq, world_size, chunk, head_dim)
    return parts.permute(0, 2, 1, 3, 4).reshape(batch, world_size * seq, chunk, head_dim)


def simulate_all_to_all_heads_back(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Single-process stand-in for :func:`all_to_all_heads_back`."""
    batch, seq, heads, head_dim = x.shape
    if seq % world_size:
        raise ValueError(f"sequence {seq} not divisible by world_size {world_size}")
    chunk = seq // world_size
    parts = x.reshape(batch, world_size, chunk, heads, head_dim)
    return parts.permute(0, 2, 1, 3, 4).reshape(batch, chunk, world_size * heads, head_dim)
