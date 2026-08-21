# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the sharded base-weight cache.

Runs on a single-process gloo group with a (1, 1) CPU device mesh: DTensor
round-trip through write/load, the FQN reconciliation matrix (allowed
zero-init params, disallowed extras, shape mismatches), and the
never-fail-the-run contract.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from fastvideo.models.loader.shard_cache import (
    ShardCacheContext,
    try_load_from_shard_cache,
    write_shard_cache,
)


@pytest.fixture(scope="module")
def cpu_mesh():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    return init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))


def _make_model(
    cpu_mesh,
    *,
    extra_param: str | None = None,
    weight_rows: int = 8,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    model = nn.Module()
    placements = (Replicate(), Shard(0))
    weight = distribute_tensor(torch.randn(weight_rows, 4, dtype=dtype), cpu_mesh, placements)
    bias = distribute_tensor(torch.randn(weight_rows, dtype=dtype), cpu_mesh, placements)
    model.register_parameter("weight", nn.Parameter(weight))
    model.register_parameter("bias", nn.Parameter(bias))
    model.register_buffer("scale", torch.full((1, ), 2.0))
    if extra_param is not None:
        extra = distribute_tensor(torch.randn(4, 4), cpu_mesh, placements)
        model.register_parameter(extra_param.replace(".", "_"), nn.Parameter(extra))
        # register under the dotted name via a child module for realism
    model.reverse_param_names_mapping = {"weight": ("hf.weight", None, None)}
    return model


def _ctx(tmp_path) -> ShardCacheContext:
    return ShardCacheContext(entry_dir=tmp_path / "entry", key="testkey", shard_index=0, num_shards=1, is_writer=True)


def test_round_trip_restores_tensors_and_reverse_mapping(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    with torch.no_grad():
        dst.weight.mul_(0)
        dst.bias.mul_(0)
    dst.reverse_param_names_mapping = {}
    assert try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.weight.to_local(), src.weight.to_local())
    assert torch.equal(dst.bias.to_local(), src.bias.to_local())
    assert torch.equal(dst.scale, src.scale)
    assert dst.reverse_param_names_mapping == {"weight": ("hf.weight", None, None)}


def test_allowed_new_param_zero_inits_on_hit(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    gate = distribute_tensor(torch.randn(4, 4), cpu_mesh, (Replicate(), Shard(0)))
    dst.register_parameter("to_gate_compress", nn.Parameter(gate))
    assert try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.to_gate_compress.to_local(), torch.zeros(4, 4))


def test_disallowed_missing_param_misses_without_mutation(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh)
    mystery = distribute_tensor(torch.randn(4, 4), cpu_mesh, (Replicate(), Shard(0)))
    dst.register_parameter("mystery", nn.Parameter(mystery))
    before = dst.weight.to_local().clone()
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))
    assert torch.equal(dst.weight.to_local(), before)


def test_shape_mismatch_misses(cpu_mesh, tmp_path):
    src = _make_model(cpu_mesh)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)

    dst = _make_model(cpu_mesh, weight_rows=9)
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))


def test_missing_entry_misses_cleanly(cpu_mesh, tmp_path):
    dst = _make_model(cpu_mesh)
    ctx = ShardCacheContext(entry_dir=tmp_path / "absent", key="k2", shard_index=0, num_shards=1, is_writer=True)
    assert not try_load_from_shard_cache(dst, ctx, torch.device("cpu"))


def test_ac_wrapped_buffer_stays_buffer_on_cache_hit(cpu_mesh, tmp_path):
    """Warm-path counterpart of the cold-path AC-prefix fix in fsdp_load.

    Under pre-FSDP activation checkpointing the model handed to
    ``try_load_from_shard_cache`` is already checkpoint-wrapped:
    ``state_dict()`` (and manifest) keys are clean, but raw
    ``named_buffers()`` keys carry the ``_checkpoint_wrapped_module.``
    segment. The buffer-membership test must compare canonical names, or a
    cached persistent buffer inside a wrapped block is reassigned as an
    ``nn.Parameter`` by ``load_state_dict(assign=True)`` — on warm boots
    only, silently diverging from the (already fixed) cold-boot path.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, )

    def _block_model() -> nn.Module:
        model = nn.Module()
        block = nn.Module()
        weight = distribute_tensor(torch.randn(8, 4), cpu_mesh, (Replicate(), Shard(0)))
        block.register_parameter("weight", nn.Parameter(weight))
        block.register_buffer("gain", torch.full((4, ), 3.0))
        model.block = block
        model.reverse_param_names_mapping = {}
        return model

    # Cold boot writes the cache from the same (wrapped) model shape; keys in
    # the manifest are clean either way because state_dict strips the prefix.
    src = _block_model()
    src.block = checkpoint_wrapper(src.block)
    ctx = _ctx(tmp_path)
    write_shard_cache(src, ctx)
    assert "block.gain" in src.state_dict()

    dst = _block_model()
    dst.block = checkpoint_wrapper(dst.block)
    with torch.no_grad():
        dst.block.weight.mul_(0)
        dst.block.gain.mul_(0)
    assert try_load_from_shard_cache(dst, ctx, torch.device("cpu"))

    buffer_names = {name for name, _ in dst.named_buffers()}
    parameter_names = {name for name, _ in dst.named_parameters()}
    assert "block._checkpoint_wrapped_module.gain" in buffer_names
    assert not any(name.endswith("gain") for name in parameter_names)
    assert torch.equal(dst.block.gain, torch.full((4, ), 3.0))
    assert torch.equal(dst.block.weight.to_local(), src.block.weight.to_local())


def test_model_selected_dtype_rejects_stale_cache_and_hits_fresh_cache(cpu_mesh, tmp_path):
    stale = _make_model(cpu_mesh, dtype=torch.bfloat16)
    stale_ctx = _ctx(tmp_path / "stale")
    write_shard_cache(stale, stale_ctx)

    def select_dtype(name: str, default: torch.dtype) -> torch.dtype:
        return torch.float32 if name == "weight" else default

    destination = _make_model(cpu_mesh, dtype=torch.bfloat16)
    destination._get_parameter_dtype = select_dtype
    assert not try_load_from_shard_cache(destination, stale_ctx, torch.device("cpu"))

    fresh = _make_model(cpu_mesh, dtype=torch.bfloat16)
    fresh_weight = distribute_tensor(
        torch.randn(8, 4, dtype=torch.float32),
        cpu_mesh,
        (Replicate(), Shard(0)),
    )
    fresh.weight = nn.Parameter(fresh_weight)
    fresh_ctx = _ctx(tmp_path / "fresh")
    write_shard_cache(fresh, fresh_ctx)

    destination = _make_model(cpu_mesh, dtype=torch.bfloat16)
    destination._get_parameter_dtype = select_dtype
    assert try_load_from_shard_cache(destination, fresh_ctx, torch.device("cpu"))
    assert destination.weight.dtype == torch.float32
