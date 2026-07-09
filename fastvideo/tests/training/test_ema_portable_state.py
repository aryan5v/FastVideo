# SPDX-License-Identifier: Apache-2.0
"""EMA checkpoint state must be world-size-portable.

Regression for the corrupted `--ema` export: the EMA shadow is a dict of
per-rank local shards keyed by live module names (with
activation-checkpointing wrapper prefixes). Checkpointed directly, DCP saved
only rank 0's quarter shards from a 4-GPU run, and a 1-GPU export loaded
them into full-shape buffers — producing noise weights whose uniform
INT8-vs-FP16 SSIM masqueraded as a QAT win. The callback now saves full
tensors with normalized names and re-slices on load.

CPU tests cover the world-size-1 path and name normalization; the multi-rank
gather/scatter runs through the same DTensor helpers and is exercised by the
next DGX smoke run.
"""

from __future__ import annotations

import types

import torch
import torch.distributed.checkpoint.stateful  # noqa: F401  (import-order guard, see test_mlx_qat_callback)

from fastvideo.train.callbacks.ema import EMACallback


class _Tiny(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(16, 16, bias=False)
        self.proj_out = torch.nn.Linear(16, 8, bias=True)


def _method(transformer: torch.nn.Module):
    return types.SimpleNamespace(student=types.SimpleNamespace(transformer=transformer))


def _armed_callback(transformer: torch.nn.Module) -> EMACallback:
    cb = EMACallback(decay=0.9, start_iter=0)
    cb.on_train_start(_method(transformer))
    cb._ema_started = True
    return cb


def test_state_dict_uses_clean_names_and_dcp_key() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    cb = _armed_callback(model)
    # Simulate activation-checkpointing wrapper names in the live shadow.
    cb.student_ema.shadow = {
        f"{name.rsplit('.', 1)[0]}._checkpoint_wrapped_module.{name.rsplit('.', 1)[1]}": value
        for name, value in cb.student_ema.shadow.items()
    }

    state = cb.state_dict()
    assert state["ema_started"] is True
    # New DCP-native key; the legacy plain-shard key must be gone so old
    # loaders cannot silently accept the new format's shapeless shards.
    assert "student_ema" not in state
    sharded = state["student_ema_sharded"]
    # Names are normalized (AC wrapper stripped) so they match live params.
    assert set(sharded) == {"to_q.weight", "proj_out.weight", "proj_out.bias"}
    # On a non-distributed (world_size=1) model the params are plain tensors,
    # so the shards round-trip at full shape.
    for name, param in model.named_parameters():
        assert sharded[name].shape == param.shape


def test_round_trip_restores_shadow_for_current_param_names() -> None:
    torch.manual_seed(1)
    source_model = _Tiny()
    source = _armed_callback(source_model)
    for shard in source.student_ema.shadow.values():
        shard.add_(torch.randn_like(shard))  # make shadow distinct from init
    saved = source.state_dict()

    target_model = _Tiny()
    target = _armed_callback(target_model)
    target._ema_started = False
    target.load_state_dict(saved)

    assert target._ema_started is True
    for name in dict(target_model.named_parameters()):
        torch.testing.assert_close(
            target.student_ema.shadow[name], source.student_ema.shadow[name], atol=0, rtol=0)

    # And the swapped-in weights match the source EMA, dtype-preserved.
    with target.ema_context(target_model):
        torch.testing.assert_close(
            target_model.to_q.weight.detach().float(),
            source.student_ema.shadow["to_q.weight"],
            atol=1e-6, rtol=1e-6)


def test_legacy_per_shard_state_is_refused() -> None:
    model = _Tiny()
    cb = _armed_callback(model)
    legacy = {"student_ema": {"to_q.weight": torch.zeros(4, 16)}, "ema_started": True}
    try:
        cb.load_state_dict(legacy)
    except ValueError as exc:
        assert "legacy" in str(exc).lower() or "per-shard" in str(exc)
    else:
        raise AssertionError("legacy per-shard EMA state must be refused, not silently loaded")
