# SPDX-License-Identifier: Apache-2.0
"""NVFP4 layer-prefix selection: LTX-2 default, MiniMax-H3 opt-in.

``NVFP4Config`` used to hardcode the LTX-2 layer paths, so the same config
handed to MiniMax-H3 attached no quant methods at all and the model ran dense
without an error. These CPU-only tests pin both halves of the fix: the default
still selects exactly the LTX-2 set, and an H3-configured instance selects the
300 block linears while refusing to quantize the VSA compression gate.

No flashinfer and no CUDA are required — the tests use real
``ReplicatedLinear`` layers (which is where ``get_quant_method`` is called from)
with only ``NVFP4QuantizeMethod.__init__`` replaced, because the real one
allocates ``x_global_sf`` on ``cuda``. The same shim is used by
``test_nvfp4_purge.py``.
"""
from __future__ import annotations

import sys

import pytest
import torch

import fastvideo.layers.quantization.nvfp4_config as nv
from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

_H3_BLOCK = "minimax_h3.transformer_blocks.{idx}.{suffix}"
_GATE_PREFIX = _H3_BLOCK.format(idx=0, suffix="attn.to_gate_compress")


@pytest.fixture(autouse=True)
def _cpu_quantize_method(monkeypatch):
    """Build NVFP4QuantizeMethod without its cuda-allocated ``x_global_sf``."""

    def _init(self, layer_prefix: str = ""):
        self.weight_fp4 = None
        self.weight_scale = None
        self.x_global_sf = torch.tensor(1.0, dtype=torch.float32)
        self.layer_prefix = layer_prefix
        self._is_refine_only_layer = nv._is_ltx2_refine_only_prefix(layer_prefix)
        self._retain_original_weights = None

    monkeypatch.setattr(nv.NVFP4QuantizeMethod, "__init__", _init)


def _linear(quant_config, prefix: str) -> ReplicatedLinear:
    return ReplicatedLinear(8, 8, bias=False, quant_config=quant_config, prefix=prefix)


def test_default_config_keeps_the_ltx2_layer_set() -> None:
    """No regression: the historical default is still the LTX-2 set."""
    config = nv.NVFP4Config()
    assert config.layer_prefixes == nv._LTX2_NVFP4_LINEAR_PREFIXES
    assert len(config.layer_prefixes) == 577
    assert config.exclude_prefixes == frozenset()
    assert config.is_nvfp4_linear_prefix("ltx2.blocks.0.attn1.to_q")
    assert config.is_nvfp4_linear_prefix("ltx2.adaln_single.linear")
    assert not config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=0, suffix="attn.to_q"))


def test_h3_config_selects_the_300_block_linears() -> None:
    config = nv.NVFP4Config.for_minimax_h3()
    assert len(config.layer_prefixes) == 300
    assert config.layer_prefixes == nv.MINIMAX_H3_NVFP4_LINEAR_PREFIXES
    # Every block, every suffix.
    for idx in range(nv.MINIMAX_H3_NUM_LAYERS):
        for suffix in nv.MINIMAX_H3_BLOCK_LINEAR_SUFFIXES:
            assert config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=idx, suffix=suffix))
    # Edge blocks are really in the set, and one past the last block is not.
    assert config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=49, suffix="ff.fc_out"))
    assert not config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=50, suffix="ff.fc_out"))
    # Nothing outside the block set is selected.
    assert not config.is_nvfp4_linear_prefix("minimax_h3.token_refiner.blocks.0.attn.to_q")
    assert not config.is_nvfp4_linear_prefix("minimax_h3.proj_in")
    assert not config.is_nvfp4_linear_prefix("minimax_h3.transformer_blocks.0.adaln_proj.linear")
    assert not config.is_nvfp4_linear_prefix("ltx2.blocks.0.attn1.to_q")


def test_h3_config_excludes_the_vsa_gate() -> None:
    """``attn.to_gate_compress`` must never be quantized."""
    config = nv.NVFP4Config.for_minimax_h3()
    assert not config.is_nvfp4_linear_prefix(_GATE_PREFIX)
    assert not config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=37, suffix="attn.to_gate_compress"))
    assert _GATE_PREFIX not in config.layer_prefixes
    assert config.exclude_prefixes == frozenset(nv.MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES)


def test_gate_is_excluded_even_if_a_caller_allowlists_it() -> None:
    """The gate exclusion is unconditional, not just absent from the set.

    A caller who builds the prefix set with a glob (or hand-lists every linear
    in the block) must still not get a quantized VSA gate.
    """
    config = nv.NVFP4Config(layer_prefixes=frozenset({_GATE_PREFIX, _H3_BLOCK.format(idx=0, suffix="attn.to_q")}))
    assert not config.is_nvfp4_linear_prefix(_GATE_PREFIX)
    assert config.is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=0, suffix="attn.to_q"))
    # The same holds when the caller passes no exclusion list at all.
    assert _GATE_PREFIX not in nv._LTX2_NVFP4_LINEAR_PREFIXES
    assert not nv.NVFP4Config(layer_prefixes=[_GATE_PREFIX]).is_nvfp4_linear_prefix(_GATE_PREFIX)


def test_exclude_prefixes_match_a_suffix_or_a_full_path() -> None:
    full_path = "custom.blocks.3.attn.to_out"
    config = nv.NVFP4Config(layer_prefixes=[full_path, "custom.other"], exclude_prefixes=["attn.to_out"])
    assert config.is_nvfp4_linear_prefix("custom.other")
    assert not config.is_nvfp4_linear_prefix(full_path)
    # A dot boundary is required, so a sibling name is not excluded by accident.
    nested = nv.NVFP4Config(layer_prefixes=["custom.cross_ff.fc_in"], exclude_prefixes=["ff.fc_in"])
    assert nested.is_nvfp4_linear_prefix("custom.cross_ff.fc_in")


def test_from_config_round_trips_layer_prefixes() -> None:
    config = nv.NVFP4Config.from_config({
        "layer_profile": "base",
        "layer_prefixes": ["minimax_h3.transformer_blocks.0.attn.to_q"],
        "exclude_prefixes": ["attn.to_gate_compress"],
    })
    assert config.layer_profile == "base"
    assert config.layer_prefixes == frozenset({"minimax_h3.transformer_blocks.0.attn.to_q"})
    assert config.exclude_prefixes == frozenset({"attn.to_gate_compress"})
    # Absent keys keep the LTX-2 default.
    assert nv.NVFP4Config.from_config({}).layer_prefixes == nv._LTX2_NVFP4_LINEAR_PREFIXES


def test_for_minimax_h3_forwards_kwargs() -> None:
    config = nv.NVFP4Config.for_minimax_h3(retain_original_weights=True)
    assert config.retain_original_weights is True
    assert config.layer_prefixes == nv.MINIMAX_H3_NVFP4_LINEAR_PREFIXES


def test_get_quant_method_attaches_for_h3_and_skips_the_gate() -> None:
    """End-to-end through ``ReplicatedLinear``, which is the real call site."""
    h3 = nv.NVFP4Config.for_minimax_h3()
    selected = _linear(h3, _H3_BLOCK.format(idx=0, suffix="attn.to_q"))
    assert isinstance(selected.quant_method, nv.NVFP4QuantizeMethod)
    assert selected.quant_method.layer_prefix == _H3_BLOCK.format(idx=0, suffix="attn.to_q")

    gate = _linear(h3, _GATE_PREFIX)
    assert type(gate.quant_method) is UnquantizedLinearMethod

    # The pre-fix bug, pinned: the default config attaches nothing on H3.
    default = nv.NVFP4Config()
    untagged = _linear(default, _H3_BLOCK.format(idx=0, suffix="attn.to_q"))
    assert type(untagged.quant_method) is UnquantizedLinearMethod
    # ... and still tags LTX-2.
    ltx2 = _linear(default, "ltx2.blocks.0.attn1.to_q")
    assert isinstance(ltx2.quant_method, nv.NVFP4QuantizeMethod)


def test_module_imports_without_flashinfer(monkeypatch) -> None:
    """The H3 surface must import on hosts with no flashinfer (only the
    kernels fail, at use time)."""
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    # delitem (not a bare pop) so monkeypatch puts the original module object
    # back on teardown: leaving a re-imported copy in sys.modules would give
    # later tests a second, non-identical NVFP4Config class.
    monkeypatch.delitem(sys.modules, "fastvideo.layers.quantization.nvfp4_config", raising=False)
    import importlib

    reloaded = importlib.import_module("fastvideo.layers.quantization.nvfp4_config")
    assert len(reloaded.MINIMAX_H3_NVFP4_LINEAR_PREFIXES) == 300
    assert reloaded.NVFP4Config.for_minimax_h3().is_nvfp4_linear_prefix(_H3_BLOCK.format(idx=0, suffix="attn.to_k"))
