# SPDX-License-Identifier: Apache-2.0
"""FP8 suffix matching for hybrid MiniMax H3 extras.

Locks the substring list in ``FP8Config.get_quant_method`` so hybrid
``beta_proj`` / gates / ``to_out_linear`` are tagged while KDA ``alpha``
stays unquantized. CPU-only: constructing ``ReplicatedLinear`` is enough
to exercise the LinearBase ``quant_config`` seam.
"""
from __future__ import annotations

import pytest

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod

PREFIX = "blocks.0.attn"
FP8_PREFIXES = (
    f"{PREFIX}.to_q",
    f"{PREFIX}.to_k",
    f"{PREFIX}.to_v",
    f"{PREFIX}.to_out",
    f"{PREFIX}.to_out_linear",
    f"{PREFIX}.linear_attention.beta_proj",
    f"{PREFIX}.softmax_gate.up",
    f"{PREFIX}.linear_attention.output_gate.down",
    f"{PREFIX}.linear_attention.output_gate.up",
)
FP32_ALPHA_PREFIXES = (
    f"{PREFIX}.linear_attention.alpha.down",
    f"{PREFIX}.linear_attention.alpha.up",
)


def _linear(prefix: str, quant_config: FP8Config | None) -> ReplicatedLinear:
    return ReplicatedLinear(8, 8, bias=False, quant_config=quant_config, prefix=prefix)


@pytest.mark.parametrize("prefix", FP8_PREFIXES)
def test_hybrid_fp8_suffixes_tag_replicated_linears(prefix: str) -> None:
    cfg = FP8Config()
    layer = _linear(prefix, cfg)
    assert isinstance(layer.quant_method, FP8QuantizeMethod)
    assert isinstance(cfg.get_quant_method(layer, prefix), FP8QuantizeMethod)


@pytest.mark.parametrize("prefix", FP32_ALPHA_PREFIXES)
def test_kda_alpha_linears_stay_unquantized(prefix: str) -> None:
    cfg = FP8Config()
    # Even if a caller passed FP8Config, alpha prefixes must not match.
    layer = _linear(prefix, cfg)
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert cfg.get_quant_method(layer, prefix) is None


def test_no_quant_config_keeps_hybrid_linears_unquantized() -> None:
    for prefix in (*FP8_PREFIXES, *FP32_ALPHA_PREFIXES):
        layer = _linear(prefix, None)
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
