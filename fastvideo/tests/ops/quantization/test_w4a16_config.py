# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the W4A16 (4-bit weight / 16-bit activation) config.

Scope: the quantizer round-trip, the packed-code layout, the MiniMax-H3 layer
selection (including the ``attn.to_gate_compress`` exclusion) and the load-time
conversion path. Nothing here needs CUDA, a GPU kernel, or flashinfer — the
compute path under test is the documented ``dequantize then dense GEMM``
reference, which is pure PyTorch.

Tolerance note: W4A16's 4-bit codes give a per-element reconstruction error
bounded by half the group's quantizer step,
``scale = (max - min) / 15``. Tests assert against that *derived* bound rather
than a hand-picked constant, so the assertion stays meaningful if the group
size changes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.w4a16_config import (
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZE,
    MINIMAX_H3_MAIN_STACK_LINEAR_SUFFIXES,
    W4A16Config,
    W4A16QuantizeMethod,
    convert_model_to_w4a16,
    minimax_h3_w4a16_prefixes,
    w4a16_dequantize,
    w4a16_quantize,
)


def _random_weight(out_dim: int, in_dim: int, *, generator: torch.Generator) -> torch.Tensor:
    """Deterministic, roughly-linear-layer-shaped weights (non-uniform rows)."""
    return torch.randn(out_dim, in_dim, generator=generator) * 0.02


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_imports_without_cuda_dependencies():
    """Importing the config must not require CUDA, a kernel, or flashinfer."""
    config = W4A16Config()
    assert config.get_name() == "W4A16"
    assert torch.bfloat16 in config.get_supported_act_dtypes()
    assert config.get_config_filenames() == []
    # Declared contract only -- the reference path is a dense 16-bit GEMM, so
    # no 4-bit tensor-core class is required to *load* it.
    assert W4A16Config.get_min_capability() >= 70
    assert DEFAULT_BITS == 4


def test_config_rejects_unsupported_bits():
    with pytest.raises(ValueError):
        W4A16Config(bits=3)
    with pytest.raises(ValueError):
        W4A16Config(group_size=0)


def test_from_config_round_trips_fields():
    config = W4A16Config.from_config({
        "group_size": 32,
        "bits": 4,
        "target_layers": ["a.b"],
        "exclude_substrings": ["keep_me_dense"],
        "retain_original_weight": False,
    })
    assert config.group_size == 32
    assert config.target_layers == frozenset({"a.b"})
    assert "keep_me_dense" in config.exclude_substrings
    assert config.retain_original_weight is False


# ---------------------------------------------------------------------------
# Quantizer round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_quantize_dequantize_round_trip_within_step_bound(group_size: int):
    """Every reconstructed element is within half a quantizer step of the source."""
    generator = torch.Generator().manual_seed(0)
    weight = _random_weight(48, 4 * group_size, generator=generator)

    codes, scales, zeros = w4a16_quantize(weight, group_size=group_size, bits=4)
    restored = w4a16_dequantize(codes, scales, zeros, group_size=group_size, bits=4, out_shape=weight.shape)

    # The quantizer returns codes in the group layout; the logical weight shape
    # is the caller's to supply. Codes are packed two-per-byte for bits=4.
    assert codes.dtype == torch.uint8
    assert codes.shape == (weight.shape[0], weight.shape[1] // 2)
    assert scales.shape == (weight.shape[0], weight.shape[1] // group_size)
    assert zeros.shape == scales.shape
    assert restored.shape == weight.shape

    error = (restored - weight).abs()
    # Bound per group: ``code = round(w / scale + zero)`` is off by at most half
    # a code, so the reconstruction is off by at most half a step. The zero
    # point's own rounding is absorbed by that same ``round``, so it does not
    # widen the bound.
    per_element_bound = (scales.unsqueeze(-1) / 2 + 1e-6)
    assert torch.all(error.reshape(weight.shape[0], -1, group_size) <= per_element_bound)


def test_round_trip_is_exact_for_uniform_group():
    """A group that lands exactly on the code grid reconstructs exactly.

    ``zero`` is a rounded integer, so the anchor is only exact when
    ``-min / scale`` is already integral -- here the ladder spans codes 0..15
    with a step of 0.1, giving ``zero = 8`` exactly.
    """
    group_size = 64
    ladder = ((torch.arange(16).float() - 8) * 0.1).repeat(group_size // 16)
    weight = ladder.repeat(2, 1)

    codes, scales, zeros = w4a16_quantize(weight, group_size=group_size, bits=4)
    restored = w4a16_dequantize(codes, scales, zeros, group_size=group_size, bits=4, out_shape=weight.shape)

    assert torch.allclose(restored, weight, atol=1e-6)


def test_round_trip_holds_across_activation_dtypes():
    """bf16/fp16 sources still reconstruct within the step bound."""
    group_size = 64
    generator = torch.Generator().manual_seed(1)
    base = _random_weight(16, group_size * 2, generator=generator)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        weight = base.to(dtype)
        codes, scales, zeros = w4a16_quantize(weight, group_size=group_size, bits=4)
        restored = w4a16_dequantize(codes, scales, zeros, group_size=group_size, bits=4, out_shape=weight.shape)
        error = (restored - weight.float()).abs()
        # A 16-bit source rounds before quantizing, so the tolerance over the
        # half-step bound is widened by that dtype's own resolution.
        bound = (scales.unsqueeze(-1) / 2 + 1e-2)
        assert torch.all(error.reshape(weight.shape[0], -1, group_size) <= bound), dtype


def test_quantize_rejects_indivisible_group_size():
    weight = torch.randn(4, 100)
    with pytest.raises(ValueError, match="not divisible"):
        w4a16_quantize(weight, group_size=64)


def test_packed_codes_store_four_bits_per_weight():
    """The 4-bit storage claim is real: 4 bits/weight for the codes themselves.

    (Excludes the per-group scales/zeros, which add ``2 * 32 / group_size``
    bits per weight -- 1 bit/weight at ``group_size=64``.)
    """
    weight = torch.randn(32, 128)
    codes, scales, zeros = w4a16_quantize(weight, group_size=64, bits=4)
    assert codes.numel() == weight.numel() // 2
    assert codes.element_size() == 1
    assert codes.numel() * 8 == weight.numel() * 4
    assert scales.numel() == zeros.numel() == weight.numel() // 64


def test_packed_nibble_layout_is_low_first():
    """Low nibble holds the lower K index -- the documented packing order.

    Asserted structurally rather than against hand-computed codes: a strictly
    increasing row must unpack to a non-decreasing code sequence. Swapping the
    nibble order would make the unpacked sequence zig-zag within every pair,
    so this distinguishes the two conventions without depending on where the
    zero-point rounding lands.
    """
    from fastvideo.layers.quantization.w4a16_config import _unpack_4bit

    group_size = 64
    weight = torch.linspace(-1.0, 1.0, group_size).unsqueeze(0)
    codes, _, _ = w4a16_quantize(weight, group_size=group_size, bits=4)
    unpacked = _unpack_4bit(codes)

    assert unpacked.shape == weight.shape
    assert unpacked.dtype == torch.uint8
    deltas = unpacked[0, 1:].to(torch.int16) - unpacked[0, :-1].to(torch.int16)
    assert torch.all(deltas >= 0), unpacked[0]


def test_nan_does_not_poison_the_group():
    """nan_to_num matches the other configs: one NaN must not nuke its group."""
    group_size = 64
    weight = torch.randn(1, group_size)
    weight[0, 3] = float("nan")
    codes, scales, zeros = w4a16_quantize(weight, group_size=group_size, bits=4)
    restored = w4a16_dequantize(codes, scales, zeros, group_size=group_size, bits=4, out_shape=weight.shape)
    assert torch.isfinite(restored).all()


# ---------------------------------------------------------------------------
# MiniMax-H3 layer selection
# ---------------------------------------------------------------------------


def test_h3_allowlist_is_362_linears():
    config = W4A16Config.for_minimax_h3()
    assert len(config.target_layers) == 362
    # 50 main blocks x (4 attn + 2 ff + 1 adaln) + 2 refiner blocks x (4 attn + 2 ff)
    assert len(config.target_layers) == 50 * 7 + 2 * 6
    assert len(minimax_h3_w4a16_prefixes()) == 362


def test_h3_selection_includes_the_intended_linears():
    config = W4A16Config.for_minimax_h3()
    included = [
        "minimax_h3.transformer_blocks.0.attn.to_q",
        "minimax_h3.transformer_blocks.0.attn.to_k",
        "minimax_h3.transformer_blocks.0.attn.to_v",
        "minimax_h3.transformer_blocks.0.attn.to_out",
        "minimax_h3.transformer_blocks.0.ff.fc_in",
        "minimax_h3.transformer_blocks.0.ff.fc_out",
        "minimax_h3.transformer_blocks.0.adaln_proj.linear",
        "minimax_h3.transformer_blocks.49.attn.to_q",
        "minimax_h3.transformer_blocks.49.ff.fc_out",
        "minimax_h3.token_refiner.refiner_blocks.0.attn.to_q",
        "minimax_h3.token_refiner.refiner_blocks.1.ff.fc_out",
    ]
    for prefix in included:
        assert config.is_target_layer(prefix), prefix
    assert "adaln_proj.linear" in MINIMAX_H3_MAIN_STACK_LINEAR_SUFFIXES


def test_h3_selection_excludes_to_gate_compress():
    """The VSA gate steers discrete sparse routing -- it must never be quantized."""
    config = W4A16Config.for_minimax_h3()
    for index in (0, 17, 49):
        assert not config.is_target_layer(f"minimax_h3.transformer_blocks.{index}.attn.to_gate_compress")
        assert not config.is_target_layer("minimax_h3.token_refiner.refiner_blocks.0.attn.to_gate_compress")


def test_to_gate_compress_exclusion_cannot_be_widened_away():
    """A caller cannot opt the gate back in, even by naming it in the allowlist."""
    gate = "minimax_h3.transformer_blocks.0.attn.to_gate_compress"
    config = W4A16Config(target_layers=[gate, "minimax_h3.transformer_blocks.0.attn.to_q"])
    assert not config.is_target_layer(gate)
    assert config.is_target_layer("minimax_h3.transformer_blocks.0.attn.to_q")


def test_h3_gate_module_is_built_and_left_dense(monkeypatch):
    """The strongest form of the gate check: the real H3 attention module.

    H3 only builds ``attn.to_gate_compress`` when the VSA backend resolves
    (``use_vsa`` guards the construction), which does not happen on a CPU-only
    host. Forcing the backend resolution is enough to get the module built and
    observe which quant method it receives -- this is the wiring the exclusion
    actually has to protect, not just the string predicate.
    """
    from fastvideo.layers.linear import UnquantizedLinearMethod

    import fastvideo.models.dits.minimax_h3 as h3_module
    from fastvideo.platforms import AttentionBackendEnum

    class _FakeVSABackend:

        def get_name(self) -> str:
            return "VIDEO_SPARSE_ATTN_H3"

    monkeypatch.setattr(h3_module, "get_attn_backend", lambda *args, **kwargs: _FakeVSABackend())

    prefix = "minimax_h3.transformer_blocks.0.attn"
    attention = h3_module.MiniMaxH3Attention(
        64,
        2,
        32,
        1e-5,
        (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, ),
        W4A16Config.for_minimax_h3(),
        prefix=prefix,
    )
    assert attention.to_gate_compress is not None
    assert isinstance(attention.to_gate_compress.quant_method, UnquantizedLinearMethod)
    assert isinstance(attention.to_q.quant_method, W4A16QuantizeMethod)
    assert isinstance(attention.to_out.quant_method, W4A16QuantizeMethod)


def test_h3_selection_excludes_fp32_pinned_modules():
    config = W4A16Config.for_minimax_h3()
    for prefix in (
            "minimax_h3.proj_in",
            "minimax_h3.audio_proj_in",
            "minimax_h3.time_embedder.fc_in",
            "minimax_h3.proj_out",
            "minimax_h3.audio_proj_out",
            "minimax_h3.adaln_basis",
            "minimax_h3.norm_out.linear",
            "minimax_h3.context_embedder",
    ):
        assert not config.is_target_layer(prefix), prefix


def test_generic_suffix_matching_respects_dot_boundaries():
    """``ff.fc_in`` must not match a hypothetical ``cross_ff.fc_in``."""
    config = W4A16Config()
    assert config.is_target_layer("model.blocks.0.ff.fc_in")
    assert not config.is_target_layer("model.blocks.0.cross_ff.fc_in")


def test_get_quant_method_skips_indivisible_and_odd_dims():
    """A group must fit inside one row and 4-bit codes need an even last dim."""
    config = W4A16Config(target_layers=["blk.a", "blk.b", "blk.c"], group_size=64)
    assert isinstance(config.get_quant_method(ReplicatedLinear(64, 8, bias=False, prefix="blk.a"), "blk.a"),
                      W4A16QuantizeMethod)
    # 100 is not divisible by 64 -> dense.
    assert config.get_quant_method(ReplicatedLinear(100, 8, bias=False, prefix="blk.b"), "blk.b") is None
    # Divisible by group_size but odd -> 4-bit packing impossible -> dense.
    odd = W4A16Config(target_layers=["blk.c"], group_size=5)
    assert odd.get_quant_method(ReplicatedLinear(25, 8, bias=False, prefix="blk.c"), "blk.c") is None


def test_get_quant_method_ignores_non_linear_layers():
    config = W4A16Config()
    assert config.get_quant_method(nn.RMSNorm(64), "minimax_h3.transformer_blocks.0.attn.norm_q") is None


# ---------------------------------------------------------------------------
# Load-time conversion path
# ---------------------------------------------------------------------------


def _tiny_model(prefix: str, quant_config: W4A16Config, in_dim: int = 64, out_dim: int = 32):
    linear = ReplicatedLinear(in_dim, out_dim, bias=False, quant_config=quant_config, prefix=prefix)
    module = nn.Module()
    module.add_module("linear", linear)
    return module, linear


def test_convert_registers_non_persistent_buffers_and_keeps_weight():
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    model, linear = _tiny_model("block.linear", config)
    linear.weight.data.normal_()

    convert_model_to_w4a16(model)

    assert linear._w4a16_codes.dtype == torch.uint8
    assert tuple(linear._w4a16_weight_shape) == tuple(linear.weight.shape)
    # Non-persistent: the quantized payload must not leak into checkpoints.
    assert list(model.state_dict().keys()) == ["linear.weight"]
    assert linear.weight is not None  # retained by default


def test_apply_matches_the_documented_dequantize_then_gemm_reference():
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    model, linear = _tiny_model("block.linear", config)
    linear.weight.data.normal_()
    convert_model_to_w4a16(model)

    x = torch.randn(4, 64)
    reference = F.linear(
        x,
        w4a16_dequantize(linear._w4a16_codes,
                         linear._w4a16_scales,
                         linear._w4a16_zeros,
                         out_shape=linear._w4a16_weight_shape,
                         out_dtype=x.dtype),
    )
    out, _ = linear(x)
    assert torch.equal(out, reference)


def test_apply_converts_lazily_when_the_loader_hook_never_ran():
    """No conversion hook -> still correct, just later and noisier."""
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    linear = ReplicatedLinear(64, 32, bias=False, quant_config=config, prefix="block.linear")
    linear.weight.data.normal_()
    assert getattr(linear, "_w4a16_codes", None) is None

    x = torch.randn(2, 64)
    with torch.no_grad():
        out, _ = linear(x)
    assert out.shape == (2, 32)
    assert getattr(linear, "_w4a16_codes", None) is not None


def test_grad_enabled_forward_stays_dense():
    """Under grad the master weight must be used -- no frozen dequantized copy."""
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    linear = ReplicatedLinear(64, 32, bias=False, quant_config=config, prefix="block.linear")
    linear.weight.data.normal_()

    x = torch.randn(2, 64)
    out, _ = linear(x)  # grad enabled by default in pytest
    assert out.shape == (2, 32)
    assert getattr(linear, "_w4a16_codes", None) is None


def test_purging_frees_the_dense_weight_when_opted_in():
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"], retain_original_weight=False)
    model, linear = _tiny_model("block.linear", config)
    linear.weight.data.normal_()

    convert_model_to_w4a16(model)

    assert linear.weight is None
    assert "linear.weight" not in model.state_dict()
    assert linear._w4a16_codes is not None
    # apply() must still work off the buffers alone.
    out, _ = linear(torch.randn(2, 64))
    assert out.shape == (2, 32)


def test_untargeted_layer_is_untouched_by_convert():
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    model, linear = _tiny_model("block.linear", config)
    dense = ReplicatedLinear(64, 32, bias=False, prefix="block.other")
    model.add_module("other", dense)
    linear.weight.data.normal_()
    dense.weight.data.normal_()

    before = dense.weight.clone()
    convert_model_to_w4a16(model)

    assert getattr(dense, "_w4a16_codes", None) is None
    assert torch.equal(dense.weight, before)


def test_bias_is_preserved_on_the_quantized_path():
    torch.manual_seed(0)
    config = W4A16Config(target_layers=["block.linear"])
    linear = ReplicatedLinear(64, 32, bias=True, quant_config=config, prefix="block.linear")
    model = nn.Module()
    model.add_module("linear", linear)
    linear.weight.data.normal_()

    convert_model_to_w4a16(model)
    x = torch.randn(3, 64)
    out, out_bias = linear(x)
    # ``skip_bias_add`` is False, so the bias is folded into the output.
    assert out_bias is None
    reference = F.linear(
        x,
        w4a16_dequantize(linear._w4a16_codes,
                         linear._w4a16_scales,
                         linear._w4a16_zeros,
                         out_shape=linear._w4a16_weight_shape,
                         out_dtype=x.dtype), linear.bias)
    assert torch.equal(out, reference)


def test_default_group_size_divides_every_h3_targeted_input_dim():
    """The H3 profile's group size is only valid if it divides all its K dims.

    Documented H3 dims (``MiniMaxH3ArchConfig``): hidden 5376, attention inner
    50 * 128 = 7168, ffn 14336, adaln 2688. This test pins that arithmetic so a
    config change that breaks divisibility fails here rather than at load time.
    """
    h3_input_dims = (5376, 7168, 14336, 2688)
    for dim in h3_input_dims:
        assert dim % DEFAULT_GROUP_SIZE == 0, dim
        assert dim % 2 == 0, dim
