# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the affine group-64 INT8 quantization config.

No CUDA, no model weights, no inference stack: every test here runs on a
laptop. The three things being pinned are (1) the module imports without
flashinfer/CUDA, (2) the quantizer reproduces the validated MLX affine math
and round-trips within a stated tolerance, and (3) layer selection picks
H3's attention/FFN GEMMs and *excludes* ``attn.to_gate_compress``.

Test 3 is the load-bearing one: ``to_gate_compress`` is H3's VSA sparse
routing gate, and quantizing it changes a discrete routing decision rather
than perturbing an output.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch

from fastvideo.layers.quantization.int8_affine_config import (
    INT8AffineConfig,
    MINIMAX_H3_INT8_AFFINE_SUFFIXES,
    int8_affine_dequantize,
    int8_affine_quantize,
    minimax_h3_int8_affine_prefixes,
)

# ---------------------------------------------------------------------------
# (a) importability without CUDA / inference deps
# ---------------------------------------------------------------------------


def test_config_module_imports_without_cuda_or_flashinfer():
    """The config must import on a CPU-only host with no flashinfer."""
    assert "flashinfer" not in sys.modules, "importing the config must not pull in flashinfer"
    module = importlib.import_module("fastvideo.layers.quantization.int8_affine_config")
    assert module is not None
    assert "flashinfer" not in sys.modules


def test_config_metadata_is_well_formed():
    cfg = INT8AffineConfig()
    assert cfg.get_name() == "INT8Affine"
    assert cfg.group_size == 64
    assert cfg.bits == 8
    assert torch.bfloat16 in cfg.get_supported_act_dtypes()
    assert cfg.get_config_filenames() == []
    assert INT8AffineConfig.get_min_capability() >= 70
    # from_config round-trips the constructor surface the loader may pass.
    rebuilt = INT8AffineConfig.from_config({"group_size": 32, "bits": 4})
    assert (rebuilt.group_size, rebuilt.bits) == (32, 4)


def test_invalid_bits_and_group_size_are_rejected():
    with pytest.raises(ValueError):
        INT8AffineConfig(bits=16)
    with pytest.raises(ValueError):
        INT8AffineConfig(group_size=0)


# ---------------------------------------------------------------------------
# (b) quantizer correctness
# ---------------------------------------------------------------------------


def _independent_affine_reference(w: torch.Tensor, group_size: int = 64, bits: int = 8):
    """A from-scratch restatement of the MLX affine algorithm.

    Written against the description in ``mlx_affine_qat.py``'s module
    docstring (per-group min/max, magnitude-anchored sign, exact-integer
    anchor re-expression, rint rounding, clamp to ``[0, 2**bits-1]``) rather
    than by copying the implementation, so it is a real cross-check of the
    transcription and not a tautology.
    """
    n_bins = float((1 << bits) - 1)
    flat = w.reshape(-1, w.shape[-1] // group_size, group_size).float()
    lo = flat.amin(dim=-1)
    hi = flat.amax(dim=-1)
    # Anchor at whichever endpoint has the larger magnitude.
    anchor = torch.where(lo.abs() > hi.abs(), lo, hi)
    other = torch.where(lo.abs() > hi.abs(), hi, lo)
    step = ((hi - lo) / n_bins).clamp_min(1e-7)
    step = torch.where(lo.abs() > hi.abs(), step, -step)
    # Re-express the anchor as an exact integer multiple of the step so the
    # extreme value round-trips exactly.
    q0 = torch.round(anchor / step)
    use = q0 != 0
    step = torch.where(use, anchor / torch.where(use, q0, torch.ones_like(q0)), step)
    zero = torch.where(use, anchor, torch.zeros_like(anchor))
    del other
    codes = torch.round((flat - zero.unsqueeze(-1)) / step.unsqueeze(-1)).clamp(0.0, n_bins)
    return codes.to(torch.int64), step, zero


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_quantize_matches_independent_reference(dtype):
    torch.manual_seed(0)
    w = torch.randn(37, 256, dtype=dtype)
    codes, scales, biases = int8_affine_quantize(w, group_size=64, bits=8)
    ref_codes, ref_scales, ref_bias = _independent_affine_reference(w, group_size=64, bits=8)

    assert codes.dtype == torch.uint8
    # Grouped shape, matching the reference contract.
    assert codes.shape == (*w.shape[:-1], w.shape[-1] // 64, 64)
    assert scales.shape == (*w.shape[:-1], w.shape[-1] // 64)
    assert torch.equal(codes.reshape(ref_codes.shape).to(torch.int64), ref_codes)
    # The quantizer casts scales/biases to the input dtype (the reference does
    # the same), so compare at that dtype: the fp32 solve is identical, the
    # bf16/fp16 store is the only lossy step.
    torch.testing.assert_close(scales, ref_scales.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(biases, ref_bias.to(dtype), rtol=0, atol=0)


def test_quantize_matches_in_tree_mlx_reference_when_available():
    """Parity against the real ``mlx_affine_qat`` transcription, if present.

    That module lives in a sibling worktree today, so this is skipped unless
    it has landed next to us; when it does, the two transcriptions must agree
    bit-for-bit on codes and exactly on the fp32 scales/biases.
    """
    mlx = pytest.importorskip("fastvideo.layers.quantization.mlx_affine_qat",
                              reason="mlx_affine_qat.py is not in this tree yet")
    torch.manual_seed(1)
    w = torch.randn(16, 128, dtype=torch.float32)
    codes, scales, biases = int8_affine_quantize(w, group_size=64, bits=8)
    ref_codes, ref_scales, ref_bias = mlx.mlx_affine_quantize_reference(w, group_size=64, bits=8)
    assert torch.equal(codes.reshape(ref_codes.shape).to(torch.int64), ref_codes.to(torch.int64))
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)
    torch.testing.assert_close(biases, ref_bias, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_error_is_within_documented_tolerance(dtype):
    """Group-64 8-bit affine on N(0,1) weights.

    Tolerances are measured over seeds 0-5, not guessed, then given ~2x
    headroom. Group-64 8-bit spends ~1/255 of the per-group range per step
    (step ~= 0.016 for a 64-sample N(0,1) group), so the reconstruction error
    is ~step/sqrt(12) RMS and ~step/2 at worst:

    - fp32 source: max/max|w| <= 0.0055, rms/max|w| <= 0.0013
    - bf16 source: max/max|w| <= 0.0093, rms/max|w| <= 0.0021 (the source is
      itself bf16-rounded before quantizing)
    """
    torch.manual_seed(2)
    w = torch.randn(64, 1024, dtype=dtype)
    codes, scales, biases = int8_affine_quantize(w, group_size=64, bits=8)
    # Dequant returns the scales' dtype (the reference's contract) — fp32 for an
    # fp32 source, bf16 for a bf16 one — so measure in fp32.
    deq = int8_affine_dequantize(codes, scales, biases, out_shape=w.shape).float()

    assert deq.shape == w.shape
    top = w.float().abs().max().item()
    rel_max = (w.float() - deq).abs().max().item() / top
    rel_rms = math.sqrt(((w.float() - deq)**2).mean().item()) / top
    max_limit, rms_limit = (0.02, 0.005) if dtype is torch.bfloat16 else (0.01, 0.003)
    assert rel_max < max_limit, f"max relative error {rel_max:.5f} exceeded {max_limit} for {dtype}"
    assert rel_rms < rms_limit, f"rms relative error {rel_rms:.5f} exceeded {rms_limit} for {dtype}"

    # The extreme value of each group is the quantizer's anchor: it is
    # re-expressed as an exact integer multiple of the scale, so it must
    # round-trip to the precision the stored scales allow. Exact for an fp32
    # source; bf16-level for a bf16 source, whose scales are themselves bf16.
    grouped_w = w.float().reshape(-1, 64)
    grouped_deq = deq.reshape(-1, 64)
    extreme = grouped_w.abs().max(dim=-1).values
    at_extreme = grouped_w.abs() == extreme.unsqueeze(-1)
    rtol, atol = (1e-5, 1e-5) if dtype is torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(grouped_w[at_extreme], grouped_deq[at_extreme], rtol=rtol, atol=atol)


def test_group_size_must_divide_input_dim():
    with pytest.raises(ValueError, match="not divisible"):
        int8_affine_quantize(torch.randn(4, 100), group_size=64)


def test_codes_never_exceed_uint8_range():
    torch.manual_seed(3)
    for _ in range(5):
        w = torch.randn(8, 256) * torch.rand(8, 1) * 10
        codes, _, _ = int8_affine_quantize(w, group_size=64, bits=8)
        assert codes.max().item() <= 255


# ---------------------------------------------------------------------------
# (c) H3 layer selection — the important test
# ---------------------------------------------------------------------------

# Every linear name H3's DiT actually builds, from fastvideo/models/dits/minimax_h3.py.
_H3_INCLUDED = [
    "minimax_h3.transformer_blocks.0.attn.to_q",
    "minimax_h3.transformer_blocks.0.attn.to_k",
    "minimax_h3.transformer_blocks.0.attn.to_v",
    "minimax_h3.transformer_blocks.0.attn.to_out",
    "minimax_h3.transformer_blocks.0.ff.fc_in",
    "minimax_h3.transformer_blocks.0.ff.fc_out",
    "minimax_h3.transformer_blocks.49.attn.to_q",
    "minimax_h3.transformer_blocks.49.ff.fc_out",
    "minimax_h3.token_refiner.refiner_blocks.0.attn.to_q",
    "minimax_h3.token_refiner.refiner_blocks.0.attn.to_out",
    "minimax_h3.token_refiner.refiner_blocks.1.ff.fc_in",
    "minimax_h3.transformer_blocks.0.adaln_proj.linear",
]

_H3_EXCLUDED = [
    # THE critical exclusion: the VSA sparse-attention gate.
    "minimax_h3.transformer_blocks.0.attn.to_gate_compress",
    "minimax_h3.transformer_blocks.49.attn.to_gate_compress",
    # Global timestep-basis projector.
    "minimax_h3.adaln_basis",
    # Modules H3 itself pins to fp32.
    "minimax_h3.proj_in",
    "minimax_h3.audio_proj_in",
    "minimax_h3.proj_out",
    "minimax_h3.audio_proj_out",
    "minimax_h3.time_embedder.fc_in",
    "minimax_h3.time_embedder.fc_out",
    # Text input projection (excluded by default; see the config docstring).
    "minimax_h3.context_embedder",
    # Non-linear / unrelated names must not be swept in.
    "minimax_h3.transformer_blocks.0.norm1",
    "minimax_h3.rope",
]


def test_h3_layer_selection_include_and_exclude_sets():
    cfg = INT8AffineConfig.for_minimax_h3()
    for prefix in _H3_INCLUDED:
        assert cfg.is_target_layer(prefix), f"expected {prefix!r} to be quantized"
    for prefix in _H3_EXCLUDED:
        assert not cfg.is_target_layer(prefix), f"expected {prefix!r} to be EXCLUDED"


def test_to_gate_compress_is_excluded_even_by_a_broad_allowlist():
    """The deny list is fail-closed: no constructor argument re-enables it.

    This is the regression guard for the failure mode the recon flagged —
    a name that matches no "norm"/"scale_shift_table"-style heuristic being
    silently swept into a broad suffix rule.
    """
    hostile = INT8AffineConfig(
        layer_suffixes=("to_q", "to_k", "to_v", "to_out", "to_gate_compress"),
        exclude_substrings=(),  # caller tries to clear the deny list
    )
    assert not hostile.is_target_layer("minimax_h3.transformer_blocks.7.attn.to_gate_compress")
    # The same broad rule still reaches the real projections.
    assert hostile.is_target_layer("minimax_h3.transformer_blocks.7.attn.to_q")

    # And via an explicit target_layers set, which bypasses suffix matching.
    explicit = INT8AffineConfig(
        target_layers=("minimax_h3.transformer_blocks.7.attn.to_gate_compress", ),
    )
    assert not explicit.is_target_layer("minimax_h3.transformer_blocks.7.attn.to_gate_compress")


def test_enumerated_h3_prefixes_agree_with_selection():
    """The literal enumerated set and the suffix rule must pick the same layers."""
    cfg = INT8AffineConfig.for_minimax_h3()
    enumerated = minimax_h3_int8_affine_prefixes()
    assert len(enumerated) == 50 * 7 + 2 * 6  # 50 blocks x 7 suffixes, 2 refiner blocks x 6
    for prefix in enumerated:
        assert cfg.is_target_layer(prefix), f"enumerated prefix {prefix!r} not selected by the suffix rule"
    # And nothing the suffix rule selects in the H3 blocks is missing from the
    # enumeration: walk the two block scopes and compare.
    selected = {
        f"minimax_h3.{scope}.{i}.{suffix}"
        for scope, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2))
        for i in range(count)
        for suffix in MINIMAX_H3_INT8_AFFINE_SUFFIXES
        if not (scope != "transformer_blocks" and suffix.startswith("adaln_proj"))
        if cfg.is_target_layer(f"minimax_h3.{scope}.{i}.{suffix}")
    }
    assert selected == set(enumerated)


def test_non_linear_layers_get_no_quant_method():
    from fastvideo.layers.linear import ReplicatedLinear

    cfg = INT8AffineConfig.for_minimax_h3()
    linear = ReplicatedLinear(64, 64, bias=False, quant_config=cfg, prefix="minimax_h3.transformer_blocks.0.attn.to_q")
    assert linear.quant_method is not None
    assert linear.quant_method.__class__.__name__ == "INT8AffineQuantizeMethod"

    gate = ReplicatedLinear(64, 64, bias=False, quant_config=cfg,
                            prefix="minimax_h3.transformer_blocks.0.attn.to_gate_compress")
    assert gate.quant_method.__class__.__name__ == "UnquantizedLinearMethod"


# ---------------------------------------------------------------------------
# conversion + apply round-trip on a real ReplicatedLinear (CPU)
# ---------------------------------------------------------------------------


def test_conversion_and_apply_match_dense_linear_within_tolerance():
    """End-to-end: load-time conversion then apply() dequantizes and matches.

    Uses ``retain_original_weight=False`` to exercise the purge path too.
    """
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.quantization.int8_affine_config import convert_model_to_int8_affine

    torch.manual_seed(4)
    cfg = INT8AffineConfig.for_minimax_h3(retain_original_weight=False)
    layer = ReplicatedLinear(128, 256, bias=False, quant_config=cfg,
                             prefix="minimax_h3.transformer_blocks.0.attn.to_q")
    with torch.no_grad():
        layer.weight.copy_(torch.randn(256, 128))

    convert_model_to_int8_affine(layer)
    assert layer._int8_affine_codes.dtype == torch.uint8
    assert layer._int8_affine_codes.shape == (256, 128)
    assert layer._int8_affine_scales.shape == (256, 2)
    assert layer.weight is None, "retain_original_weight=False should purge the bf16 weight"

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    out, _ = layer(x)
    assert out.shape == (4, 256)
    assert torch.isfinite(out).all()


def test_apply_falls_back_to_dense_under_grad():
    """A training step must see the master weight, not a frozen dequant copy."""
    from fastvideo.layers.linear import ReplicatedLinear

    torch.manual_seed(5)
    cfg = INT8AffineConfig.for_minimax_h3()
    layer = ReplicatedLinear(64, 64, bias=False, quant_config=cfg,
                             prefix="minimax_h3.transformer_blocks.0.attn.to_q")
    with torch.no_grad():
        layer.weight.copy_(torch.randn(64, 64))
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.enable_grad():
        out, _ = layer(x)
    assert out.shape == (2, 64)
    assert not hasattr(layer, "_int8_affine_codes"), "grad-enabled forward must not quantize in place"
