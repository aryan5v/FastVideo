# SPDX-License-Identifier: Apache-2.0
"""Loader / converter contracts for hybrid MiniMax H3 checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.models.dits.minimax_h3_hybrid.checkpoint import (
    assert_conversion_paths_disjoint,
    hybrid_arch_fields_from_spec,
    lora_scale_from_adapter_config,
    merge_lora_pairs,
    normalize_lora_key,
    remap_vdn_key,
)
from fastvideo.models.loader.utils import get_param_names_mapping


def _map(name: str) -> str:
    mapper = get_param_names_mapping(MiniMaxH3ArchConfig().param_names_mapping)
    target, _, _ = mapper(name)
    return target


def test_orig_to_out_lands_on_fastvideo_linear() -> None:
    assert _map("transformer_blocks.0.attn.orig.to_out.0.weight") == "transformer_blocks.0.attn.to_out.weight"
    assert remap_vdn_key("transformer_blocks.0.attn.orig.to_out.0.weight") == "transformer_blocks.0.attn.to_out.weight"


def test_orig_qkv_and_dense_to_out_zero_still_map() -> None:
    assert _map("transformer_blocks.3.attn.orig.to_q.weight") == "transformer_blocks.3.attn.to_q.weight"
    assert _map("transformer_blocks.3.attn.to_out.0.weight") == "transformer_blocks.3.attn.to_out.weight"
    assert _map("transformer_blocks.3.ff.net.0.proj.weight") == "transformer_blocks.3.ff.fc_in.weight"


def test_hybrid_branch_keys_pass_through() -> None:
    assert remap_vdn_key("transformer_blocks.0.attn.to_out_linear.weight") == (
        "transformer_blocks.0.attn.to_out_linear.weight")
    assert remap_vdn_key("transformer_blocks.0.attn.linear_attention.beta_proj.weight") == (
        "transformer_blocks.0.attn.linear_attention.beta_proj.weight")
    assert remap_vdn_key("transformer_blocks.0.attn.linear_attention.write_log_scale") == (
        "transformer_blocks.0.attn.linear_attention.write_log_scale")
    assert remap_vdn_key("transformer_blocks.0.attn.orig.to_out.1.weight") is None


def test_merge_lora_pairs_adds_b_at_a() -> None:
    weight = torch.zeros(4, 3)
    lora_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lora_b = torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0], [1.0, 1.0]])
    weights = {"transformer_blocks.0.attn.to_q.weight": weight}
    lora = {
        "transformer_blocks.0.attn.orig.to_q.lora_A.default.weight": lora_a,
        "transformer_blocks.0.attn.orig.to_q.lora_B.default.weight": lora_b,
    }
    assert normalize_lora_key("x.lora_A.turbo.weight") == "x.lora_A.weight"
    merged = merge_lora_pairs(weights, lora, scale=0.5)
    assert merged == 1
    expected = 0.5 * (lora_b @ lora_a)
    torch.testing.assert_close(weights["transformer_blocks.0.attn.to_q.weight"], expected)


def test_merge_lora_pairs_accumulates_in_fp32() -> None:
    weight = torch.full((2, 2), 0.1, dtype=torch.bfloat16)
    lora_a = torch.full((1, 2), 1e-3, dtype=torch.float32)
    lora_b = torch.full((2, 1), 1e-2, dtype=torch.float32)
    weights = {"transformer_blocks.0.attn.to_q.weight": weight.clone()}
    lora = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": lora_a,
        "transformer_blocks.0.attn.to_q.lora_B.weight": lora_b,
    }
    assert merge_lora_pairs(weights, lora, scale=1.0) == 1
    expected = (weight.float() + (lora_b @ lora_a)).to(torch.bfloat16)
    torch.testing.assert_close(weights["transformer_blocks.0.attn.to_q.weight"].float(), expected.float())


def test_lora_scale_from_adapter_config_prefers_lora_alpha() -> None:
    assert lora_scale_from_adapter_config({"lora_alpha": 16, "r": 8}, rank=8) == 2.0
    assert lora_scale_from_adapter_config({"alpha": 4, "r": 8}, rank=8) == 0.5
    assert lora_scale_from_adapter_config({"lora_alpha": 16, "alpha": 4, "r": 8}, rank=8) == 2.0
    assert lora_scale_from_adapter_config({"config": {"lora_alpha": 8, "r": 4}}, rank=None) == 2.0
    assert lora_scale_from_adapter_config({}, rank=8) == 1.0


def test_assert_conversion_paths_disjoint(tmp_path: Path) -> None:
    base = tmp_path / "base"
    hybrid = tmp_path / "hybrid"
    dst = tmp_path / "out"
    base.mkdir()
    hybrid.mkdir()
    assert_conversion_paths_disjoint(dst, base, hybrid)
    with pytest.raises(ValueError, match="overlaps source"):
        assert_conversion_paths_disjoint(base, base, hybrid)
    nested = base / "transformer"
    with pytest.raises(ValueError, match="overlaps source"):
        assert_conversion_paths_disjoint(nested, base, hybrid)
    with pytest.raises(ValueError, match="overlaps source"):
        assert_conversion_paths_disjoint(tmp_path, base, hybrid)


def test_hybrid_arch_fields_from_released_spec() -> None:
    spec = {
        "transforms": [{
            "config": {
                "hybrid_attention": {
                    "enable_softmax_gate": True,
                    "anchor_frames": "both",
                    "softmax_attention": {
                        "radius": 1,
                        "chunk": 5
                    },
                    "linear_attention": {
                        "delta_rule": "vdn_solve",
                        "enable_text_state": True,
                        "short_conv": {
                            "targets": ["k", "v"]
                        },
                    },
                }
            }
        }]
    }
    fields = hybrid_arch_fields_from_spec(spec)
    assert fields["hybrid_attention"] is True
    assert fields["hybrid_window_radius"] == 1
    assert fields["hybrid_window_chunk"] == 5
    assert fields["hybrid_anchor_frames"] == "both"
    assert fields["hybrid_delta_rule"] == "vdn_solve"
    assert fields["hybrid_short_conv_targets"] == ["k", "v"]
    assert fields["hybrid_branch_parallel"] is True
