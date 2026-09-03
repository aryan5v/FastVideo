# SPDX-License-Identifier: Apache-2.0
"""Loader / converter contracts for hybrid MiniMax H3 checkpoints."""

from __future__ import annotations

import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.models.dits.minimax_h3_hybrid.checkpoint import (
    hybrid_arch_fields_from_spec,
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
