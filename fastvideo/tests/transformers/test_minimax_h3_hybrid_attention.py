# SPDX-License-Identifier: Apache-2.0
"""Hybrid MiniMax H3 attention body on CPU (no full DiT / attention-backend resolve)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.models.dits.minimax_h3_hybrid.attention import HybridAttention
from fastvideo.models.dits.minimax_h3_hybrid.layout import (
    HybridSequenceLayout,
    window_bounds,
    windows_cover_all_frames,
)
from fastvideo.platforms import AttentionBackendEnum


HIDDEN = 16
HEADS = 2
HEAD_DIM = 8
INNER = HEADS * HEAD_DIM


class _StubParent(nn.Module):
    """QKV + ``to_out`` surface HybridAttention reuses from MiniMaxH3Attention."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = ReplicatedLinear(HIDDEN, INNER, bias=False)
        self.to_k = ReplicatedLinear(HIDDEN, INNER, bias=False)
        self.to_v = ReplicatedLinear(HIDDEN, INNER, bias=False)
        self.to_out = ReplicatedLinear(INNER, HIDDEN, bias=False)


def _init_linears(module: nn.Module) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim <= 1:
                param.zero_()
            else:
                nn.init.xavier_uniform_(param)


def _layout(*, num_frames: int, tokens_per_frame: int = 2, text: int = 3) -> HybridSequenceLayout:
    video = num_frames * tokens_per_frame
    seq = text + video
    return HybridSequenceLayout(
        seq_len=seq,
        video_start=text,
        video_end=seq,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=1,
        frame_width=tokens_per_frame,
        text_start=0,
        text_end=text,
    )


def _identity_norm_rope(query, key, rotary_emb):
    del rotary_emb
    return query, key


def _dense_sdpa_through_parent(parent: _StubParent, hidden: torch.Tensor) -> torch.Tensor:
    query = parent.to_q(hidden)[0].unflatten(-1, (HEADS, HEAD_DIM))
    key = parent.to_k(hidden)[0].unflatten(-1, (HEADS, HEAD_DIM))
    value = parent.to_v(hidden)[0].unflatten(-1, (HEADS, HEAD_DIM))
    heads = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        scale=HEAD_DIM**-0.5,
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)
    return parent.to_out(heads.flatten(2, 3).type_as(hidden))[0]


def test_hybrid_attention_full_cover_matches_dense_sdpa_without_softmax_gate() -> None:
    layout = _layout(num_frames=2)
    bounds = window_bounds(layout.num_frames, radius=1, chunk=5)
    assert windows_cover_all_frames(bounds, layout.num_frames)
    parent = _StubParent()
    hybrid = HybridAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        enable_softmax_gate=False,
        enable_text_state=False,
        short_conv_targets=(),
        branch_parallel=False,
    )
    _init_linears(parent)
    hidden = torch.randn(1, layout.seq_len, HIDDEN)
    actual = hybrid(parent, hidden, None, layout.seq_len, layout, _identity_norm_rope)
    expected = _dense_sdpa_through_parent(parent, hidden)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_hybrid_attention_softmax_gate_scales_window_output() -> None:
    layout = _layout(num_frames=2)
    parent = _StubParent()
    hybrid = HybridAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        enable_softmax_gate=True,
        enable_text_state=False,
        short_conv_targets=(),
        branch_parallel=False,
    )
    _init_linears(parent)
    assert hybrid.softmax_gate is not None
    nn.init.zeros_(hybrid.softmax_gate.up.weight)
    nn.init.zeros_(hybrid.softmax_gate.up.bias)
    hidden = torch.randn(1, layout.seq_len, HIDDEN)
    actual = hybrid(parent, hidden, None, layout.seq_len, layout, _identity_norm_rope)
    expected = _dense_sdpa_through_parent(parent, hidden) * 0.5
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_minimax_h3_attention_requires_hybrid_layout() -> None:
    backend = MagicMock()
    backend.get_name.return_value = "FLASH_ATTN"
    hybrid_config = {
        "window_radius": 1,
        "window_chunk": 5,
        "anchor_frames": "both",
        "delta_rule": "vdn_solve",
        "enable_softmax_gate": False,
        "enable_text_state": False,
        "short_conv_targets": (),
        "branch_parallel": False,
    }
    with patch("fastvideo.models.dits.minimax_h3.get_attn_backend", return_value=backend), patch(
            "fastvideo.models.dits.minimax_h3.DistributedAttention"):
        attn = MiniMaxH3Attention(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            qk_norm_eps=1e-5,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
            quant_config=None,
            prefix="blocks.0.attn",
            hybrid_config=hybrid_config,
        )
    names = attn.state_dict().keys()
    assert any(name.startswith("to_out_linear.") for name in names)
    assert any(name.startswith("linear_attention.") for name in names)
    assert not any(".hybrid." in name or name.startswith("hybrid.") for name in names)
    hidden = torch.randn(1, 6, HIDDEN)
    with pytest.raises(ValueError, match="HybridSequenceLayout"):
        attn.forward(hidden, rotary_emb=None, original_seq_len=6, hybrid_layout=None)
