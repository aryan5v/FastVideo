# SPDX-License-Identifier: Apache-2.0
"""Linear-branch scan helpers for MiniMax H3 hybrid attention."""

from __future__ import annotations

import torch

from fastvideo.models.dits.minimax_h3_hybrid.linear import (
    LinearAttentionSepConv,
    factor_delta,
    frame_statistics,
    gather_linear_state,
    initialize_hybrid_parameter,
    run_scans,
)


def test_frame_statistics_and_sana_factor_shapes() -> None:
    torch.manual_seed(0)
    frames, heads, tokens, dim = 3, 2, 4, 8
    keys = torch.randn(frames, heads, tokens, dim)
    values = torch.randn(frames, heads, tokens, dim)
    beta = torch.rand(frames, heads, tokens)
    matrix_a, matrix_b = frame_statistics(keys, values, beta, a_fp32=True)
    assert matrix_a.shape == (frames, heads, dim, dim)
    assert matrix_b.shape == (frames, heads, dim, dim)
    alpha = torch.rand(frames, heads, dim).clamp(min=0.1, max=0.9)
    transitions, injections = factor_delta("sana_scaled", alpha, matrix_a, matrix_b, tokens)
    assert transitions.shape == injections.shape == (frames, heads, dim, dim)


def test_run_scans_identity_transition_accumulates_injections() -> None:
    frames, heads, dim = 4, 2, 3
    eye = torch.eye(dim).expand(frames, heads, dim, dim).contiguous()
    injections = torch.randn(frames, heads, dim, dim)
    prefix, suffix = run_scans(eye.clone(), injections, text_state=None)
    expected_prefix = injections.clone()
    expected_prefix[1] = injections[0] + injections[1]
    expected_prefix[2] = expected_prefix[1] + injections[2]
    expected_prefix[3] = expected_prefix[2] + injections[3]
    torch.testing.assert_close(prefix, expected_prefix, atol=1e-5, rtol=1e-5)
    expected_suffix = injections.clone()
    expected_suffix[2] = injections[3] + injections[2]
    expected_suffix[1] = expected_suffix[2] + injections[1]
    expected_suffix[0] = expected_suffix[1] + injections[0]
    torch.testing.assert_close(suffix, expected_suffix, atol=1e-5, rtol=1e-5)


def test_run_scans_supports_autograd() -> None:
    frames, heads, dim = 3, 1, 2
    transitions = torch.randn(frames, heads, dim, dim, requires_grad=True)
    injections = torch.randn(frames, heads, dim, dim, requires_grad=True)

    prefix, suffix = run_scans(transitions, injections, text_state=None)
    (prefix.square().mean() + suffix.square().mean()).backward()

    assert transitions.grad is not None
    assert injections.grad is not None
    assert torch.isfinite(transitions.grad).all()
    assert torch.isfinite(injections.grad).all()


def test_gather_linear_state_radius_zero_uses_neighbours() -> None:
    frames, heads, dim = 3, 1, 2
    prefix = torch.randn(frames, heads, dim, dim)
    suffix = torch.randn(frames, heads, dim, dim)
    alpha = torch.ones(frames, heads, dim)
    bounds = [(0, 0), (1, 1), (2, 2)]
    state = gather_linear_state(prefix, suffix, alpha, bounds, text_state=None)
    # Query frame 1 sees neither itself in the far branch: before=prefix[0], after=suffix[2].
    torch.testing.assert_close(state[1], prefix[0] + suffix[2], atol=1e-5, rtol=1e-5)
    # Query frame 0 has no before; after = suffix[1].
    torch.testing.assert_close(state[0], suffix[1], atol=1e-5, rtol=1e-5)


def test_sep_conv_apply_conv_does_not_shadow_module_apply() -> None:
    conv = LinearAttentionSepConv(8, targets=("k", ))
    seen: list[str] = []
    conv.apply(lambda module: seen.append(type(module).__name__))
    assert "LinearAttentionSepConv" in seen
    tokens = torch.randn(4, 2, 4)
    skipped = conv.apply_conv("q", tokens, 4, (1, 1))
    torch.testing.assert_close(skipped, tokens)
    out = conv.apply_conv("k", tokens, 4, (1, 1))
    assert out.shape == tokens.shape


def test_identity_initialized_sep_conv_has_live_gradients() -> None:
    conv = LinearAttentionSepConv(4, targets=("k", ))
    prefix = "transformer_blocks.0.attn.linear_attention.short_conv"
    with torch.no_grad():
        spatial = initialize_hybrid_parameter(f"{prefix}.k_sp.weight", conv.k_sp.weight.shape, torch.device("cpu"),
                                              torch.float32)
        temporal = initialize_hybrid_parameter(f"{prefix}.k_tm.weight", conv.k_tm.weight.shape,
                                               torch.device("cpu"), torch.float32)
        assert spatial is not None and temporal is not None
        conv.k_sp.weight.copy_(spatial)
        conv.k_tm.weight.copy_(temporal)

    tokens = torch.randn(8, 2, 2)
    conv.apply_conv("k", tokens, num_frames=2, frame_size=(2, 2)).square().mean().backward()

    assert conv.k_sp.weight.grad is not None
    assert conv.k_tm.weight.grad is not None
    assert torch.count_nonzero(conv.k_sp.weight.grad) > 0
    assert torch.count_nonzero(conv.k_tm.weight.grad) > 0
