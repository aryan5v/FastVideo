# SPDX-License-Identifier: Apache-2.0
"""Linear-branch scan helpers for MiniMax H3 hybrid attention."""

from __future__ import annotations

import torch

from fastvideo.models.dits.minimax_h3_hybrid.linear import (
    BidirectionalLinearBranch,
    LinearAttentionSepConv,
    factor_delta,
    frame_statistics,
    gather_linear_state,
    initialize_hybrid_parameter,
    run_scans,
    scaled_exponential_write_strength,
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


def test_vdn_solve_keeps_recurrent_factors_fp32() -> None:
    frames, heads, dim = 2, 2, 4
    matrix_a = torch.eye(dim, dtype=torch.bfloat16).expand(frames, heads, dim, dim).contiguous()
    matrix_b = torch.randn(frames, heads, dim, dim, dtype=torch.bfloat16)
    alpha = torch.full((frames, heads, dim), 0.9, dtype=torch.bfloat16)

    transitions, injections = factor_delta("vdn_solve", alpha, matrix_a, matrix_b, tokens_per_frame=8)

    assert transitions.dtype == torch.float32
    assert injections.dtype == torch.float32
    prefix, suffix = run_scans(transitions, injections, text_state=None)
    assert prefix.dtype == torch.float32
    assert suffix.dtype == torch.float32


def test_scaled_exponential_write_strength_starts_at_head_dim_over_tokens() -> None:
    tokens, heads, head_dim = 4032, 3, 128
    logits = torch.zeros(tokens, heads)
    log_scale = torch.zeros(heads)

    gamma = scaled_exponential_write_strength(
        logits,
        log_scale,
        head_dim=head_dim,
        token_count=tokens,
    )

    expected = head_dim / tokens
    torch.testing.assert_close(gamma, torch.full_like(gamma, expected))
    assert abs(float(gamma.mean()) - 0.031746031746031744) < 1e-7


def test_scaled_exponential_write_strength_is_per_head_and_clamped() -> None:
    logits = torch.tensor([[0.0, 1.0], [100.0, -100.0]])
    log_scale = torch.tensor([0.0, 0.5], requires_grad=True)

    gamma = scaled_exponential_write_strength(logits, log_scale, head_dim=4, token_count=8, logit_clamp=2.0)

    expected = 0.5 * torch.exp(torch.tensor([[0.0, 1.5], [2.0, -2.0]]))
    torch.testing.assert_close(gamma, expected)
    gamma[0].sum().backward()
    assert log_scale.grad is not None
    assert torch.count_nonzero(log_scale.grad) == 2


def test_scaled_exponential_write_strength_uses_each_state_token_count() -> None:
    branch = BidirectionalLinearBranch(8, 2, 4, short_conv_targets=(), enable_text_state=True)
    torch.nn.init.zeros_(branch.beta_proj.weight)

    video_gamma = branch._write_strength(torch.zeros(16, 8), token_count=16)
    text_gamma = branch._write_strength(torch.zeros(5, 8), token_count=5)

    torch.testing.assert_close(video_gamma, torch.full_like(video_gamma, 4 / 16))
    torch.testing.assert_close(text_gamma, torch.full_like(text_gamma, 4 / 5))


def test_vdn_solve_one_token_matches_sherman_morrison() -> None:
    torch.manual_seed(3)
    dim = 5
    key = torch.randn(dim, dtype=torch.float64)
    value = torch.randn(dim, dtype=torch.float64)
    gamma = torch.tensor(0.2, dtype=torch.float64)
    matrix_a = (gamma * torch.outer(key, key)).view(1, 1, dim, dim)
    matrix_b = (gamma * torch.outer(value, key)).view(1, 1, dim, dim)
    alpha = torch.full((1, 1, dim), 0.8, dtype=torch.float64)

    transition, injection = factor_delta("vdn_solve", alpha, matrix_a, matrix_b, tokens_per_frame=1)
    inverse = torch.eye(dim, dtype=torch.float64) - (
        gamma * torch.outer(key, key) / (1.0 + gamma * torch.dot(key, key)))

    torch.testing.assert_close(transition.double()[0, 0], alpha[0, 0, :, None] * inverse, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(injection.double()[0, 0], matrix_b[0, 0] @ inverse, atol=2e-6, rtol=2e-6)


def test_vdn_solve_matches_direct_backward_euler_reference() -> None:
    torch.manual_seed(4)
    frames, heads, tokens, dim = 2, 2, 6, 4
    keys = torch.randn(frames, heads, tokens, dim)
    values = torch.randn(frames, heads, tokens, dim)
    gamma = torch.rand(frames, heads, tokens) * 0.1
    alpha = torch.rand(frames, heads, dim).clamp(0.2, 0.9)
    matrix_a, matrix_b = frame_statistics(keys, values, gamma)

    transition, injection = factor_delta("vdn_solve", alpha, matrix_a, matrix_b, tokens)
    inverse = torch.linalg.inv(torch.eye(dim) + matrix_a)

    torch.testing.assert_close(transition, alpha.unsqueeze(-1) * inverse, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(injection, matrix_b @ inverse, atol=2e-5, rtol=2e-5)


def test_vdn_frame_statistics_are_spatial_permutation_invariant() -> None:
    torch.manual_seed(5)
    frames, heads, tokens, dim = 2, 3, 7, 4
    keys = torch.randn(frames, heads, tokens, dim)
    values = torch.randn(frames, heads, tokens, dim)
    gamma = torch.rand(frames, heads, tokens)
    permutation = torch.randperm(tokens)

    actual = frame_statistics(keys, values, gamma)
    permuted = frame_statistics(keys[:, :, permutation], values[:, :, permutation], gamma[:, :, permutation])

    torch.testing.assert_close(actual[0], permuted[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], permuted[1], atol=1e-6, rtol=1e-6)


def test_vdn_branch_write_scale_is_learned_per_layer_and_head() -> None:
    first = BidirectionalLinearBranch(16, 2, 4, short_conv_targets=(), enable_text_state=False)
    second = BidirectionalLinearBranch(16, 2, 4, short_conv_targets=(), enable_text_state=False)

    assert first.write_log_scale.shape == (2,)
    assert second.write_log_scale.shape == (2,)
    assert first.write_log_scale is not second.write_log_scale
    assert first.write_log_scale.requires_grad
    torch.testing.assert_close(first.write_log_scale, torch.zeros_like(first.write_log_scale))

    first.requires_grad_(True)
    torch.nn.init.normal_(first.beta_proj.weight, std=0.02)
    strength = first._write_strength(torch.randn(6, 16), token_count=6)
    strength.square().mean().backward()
    assert first.write_log_scale.grad is not None
    assert first.beta_proj.weight.grad is not None
    assert torch.count_nonzero(first.write_log_scale.grad) == 2
    assert torch.count_nonzero(first.beta_proj.weight.grad) > 0


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
