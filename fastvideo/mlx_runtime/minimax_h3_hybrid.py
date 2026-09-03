# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""MLX window softmax + linear scan for MiniMax H3 hybrid checkpoints.

Reuses ``mx.fast.scaled_dot_product_attention`` for chunk-aligned windows and
batched GEMMs for the frame scan. No CUDA/Triton/FlexAttention. Hybrid keys are
detected from the block weight dict; dense FastH3 checkpoints take the existing
full-attention path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastvideo.models.dits.minimax_h3_hybrid.layout import window_bounds, windows_cover_all_frames

TEXT_STATE_SCALE = 0.5
SHORT_CONV_KERNEL = 5


def is_hybrid_block(weights: dict[str, Any]) -> bool:
    return "attn.to_out_linear.weight" in weights or "attn.linear_attention.alpha.down.weight" in weights


def _hybrid_geometry(layout, patch_size: tuple[int, int, int]) -> dict[str, int]:
    patch_t, patch_h, patch_w = patch_size
    num_frames = layout.num_video_latent_frames // patch_t
    frame_height = layout.latent_height // patch_h
    frame_width = layout.latent_width // patch_w
    tokens_per_frame = frame_height * frame_width
    video_end = layout.sequence_length
    video_start = video_end - tokens_per_frame * num_frames
    text_indices = np.asarray(layout.text_indices)
    return {
        "seq_len": layout.sequence_length,
        "video_start": int(video_start),
        "video_end": int(video_end),
        "num_frames": int(num_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "text_start": int(text_indices.min()) if text_indices.size else 0,
        "text_end": int(text_indices.max()) + 1 if text_indices.size else 0,
    }


def _sdpa(q, k, v, scale: float):
    import mlx.core as mx

    attended = mx.fast.scaled_dot_product_attention(
        mx.contiguous(q.transpose(1, 0, 2))[None],
        mx.contiguous(k.transpose(1, 0, 2))[None],
        mx.contiguous(v.transpose(1, 0, 2))[None],
        scale=scale,
    )[0]
    return mx.contiguous(attended.transpose(1, 0, 2))


def _window_softmax(query, key, value, geom: dict[str, int], bounds, scale: float, anchor_frames: str):
    import mlx.core as mx

    seq_len = geom["seq_len"]
    video_start, video_end = geom["video_start"], geom["video_end"]
    num_frames, per_frame = geom["num_frames"], geom["tokens_per_frame"]
    heads, head_dim = query.shape[1], query.shape[2]
    out = mx.zeros_like(query)
    global_idx = np.concatenate([np.arange(video_start), np.arange(video_end, seq_len)])
    if global_idx.size:
        gi = mx.array(global_idx)
        out = out.at[gi].set(_sdpa(query[gi], key, value, scale))

    video_query = query[video_start:video_end].reshape(num_frames, per_frame, heads, head_dim)
    video_key = key[video_start:video_end].reshape(num_frames, per_frame, heads, head_dim)
    video_value = value[video_start:video_end].reshape(num_frames, per_frame, heads, head_dim)
    dense_rows = {0, num_frames - 1} if anchor_frames in ("rows", "both") else set()
    dense_cols = {0, num_frames - 1} if anchor_frames in ("columns", "both") else set()
    groups: dict[tuple[int, ...], list[int]] = {}
    for frame, (lo, hi) in enumerate(bounds):
        lo_c, hi_c = max(lo, 0), min(hi, num_frames - 1)
        if frame in dense_rows:
            key_frames = tuple(range(num_frames))
        else:
            key_frames = tuple(sorted(set(range(lo_c, hi_c + 1)) | dense_cols))
        groups.setdefault(key_frames, []).append(frame)

    global_key = key[mx.array(global_idx)] if global_idx.size else None
    global_value = value[mx.array(global_idx)] if global_idx.size else None
    for key_frames, query_frames in groups.items():
        q_rows = mx.concatenate([video_query[f].reshape(-1, heads, head_dim) for f in query_frames], axis=0)
        k_parts = [] if global_key is None else [global_key]
        v_parts = [] if global_value is None else [global_value]
        if key_frames:
            k_parts.append(mx.concatenate([video_key[f].reshape(-1, heads, head_dim) for f in key_frames], axis=0))
            v_parts.append(mx.concatenate([video_value[f].reshape(-1, heads, head_dim) for f in key_frames], axis=0))
        attended = _sdpa(q_rows, mx.concatenate(k_parts, axis=0), mx.concatenate(v_parts, axis=0), scale)
        cursor = 0
        for frame in query_frames:
            start = video_start + frame * per_frame
            chunk = attended[cursor:cursor + per_frame]
            out = out.at[start:start + per_frame].set(chunk)
            cursor += per_frame
    return out


def _silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


def _l2norm(x, eps: float = 1e-6):
    import mlx.core as mx

    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def _sep_conv(weights: dict[str, Any], proj: str, tokens, geom: dict[str, int]):
    """Depthwise 5x5 spatial + 5-tap temporal conv when the checkpoint has taps."""
    import mlx.core as mx

    spatial_key = f"attn.linear_attention.short_conv.{proj}_sp.weight"
    temporal_key = f"attn.linear_attention.short_conv.{proj}_tm.weight"
    if spatial_key not in weights:
        return tokens
    heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    num_frames = geom["num_frames"]
    grid_h, grid_w = geom["frame_height"], geom["frame_width"]
    channels = heads * head_dim
    # MLX conv2d is NHWC; PyTorch depthwise weight is (C, 1, K, K).
    volume = tokens.reshape(num_frames, grid_h, grid_w, channels)
    weight_sp = weights[spatial_key]
    if weight_sp.ndim == 4:
        kernel = weight_sp.transpose(0, 2, 3, 1)
        volume = mx.conv2d(volume, kernel, padding=SHORT_CONV_KERNEL // 2, groups=channels)
    frames = volume.reshape(num_frames, grid_h * grid_w, channels)
    weight_t = weights[temporal_key]
    if weight_t.ndim == 3:
        weight_t = weight_t.squeeze(1)
    pad = SHORT_CONV_KERNEL // 2
    padded = mx.pad(frames, [(pad, pad), (0, 0), (0, 0)])
    out = None
    for tap in range(SHORT_CONV_KERNEL):
        part = padded[tap:tap + num_frames] * weight_t[:, tap]
        out = part if out is None else out + part
    assert out is not None
    return out.reshape(-1, heads, head_dim)


def _activate(tokens, *, l2norm: bool):
    activated = _silu(tokens)
    return _l2norm(activated) if l2norm else activated


def _kda_alpha(weights: dict[str, Any], frame_mean, num_frames: int, heads: int, head_dim: int):
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import linear

    delta = linear(frame_mean, weights["attn.linear_attention.alpha.down.weight"])
    delta = linear(delta, weights["attn.linear_attention.alpha.up.weight"])
    delta = (delta + weights["attn.linear_attention.alpha.dt_bias"]).reshape(num_frames, heads, head_dim)
    a_log = weights["attn.linear_attention.alpha.A_log"]
    return mx.exp(-mx.exp(a_log)[:, None] * mx.log(1 + mx.exp(delta)))


def _factor_delta(delta_rule: str, alpha, matrix_a, matrix_b, per_frame: int):
    import mlx.core as mx

    eye = mx.eye(matrix_a.shape[-1], dtype=matrix_a.dtype)
    if delta_rule == "sana_scaled":
        inv = 1.0 / per_frame
        return alpha[..., None] * (eye - inv * matrix_a), (inv**0.5) * matrix_b
    scaled = 1.0 / per_frame if delta_rule == "vdn_scaled" else 1.0
    inverse = mx.linalg.inv(matrix_a.astype(mx.float32) * scaled + eye.astype(mx.float32), stream=mx.cpu)
    transitions = (alpha[..., None] * inverse).astype(matrix_a.dtype)
    injections = ((matrix_b.astype(mx.float32) * (scaled**0.5)) @ inverse).astype(matrix_b.dtype)
    return transitions, injections


def _text_state(weights: dict[str, Any], text_x, text_qkv, *, delta_rule: str, heads: int, head_dim: int):
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import linear

    length = text_qkv[1].shape[0]
    key = _activate(text_qkv[1], l2norm=True).reshape(1, length, heads, head_dim).transpose(0, 2, 1, 3)
    value = _activate(text_qkv[2], l2norm=False).reshape(1, length, heads, head_dim).transpose(0, 2, 1, 3)
    beta = mx.sigmoid(linear(text_x, weights["attn.linear_attention.beta_proj.weight"]))
    beta = beta.reshape(1, length, heads).transpose(0, 2, 1)
    weighted = key * beta[..., None]
    matrix_a = mx.matmul(weighted.transpose(0, 1, 3, 2), key)
    matrix_b = mx.matmul(value.transpose(0, 1, 3, 2), weighted)
    ones = mx.ones((1, heads, head_dim), dtype=matrix_a.dtype)
    _, injection = _factor_delta(delta_rule, ones, matrix_a, matrix_b, length)
    return TEXT_STATE_SCALE * injection[0]


def _gather_state(prefix, suffix, alpha, bounds, start, text_state):
    import mlx.core as mx

    num_frames = prefix.shape[0]
    log_alpha = mx.log(mx.maximum(alpha, 1e-12))
    zeros = mx.zeros_like(log_alpha[:1])
    log_prefix = mx.concatenate([zeros, mx.cumsum(log_alpha, axis=0)], axis=0)
    gathered = []
    for frame, (lo, hi) in enumerate(bounds):
        last_before = lo - 1
        first_after = hi + 1
        fallback = start if text_state is None else text_state
        before = prefix[last_before] if last_before >= 0 else fallback
        after = suffix[first_after] if first_after < num_frames else fallback
        bridge_before = max(last_before + 1, 0)
        bridge_after = min(first_after, num_frames)
        alpha_from_before = mx.exp(log_prefix[frame + 1] - log_prefix[bridge_before])
        alpha_from_after = mx.exp(log_prefix[bridge_after] - log_prefix[frame])
        gathered.append(before * alpha_from_before[:, None] + after * alpha_from_after[:, None])
    return mx.stack(gathered)


def _linear_scan(weights: dict[str, Any],
                 video_x,
                 qkv_raw,
                 geom: dict[str, int],
                 bounds,
                 *,
                 skip_ends: bool,
                 text_x=None,
                 text_qkv=None,
                 delta_rule: str):
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import linear

    num_frames, per_frame = geom["num_frames"], geom["tokens_per_frame"]
    heads = qkv_raw[0].shape[1]
    head_dim = qkv_raw[0].shape[2]
    query, key, value = qkv_raw
    query = _sep_conv(weights, "q", query, geom)
    key = _sep_conv(weights, "k", key, geom)
    value = _sep_conv(weights, "v", value, geom)
    query = _activate(query, l2norm=True)
    key = _activate(key, l2norm=True)
    value = _activate(value, l2norm=False)
    if skip_ends and num_frames > 2:
        query, key, value = query[per_frame:-per_frame], key[per_frame:-per_frame], value[per_frame:-per_frame]
        video_x = video_x[per_frame:-per_frame]
        num_frames = num_frames - 2
        bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]
    query = query.reshape(num_frames, per_frame, heads, head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(num_frames, per_frame, heads, head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(num_frames, per_frame, heads, head_dim).transpose(0, 2, 1, 3)
    beta = mx.sigmoid(linear(video_x, weights["attn.linear_attention.beta_proj.weight"]))
    beta = beta.reshape(num_frames, per_frame, heads).transpose(0, 2, 1)
    weighted = key * beta[..., None]
    matrix_a = mx.matmul(weighted.transpose(0, 1, 3, 2), key)
    matrix_b = mx.matmul(value.transpose(0, 1, 3, 2), weighted)
    frame_mean = video_x.reshape(num_frames, per_frame, -1).mean(axis=1)
    alpha = _kda_alpha(weights, frame_mean, num_frames, heads, head_dim)
    transitions, injections = _factor_delta(delta_rule, alpha, matrix_a, matrix_b, per_frame)

    text_state = None
    if text_x is not None and text_qkv is not None:
        text_state = _text_state(weights, text_x, text_qkv, delta_rule=delta_rule, heads=heads, head_dim=head_dim)
    start = mx.zeros((heads, head_dim, head_dim), dtype=injections.dtype) if text_state is None else text_state

    def _scan(initial):
        states = []
        state = initial
        for frame in range(num_frames):
            state = injections[frame] + mx.matmul(state, transitions[frame])
            states.append(state)
        return mx.stack(states)

    prefix = _scan(start)
    suffix_states = []
    state = start
    for frame in range(num_frames - 1, -1, -1):
        state = injections[frame] + mx.matmul(state, transitions[frame])
        suffix_states.append(state)
    suffix = mx.stack(list(reversed(suffix_states)))
    state_bank = _gather_state(prefix, suffix, alpha, bounds, start, text_state)
    readout = mx.matmul(query.astype(mx.float32), state_bank.transpose(0, 1, 3, 2)).astype(query.dtype)
    ms = mx.mean(readout.astype(mx.float32)**2, axis=-1, keepdims=True)
    readout = readout * mx.rsqrt(ms + 1e-6).astype(readout.dtype) * weights["attn.linear_attention.norm.weight"]
    gate_hidden = video_x
    if "attn.linear_attention.output_gate.down.weight" in weights:
        gate_hidden = linear(gate_hidden, weights["attn.linear_attention.output_gate.down.weight"])
    gate = mx.sigmoid(
        linear(gate_hidden, weights["attn.linear_attention.output_gate.up.weight"],
               weights.get("attn.linear_attention.output_gate.up.bias")))
    gate = gate.reshape(num_frames, per_frame, heads, head_dim).transpose(0, 2, 1, 3)
    gated = (readout * gate).transpose(0, 2, 1, 3).reshape(-1, heads * head_dim)
    if skip_ends and geom["num_frames"] > 2:
        full = mx.zeros((geom["num_frames"] * per_frame, heads * head_dim), dtype=gated.dtype)
        return full.at[per_frame:-per_frame].set(gated)
    return gated


def hybrid_attention(
    weights: dict[str, Any],
    hidden_states,
    query,
    key,
    value,
    query_raw,
    key_raw,
    value_raw,
    layout,
    patch_size: tuple[int, int, int],
    *,
    num_heads: int,
    head_dim: int,
    window_radius: int = 1,
    window_chunk: int = 5,
    anchor_frames: str = "both",
    delta_rule: str = "vdn_solve",
    enable_text_state: bool = True,
):
    """Window softmax + linear far branch, then the existing ``to_out`` projection."""
    import mlx.core as mx

    from fastvideo.mlx_runtime.fastwan import linear

    geom = _hybrid_geometry(layout, patch_size)
    bounds = window_bounds(geom["num_frames"], window_radius, window_chunk)
    scale = head_dim**-0.5
    full_cover = windows_cover_all_frames(bounds, geom["num_frames"])
    if full_cover:
        softmax = _sdpa(query, key, value, scale)
    else:
        softmax = _window_softmax(query, key, value, geom, bounds, scale, anchor_frames)
    if "attn.softmax_gate.up.weight" in weights:
        gate = mx.sigmoid(
            linear(hidden_states, weights["attn.softmax_gate.up.weight"], weights.get("attn.softmax_gate.up.bias")))
        gate = gate.reshape(-1, num_heads, 1)
        softmax = softmax * gate
    flat = softmax.reshape(hidden_states.shape[0], num_heads * head_dim)
    to_out_key = "attn.to_out.weight" if "attn.to_out.weight" in weights else "attn.to_out.0.weight"
    out = linear(flat, weights[to_out_key])
    if not full_cover:
        video = slice(geom["video_start"], geom["video_end"])
        text = slice(geom["text_start"], geom["text_end"])
        text_x = hidden_states[text] if enable_text_state and geom["text_end"] > geom["text_start"] else None
        text_qkv = None
        if text_x is not None:
            text_qkv = (query_raw[text], key_raw[text], value_raw[text])
        readout = _linear_scan(
            weights,
            hidden_states[video],
            (query_raw[video], key_raw[video], value_raw[video]),
            geom,
            bounds,
            skip_ends=anchor_frames == "both",
            text_x=text_x,
            text_qkv=text_qkv,
            delta_rule=delta_rule,
        )
        far = linear(readout, weights["attn.to_out_linear.weight"])
        out = out.at[video].add(far)
    return out
