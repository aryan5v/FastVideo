# SPDX-License-Identifier: Apache-2.0
"""Frame-wise linear attention branch for MiniMax H3 hybrid inference.

Summarises everything the softmax window does not see with a bidirectional
delta-rule scan over latent frames. QKV is shared with the softmax branch
(raw, pre-QK-norm, pre-RoPE). This is a FastVideo-native module: linears use
``ReplicatedLinear``, packing comes from ``HybridSequenceLayout``, and the
scan is the existing batched-GEMM recurrence — not a vendored copy of an
external runtime.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.models.dits.minimax_h3_hybrid.layout import HybridSequenceLayout

DELTA_RULES = ("sana_scaled", "vdn_solve", "vdn_scaled")
SHORT_CONV_TARGETS = ("q", "k", "v")
TEXT_STATE_SCALE = 0.5


class OutputGate(nn.Module):
    """Sigmoid gate on a branch output.

    Softmax (``head_dim is None``): one value per (token, head), constant-init
    so the window branch starts near the dense teacher. Linear (``head_dim``
    set): per-channel low-rank gate with live default init.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bottleneck: int | None = None,
        init_value: float = 0.99,
        init: str = "constant",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        out_features = num_heads * (head_dim or 1)
        self.down = None if bottleneck is None else ReplicatedLinear(
            hidden_size,
            bottleneck,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down",
        )
        self.up = ReplicatedLinear(
            bottleneck or hidden_size,
            out_features,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.up",
        )
        if init == "constant":
            nn.init.zeros_(self.up.weight)
            nn.init.constant_(self.up.bias, math.log(init_value / (1.0 - init_value)))
        elif init != "random":
            raise ValueError(f"OutputGate init must be 'constant' or 'random', got {init!r}.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = hidden_states if self.down is None else self.down(hidden_states)[0]
        gate, _ = self.up(projected)
        return torch.sigmoid(gate).view(-1, self.num_heads, self.head_dim or 1)


class FrameKDAAlpha(nn.Module):
    """Per-frame, per-head, per-channel delta-rule gate ``alpha``."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.down = ReplicatedLinear(hidden_size, head_dim, bias=False, prefix=f"{prefix}.down")
        self.up = ReplicatedLinear(head_dim, num_heads * head_dim, bias=False, prefix=f"{prefix}.up")
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(torch.rand(num_heads * head_dim, dtype=torch.float32) *
                       (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=frame_mean.device.type, enabled=False):
            delta = F.linear(frame_mean.float(), self.down.weight.float())
            delta = F.linear(delta, self.up.weight.float())
            delta = delta + self.dt_bias.float()
            scale = torch.exp(self.A_log.float())[:, None]
            delta = delta.view(-1, self.num_heads, self.head_dim)
            return torch.exp(-scale * F.softplus(delta))


class LinearAttentionSepConv(nn.Module):
    """Separable 5x5 spatial + 5-tap temporal depthwise conv on named projections."""

    KERNEL = 5

    def __init__(self, channels: int, targets: tuple[str, ...] = ("k", "v")) -> None:
        super().__init__()
        self.targets = tuple(targets)
        kernel = self.KERNEL
        for name in self.targets:
            spatial = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
            nn.init.normal_(spatial.weight, std=(kernel * kernel)**-0.5)
            temporal = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
            nn.init.normal_(temporal.weight, std=kernel**-0.5)
            setattr(self, f"{name}_sp", spatial)
            setattr(self, f"{name}_tm", temporal)

    def apply(self, proj: str, tokens: torch.Tensor, num_frames: int, frame_size: tuple[int, int]) -> torch.Tensor:
        if proj not in self.targets:
            return tokens
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, getattr(self, f"{proj}_sp").weight, padding=self.KERNEL // 2, groups=channels)
        frames = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        weight_t = getattr(self, f"{proj}_tm").weight.squeeze(1).to(frames.dtype)
        pad = self.KERNEL // 2
        padded = F.pad(frames, (0, 0, 0, 0, pad, pad))
        out = None
        for tap in range(self.KERNEL):
            part = padded[tap:tap + num_frames] * weight_t[:, tap].view(1, 1, -1)
            out = part if out is None else out + part
        assert out is not None
        return out.reshape(-1, heads, head_dim)


def _activate_features(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    activated = F.silu(tokens)
    if not l2norm:
        return activated
    return F.normalize(activated, dim=-1, eps=1e-6).to(activated.dtype)


def frame_statistics(
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    a_fp32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ``A = k^T beta k``, ``B = v^T beta k``. keys/values: [F, H, S, d]."""
    keys = keys.contiguous()
    values = values.contiguous()
    weighted = keys * beta.unsqueeze(-1)
    if a_fp32:
        with torch.autocast(device_type=keys.device.type, enabled=False):
            matrix_a = torch.matmul(weighted.float().transpose(-1, -2), keys.float())
            matrix_b = torch.matmul(values.float().transpose(-1, -2), weighted.float())
        return matrix_a, matrix_b
    matrix_a = torch.matmul(weighted.transpose(-1, -2), keys)
    matrix_b = torch.matmul(values.transpose(-1, -2), weighted)
    return matrix_a, matrix_b


def _factor_sana(alpha: torch.Tensor, matrix_a: torch.Tensor, matrix_b: torch.Tensor,
                 tokens_per_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_tokens = 1.0 / tokens_per_frame
    eye = torch.eye(matrix_a.shape[-1], device=matrix_a.device, dtype=matrix_a.dtype)
    transition = alpha.unsqueeze(-1) * (eye - inv_tokens * matrix_a)
    injection = (inv_tokens**0.5) * matrix_b
    return transition, injection


def _factor_vdn(alpha: torch.Tensor, matrix_a: torch.Tensor, matrix_b: torch.Tensor, scaled: bool,
                tokens_per_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (1.0 / tokens_per_frame) if scaled else 1.0
    sqrt_scale = scale**0.5 if scaled else 1.0
    matrix_a32 = matrix_a.float() * scale
    eye = torch.eye(matrix_a32.shape[-1], device=matrix_a32.device, dtype=torch.float32).expand_as(matrix_a32)
    chol = torch.linalg.cholesky(matrix_a32 + eye)
    inverse = torch.cholesky_solve(eye.contiguous(), chol)
    transition = alpha.unsqueeze(-1) * inverse
    injection = (matrix_b.float() * sqrt_scale) @ inverse
    return transition.to(matrix_a.dtype), injection.to(matrix_b.dtype)


def factor_delta(
    rule: str,
    alpha: torch.Tensor,
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule not in DELTA_RULES:
        raise ValueError(f"unknown delta_rule {rule!r}; expected one of {DELTA_RULES}.")
    if rule == "sana_scaled":
        return _factor_sana(alpha, matrix_a, matrix_b, tokens_per_frame)
    return _factor_vdn(alpha, matrix_a, matrix_b, scaled=rule == "vdn_scaled", tokens_per_frame=tokens_per_frame)


def run_scans(transitions: torch.Tensor, injections: torch.Tensor,
              text_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional frame scans. transitions/injections: [F, H, d, d] / [F, H, d, d]."""
    start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
    prefix = torch.empty((transitions.shape[0], *start.shape), dtype=injections.dtype, device=injections.device)
    suffix = torch.empty_like(prefix)
    state = start
    for frame in range(transitions.shape[0]):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        state = prefix[frame]
    state = start
    for frame in range(transitions.shape[0] - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def gather_linear_state(
    prefix_states: torch.Tensor,
    suffix_states: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    text_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """State of frames outside each query window, decayed into that frame."""
    num_frames = prefix_states.shape[0]
    device = prefix_states.device
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    before_idx = last_before.clamp(min=0)
    after_idx = first_after.clamp(max=num_frames - 1)
    has_before = last_before >= 0
    has_after = first_after < num_frames
    state_before = prefix_states[before_idx]
    state_after = suffix_states[after_idx]
    if text_state is not None:
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, text_state)
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
    frames = torch.arange(num_frames, device=device)
    bridge_before = (last_before + 1).clamp(min=0)
    bridge_after = first_after.clamp(max=num_frames)
    alpha_from_before = torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before])
    alpha_from_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
    state_before = state_before * alpha_from_before.unsqueeze(2)
    state_after = state_after * alpha_from_after.unsqueeze(2)
    if text_state is not None:
        return state_before + state_after
    return state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(-1, 1, 1, 1)


class BidirectionalLinearBranch(nn.Module):
    """NoPE linear-attention branch over generated video frames."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        delta_rule: str = "vdn_solve",
        short_conv_targets: tuple[str, ...] = ("k", "v"),
        enable_text_state: bool = True,
        a_fp32: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if delta_rule not in DELTA_RULES:
            raise ValueError(f"delta_rule={delta_rule!r}; expected one of {DELTA_RULES}.")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.delta_rule = delta_rule
        self.enable_text_state = enable_text_state
        self.a_fp32 = a_fp32
        self.alpha = FrameKDAAlpha(hidden_size, num_heads, head_dim, prefix=f"{prefix}.alpha")
        self.beta_proj = ReplicatedLinear(hidden_size, num_heads, bias=False, prefix=f"{prefix}.beta_proj")
        self.output_gate = OutputGate(
            hidden_size,
            num_heads,
            head_dim,
            bottleneck=head_dim,
            init="random",
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.short_conv = (LinearAttentionSepConv(num_heads * head_dim, short_conv_targets)
                           if short_conv_targets else None)

    def _features(
        self,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: HybridSequenceLayout,
        use_conv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outs = []
        for proj, tokens in zip(("q", "k", "v"), qkv_raw, strict=True):
            if use_conv and self.short_conv is not None:
                tokens = self.short_conv.apply(proj, tokens, layout.num_frames,
                                               (layout.frame_height, layout.frame_width))
            outs.append(_activate_features(tokens, l2norm=proj != "v"))
        return outs[0], outs[1], outs[2]

    def _text_state(
        self,
        text_x: torch.Tensor,
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        length = text_qkv[1].shape[0]
        key = _activate_features(text_qkv[1], l2norm=True)
        value = _activate_features(text_qkv[2], l2norm=False)
        key = key.view(1, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x)[0]).view(1, length, self.num_heads).permute(0, 2, 1)
        matrix_a, matrix_b = frame_statistics(key, value, beta, a_fp32=self.a_fp32)
        ones = torch.ones(1, self.num_heads, self.head_dim, device=matrix_a.device, dtype=matrix_a.dtype)
        _, injection = factor_delta(self.delta_rule, ones, matrix_a, matrix_b, length)
        return TEXT_STATE_SCALE * injection[0]

    def forward(
        self,
        video_hidden: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: HybridSequenceLayout,
        bounds: list[tuple[int, int]],
        skip_ends: bool = False,
        text_hidden: torch.Tensor | None = None,
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """video_hidden: [F*S, hidden] -> gated, RMSNormed readout [F*S, H*d]."""
        query, key, value = self._features(qkv_raw, layout, use_conv=True)
        scan_frames = layout.num_frames
        if skip_ends and layout.num_frames > 2:
            per = layout.tokens_per_frame
            video_hidden = video_hidden[per:-per]
            query, key, value = query[per:-per], key[per:-per], value[per:-per]
            scan_frames = layout.num_frames - 2
            bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]

        per_frame = layout.tokens_per_frame
        query = query.view(scan_frames, per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(scan_frames, per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(scan_frames, per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(video_hidden)[0])
        beta = beta.view(scan_frames, per_frame, self.num_heads).permute(0, 2, 1)
        frame_mean = video_hidden.view(scan_frames, per_frame, -1).mean(dim=1)
        alpha = self.alpha(frame_mean)
        matrix_a, matrix_b = frame_statistics(key, value, beta, a_fp32=self.a_fp32)
        transitions, injections = factor_delta(self.delta_rule, alpha, matrix_a, matrix_b, per_frame)
        text_state = None
        if self.enable_text_state and text_hidden is not None and text_qkv is not None:
            text_state = self._text_state(text_hidden, text_qkv)
        prefix, suffix = run_scans(transitions, injections, text_state)
        state = gather_linear_state(prefix, suffix, alpha, bounds, text_state=text_state)
        # readout: q @ state^T -> [F, H, S, d]
        readout = torch.matmul(query.float(), state.transpose(-1, -2)).to(query.dtype)
        readout = self.norm(readout)
        gate = self.output_gate(video_hidden).view(scan_frames, per_frame, self.num_heads,
                                                   self.head_dim).permute(0, 2, 1, 3)
        gated = (readout * gate).permute(0, 2, 1, 3).reshape(-1, self.num_heads * self.head_dim)
        if skip_ends and layout.num_frames > 2:
            full = gated.new_zeros(layout.num_frames * per_frame, self.num_heads * self.head_dim)
            full[per_frame:-per_frame] = gated
            return full
        return gated
