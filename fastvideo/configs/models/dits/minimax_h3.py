# SPDX-License-Identifier: Apache-2.0
"""Architecture configuration for the MiniMax H3 joint audio-video DiT."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_minimax_h3_block(name: str, module: object) -> bool:
    """Select the main and text-refiner transformer blocks for FSDP."""
    del module
    parts = name.split(".")
    return ((len(parts) == 2 and parts[0] == "transformer_blocks" and parts[1].isdigit())
            or (len(parts) == 3 and parts[:2] == ["token_refiner", "refiner_blocks"] and parts[2].isdigit()))


@dataclass
class MiniMaxH3ArchConfig(DiTArchConfig):
    """One-to-one representation of the released transformer config."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_minimax_h3_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
        # FP4-quantized QK attention (fa4_fp4 on sm_100/sm_103, cutlass on
        # sm_12x). Enabled for speed experiments; output quality against the
        # SSIM references is not yet validated.
        AttentionBackendEnum.ATTN_QAT_INFER,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^time_embedder\.linear_1\.(.*)$": r"time_embedder.fc_in.\1",
            r"^time_embedder\.linear_2\.(.*)$": r"time_embedder.fc_out.\1",
            # Hybrid checkpoints wrap dense attention under ``attn.orig``. First
            # match wins, so ``orig.to_out.0`` must land on ``to_out`` in one
            # substitution — a generic ``orig`` rewrite would stop at
            # ``to_out.0`` and miss the FastVideo linear name.
            r"^(.*)\.attn\.orig\.to_out\.0\.(.*)$": r"\1.attn.to_out.\2",
            r"^(.*)\.attn\.orig\.(.*)$": r"\1.attn.\2",
            r"^(.*)\.attn\.to_out\.0\.(.*)$": r"\1.attn.to_out.\2",
            r"^(.*)\.ff\.net\.0\.proj\.(.*)$": r"\1.ff.fc_in.\2",
            r"^(.*)\.ff\.net\.2\.(.*)$": r"\1.ff.fc_out.\2",
        })
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_rank: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # Window softmax + linear far-branch. Off by default so dense / VSA
    # checkpoints keep their existing attention body. A converted hybrid
    # checkpoint stamps these fields in transformer/config.json.
    hybrid_attention: bool = False
    hybrid_window_radius: int = 1
    hybrid_window_chunk: int = 5
    hybrid_anchor_frames: str = "both"
    hybrid_delta_rule: str = "vdn_solve"
    hybrid_enable_softmax_gate: bool = True
    hybrid_enable_text_state: bool = True
    hybrid_short_conv_targets: tuple[str, ...] = ("k", "v")
    # Two sequence-parallel ranks split softmax vs linear (all-reduce). Off
    # automatically when SP != 2; True here so a dual-GPU/Spark job opts in
    # without a second flag.
    hybrid_branch_parallel: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.patch_size) != 3:
            raise ValueError(f"MiniMax H3 patch_size must have three axes, got {self.patch_size}.")
        self.patch_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        self.num_channels_latents = self.in_channels
        self.out_channels = self.in_channels
        if self.adaln_rank is not None and not 0 < self.adaln_rank <= self.time_embed_dim:
            raise ValueError(f"MiniMax H3 adaln_rank must be in (0, time_embed_dim={self.time_embed_dim}], "
                             f"got {self.adaln_rank}.")
        rotary_dim = 2 * 3 * self.rope_freq_dim
        if rotary_dim > self.attention_head_dim or rotary_dim % 2:
            raise ValueError(f"MiniMax H3 rotary width must be even and no larger than the head width; got "
                             f"rotary_dim={rotary_dim}, attention_head_dim={self.attention_head_dim}.")
        if self.hybrid_anchor_frames not in ("none", "columns", "rows", "both"):
            raise ValueError(f"hybrid_anchor_frames must be none/columns/rows/both, got {self.hybrid_anchor_frames!r}.")
        if self.hybrid_delta_rule not in ("sana_scaled", "vdn_solve", "vdn_scaled"):
            raise ValueError("hybrid_delta_rule must be sana_scaled/vdn_solve/vdn_scaled, "
                             f"got {self.hybrid_delta_rule!r}.")
        if self.hybrid_window_radius < 0 or self.hybrid_window_chunk < 0:
            raise ValueError("hybrid window radius and chunk must be >= 0.")
        self.hybrid_short_conv_targets = tuple(self.hybrid_short_conv_targets)
        allowed_conv = {"q", "k", "v"}
        if any(target not in allowed_conv for target in self.hybrid_short_conv_targets):
            raise ValueError(f"hybrid_short_conv_targets must be a subset of {sorted(allowed_conv)}, "
                             f"got {self.hybrid_short_conv_targets!r}.")


@dataclass
class MiniMaxH3Config(DiTConfig):
    """FastVideo component configuration for MiniMax H3 transformers."""

    arch_config: MiniMaxH3ArchConfig = field(default_factory=MiniMaxH3ArchConfig)
    prefix: str = "minimax_h3"
    # FastVideo's Fully Sharded Data Parallel (FSDP) loading path requires one
    # parameter dtype, while H3 inference keeps boundary projections in FP32.
    uniform_parameter_dtype: bool = False
