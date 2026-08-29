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
            r"^(.*)\.attn\.to_out\.0\.(.*)$": r"\1.attn.to_out.\2",
            r"^(.*)\.ff\.net\.0\.proj\.(.*)$": r"\1.ff.fc_in.\2",
            r"^(.*)\.ff\.net\.2\.(.*)$": r"\1.ff.fc_out.\2",
        })
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    # A pruned student stores local blocks densely as ``0..num_layers-1``.
    # ``block_map[local_index]`` records which source block supplied that
    # local block.  Both fields are checkpoint metadata: they do not change
    # weight names or skip blocks during a forward pass.
    source_num_layers: int | None = None
    block_map: tuple[int, ...] | list[int] | None = None
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_layers <= 0:
            raise ValueError(f"MiniMax H3 num_layers must be positive, got {self.num_layers}.")
        if self.source_num_layers is not None and self.source_num_layers <= 0:
            raise ValueError(f"MiniMax H3 source_num_layers must be positive when set, got {self.source_num_layers}.")
        if self.block_map is None:
            if self.source_num_layers not in (None, self.num_layers):
                raise ValueError("A pruned MiniMax H3 config must define block_map when "
                                 f"source_num_layers={self.source_num_layers} and num_layers={self.num_layers}.")
        else:
            if self.source_num_layers is None:
                raise ValueError("MiniMax H3 block_map requires source_num_layers.")
            if any(not isinstance(index, int) or isinstance(index, bool) for index in self.block_map):
                raise ValueError(f"MiniMax H3 block_map must contain only integer indices, got {self.block_map}.")
            block_map = tuple(self.block_map)
            if len(block_map) != self.num_layers:
                raise ValueError("MiniMax H3 block_map length must equal num_layers; "
                                 f"got {len(block_map)} and {self.num_layers}.")
            if any(left >= right for left, right in zip(block_map, block_map[1:], strict=False)):
                raise ValueError(f"MiniMax H3 block_map must be strictly increasing, got {block_map}.")
            if block_map[0] < 0 or block_map[-1] >= self.source_num_layers:
                raise ValueError("MiniMax H3 block_map indices must be in "
                                 f"[0, {self.source_num_layers}), got {block_map}.")
            self.block_map = block_map
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

    def source_block_indices(self) -> tuple[int, ...]:
        """Return the source provenance for every densely stored local block."""
        if self.block_map is not None:
            return tuple(self.block_map)
        return tuple(range(self.num_layers))


@dataclass
class MiniMaxH3Config(DiTConfig):
    """FastVideo component configuration for MiniMax H3 transformers."""

    arch_config: MiniMaxH3ArchConfig = field(default_factory=MiniMaxH3ArchConfig)
    prefix: str = "minimax_h3"
    # FastVideo's Fully Sharded Data Parallel (FSDP) loading path requires one
    # parameter dtype, while H3 inference keeps boundary projections in FP32.
    uniform_parameter_dtype: bool = False
