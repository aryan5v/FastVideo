# SPDX-License-Identifier: Apache-2.0
"""
Data structures for functional pipeline processing.

This module defines the dataclasses used to pass state between pipeline components
in a functional manner, reducing the need for explicit parameter passing.
"""

import pprint
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import PIL.Image
import torch

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder

    from fastvideo.api.schema import ContinuationState

import time
from collections import OrderedDict

from fastvideo.attention import AttentionMetadata


class PipelineLoggingInfo:
    """Simple approach using OrderedDict to track stage metrics."""

    def __init__(self):
        self.stages: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add_stage_execution_time(self, stage_name: str, execution_time: float):
        """Add execution time for a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name]['execution_time'] = execution_time
        self.stages[stage_name]['timestamp'] = time.time()

    def add_stage_metric(self, stage_name: str, metric_name: str, value: Any):
        """Add any metric for a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name][metric_name] = value

    def get_stage_info(self, stage_name: str) -> dict[str, Any]:
        """Get all info for a specific stage."""
        return self.stages.get(stage_name, {})

    def get_execution_order(self) -> list[str]:
        """Get stages in execution order."""
        return list(self.stages.keys())

    def get_total_execution_time(self) -> float:
        """Get total pipeline execution time."""
        return sum(stage.get('execution_time', 0) for stage in self.stages.values())


@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.
    
    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """
    data_type: str

    generator: torch.Generator | list[torch.Generator] | None = None

    image_path: str | None = None
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    pil_image: torch.Tensor | PIL.Image.Image | None = None
    last_image: torch.Tensor | PIL.Image.Image | None = None
    references: list[Any] | None = None
    preprocessed_image: torch.Tensor | None = None
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_video_name: str | None = None

    video_path: str | None = None
    video_latent: torch.Tensor | None = None

    refine_from: str | None = None
    t_thresh: float = 0.5
    spatial_refine_only: bool = False
    num_cond_frames: int = 0
    stage1_video: list[PIL.Image.Image] | None = None  # Loaded frames from refine_from

    prompt_embeds: list[torch.Tensor] = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None
    clip_embedding_pos: list[torch.Tensor] | None = None
    clip_embedding_neg: list[torch.Tensor] | None = None

    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False
    use_embedded_guidance: bool = False
    true_cfg_scale: float = 1.0

    batch_size: int | None = None
    num_videos_per_prompt: int = 1
    seed: int | None = None
    seeds: list[int] | None = None

    is_prompt_processed: bool = False

    latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None
    lq_latents: torch.Tensor | None = None
    raw_latent_shape: tuple[int, ...] | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None
    first_frame_latent: torch.Tensor | None = None

    mouse_cond: torch.Tensor | None = None  # Shape: (B, T, 2)
    keyboard_cond: torch.Tensor | None = None  # Shape: (B, T, K)
    grid_sizes: torch.Tensor | None = None  # Shape: (3,) [F,H,W]
    num_iterations: int | None = None
    use_base_model: bool = False

    pose: str | None = None  # Camera trajectory: pose string (e.g., 'w-31') or JSON file path

    camera_states: torch.Tensor | None = None  # Plücker coordinates [B, T, 6, H, W]
    gt_latents: torch.Tensor | None = None  # Ground truth latents for conditioning [B, 16, T, H, W]
    conditioning_mask: torch.Tensor | None = None  # Mask for conditioning [B, 1, T, H, W]
    camera_trajectory: str | None = None  # Camera trajectory file/identifier
    action_list: list[str] | None = None  # List of actions (e.g., ['forward', 'left'])
    action_speed_list: list[float] | None = None  # Speed for each action
    c2ws_plucker_emb: torch.Tensor | None = None  # Plucker embedding: [B, C, F_lat, H_lat, W_lat]
    action_path: str | None = None  # Directory containing poses.npy and intrinsics.npy

    trajectory_type: str | None = None
    movement_distance: float | None = None
    camera_rotation: str | None = None

    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: list[int] | int = 1  # Default for image models

    height: list[int] | int | None = None
    width: list[int] | int | None = None
    height_sr: list[int] | int | None = None
    width_sr: list[int] | int | None = None
    fps: list[int] | int | None = None

    timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None
    boundary_ratio: float | None = None

    num_inference_steps: int = 50
    num_inference_steps_sr: int = 50
    guidance_scale: float = 1.0
    batch_cfg: bool = False
    guidance_scale_2: float | None = None
    cfg_normalization: bool = False
    cfg_truncation: float | None = 1.0
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    enable_teacache: bool = False

    ltx2_cfg_scale_video: float = 1.0
    ltx2_cfg_scale_audio: float = 1.0
    ltx2_modality_scale_video: float = 1.0
    ltx2_modality_scale_audio: float = 1.0
    ltx2_rescale_scale: float = 0.0
    ltx2_stg_scale_video: float = 0.0
    ltx2_stg_scale_audio: float = 0.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=list)
    ltx2_stg_blocks_audio: list[int] = field(default_factory=list)

    ltx2_images: list[tuple[str, int, float]] | None = None
    ltx2_image_crf: float = 33.0
    ltx2_conditioning_latent_stage1: torch.Tensor | None = None
    ltx2_conditioning_latent_stage2: torch.Tensor | None = None
    ltx2_video_conditions: list[tuple[list[str], int, float]] | None = None

    audio_start_in_s: float | None = None
    audio_end_in_s: float | None = None

    init_audio: Any = None
    init_audio_strength: float | None = None
    init_noise_level: float | None = None
    inpaint_audio: Any = None
    inpaint_mask: Any = None

    n_tokens: int | None = None

    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    modules: dict[str, Any] = field(default_factory=dict)

    output: torch.Tensor | None = None
    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None

    continuation_state: "ContinuationState | None" = None
    return_continuation_state: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    save_video: bool = True
    return_frames: bool = False

    is_cfg_negative: bool = False

    VSA_sparsity: float = 0.0

    logging_info: PipelineLoggingInfo = field(default_factory=PipelineLoggingInfo)

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""

        ltx2_text_cfg_enabled = (self.ltx2_cfg_scale_video != 1.0 or self.ltx2_cfg_scale_audio != 1.0)
        if self.use_embedded_guidance:
            self.do_classifier_free_guidance = (self.true_cfg_scale > 1.0) or ltx2_text_cfg_enabled
        elif self.guidance_scale > 1.0 or ltx2_text_cfg_enabled:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)


@dataclass
class TrainingBatch:
    current_timestep: int = 0
    current_vsa_sparsity: float = 0.0

    latents: torch.Tensor | None = None
    raw_latent_shape: tuple[int, ...] | None = None
    noise_latents: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    encoder_attention_mask: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None
    audio_noisy_model_input: torch.Tensor | None = None
    audio_timesteps: torch.Tensor | None = None
    audio_sigmas: torch.Tensor | None = None
    audio_noise: torch.Tensor | None = None
    audio_encoder_hidden_states: torch.Tensor | None = None
    audio_encoder_attention_mask: torch.Tensor | None = None
    conditioning_mask: torch.Tensor | None = None
    preprocessed_image: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None
    image_latents: torch.Tensor | None = None
    infos: list[dict[str, Any]] | None = None
    mask_lat_size: torch.Tensor | None = None

    trajectory_latents: torch.Tensor | None = None
    trajectory_timesteps: torch.Tensor | None = None

    noisy_model_input: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    noise: torch.Tensor | None = None

    minimax_h3_layout: Any | None = None
    minimax_h3_dmd_layout: Any | None = None

    attn_metadata_vsa: AttentionMetadata | None = None
    attn_metadata: AttentionMetadata | None = None

    input_kwargs: dict[str, Any] | None = None

    loss: torch.Tensor | None = None

    total_loss: float | None = None
    grad_norm: float | None = None

    encoder_hidden_states_neg: torch.Tensor | None = None
    encoder_attention_mask_neg: torch.Tensor | None = None
    conditional_dict: dict[str, Any] | None = None
    unconditional_dict: dict[str, Any] | None = None

    generator_loss: float = 0.0
    fake_score_loss: float = 0.0

    dmd_latent_vis_dict: dict[str, Any] = field(default_factory=dict)
    latent_vis_dict: dict[str, Any] = field(default_factory=dict)
    fake_score_latent_vis_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessBatch(ForwardBatch):
    video_loader: list["VideoDecoder"] | list[str] = field(default_factory=list)
    video_file_name: list[str] = field(default_factory=list)
