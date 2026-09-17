# SPDX-License-Identifier: Apache-2.0
"""Typed training config — replaces TrainingArgs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass(slots=True)
class DistributedConfig:
    num_gpus: int = 1
    tp_size: int = 1
    sp_size: int = 1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    pin_cpu_memory: bool = False


@dataclass(slots=True)
class DataConfig:
    data_path: str | list[str] | dict[str, int] = ""
    preprocessed_data_type: str = "t2v"
    train_batch_size: int = 1
    dataloader_num_workers: int = 0
    training_cfg_rate: float = 0.0
    seed: int = 0
    num_height: int = 0
    num_width: int = 0
    num_latent_t: int = 0
    num_frames: int = 0
    # Preserve each T2VA row's native temporal/spatial latent geometry and
    # schedule exact-shape global microbatches across data-parallel ranks.
    # False retains the legacy fixed-shape/truncation contract.
    native_shape_bucketing: bool = False


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 0
    lr_power: float = 0.0
    min_lr_ratio: float = 0.5


@dataclass(slots=True)
class TrainingLoopConfig:
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 1


@dataclass(slots=True)
class CheckpointConfig:
    output_dir: str = ""
    resume_from_checkpoint: str = ""
    # Deployable, model-only checkpoints are saved for every scheduled
    # validation event and are independent from rolling resumable state.
    save_inference_checkpoint_on_validation: bool = False
    inference_checkpoint_role: str = "student"
    inference_checkpoint_dtype: str = "bfloat16"
    training_state_checkpointing_steps: int = 0
    # Opt in to resumable checkpoints published only after DCP state and every
    # rank's RNG snapshot are complete. False keeps legacy checkpoints usable.
    require_complete_training_checkpoint: bool = False
    # Applies only to resumable ``checkpoint-<step>`` directories. Inference
    # checkpoints are retained as the run's immutable model lineage.
    checkpoints_total_limit: int = 0
    checkpointing_start_step: int = 0
    # DCP checkpoints restore optimizer param-group LRs and scheduler
    # base_lrs, silently overriding the YAML on resume. Set true to re-apply
    # the configured learning rates after loading (LR-change experiments).
    reset_lr_on_resume: bool = False


@dataclass(slots=True)
class TrackerConfig:
    trackers: list[str] = field(default_factory=list)
    entity: str = ""
    project_name: str = "fastvideo"
    run_name: str = ""


@dataclass(slots=True)
class ModelTrainingConfig:
    weighting_scheme: str = "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.0
    precondition_outputs: bool = False
    moba_config: dict = field(default_factory=dict)
    enable_gradient_checkpointing_type: str | None = None
    # Optimizer steps applied to bf16/fp16 parameter storage round away
    # updates smaller than ~half an ulp of each weight's magnitude —
    # O(1)-magnitude parameters (norm gains) freeze entirely at typical
    # distillation LRs. The trainer refuses to start unless master weights
    # are fp32 (training.dit_precision: fp32) or this explicit opt-in
    # acknowledges the effect (memory-constrained topologies).
    allow_low_precision_master_weights: bool = False
    enable_torch_compile: bool = False
    torch_compile_kwargs: dict = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    vsa_sparsity: float = 0.0
    # Tokens per sparse-attention tile for the VSA student. 256 (default)
    # keeps the (4,8,8) tiles and today's VSA-256 CuTe/Triton routing; 64
    # selects (4,4,4) tiles on the native 64-token Triton block-sparse
    # kernels (forward and backward). Consumed by the VSA-H3 (MiniMax H3)
    # backend; Wan's VSA path ignores it.
    vsa_tile_size: int = 256
    # Reuse the per-step padded VSA tile buffer across attention layers.
    # Defaults to False for training: under full activation checkpointing the
    # cached buffer survives into the backward recompute and inflates peak
    # memory (see #1423). Enable on memory-rich setups to keep the per-step
    # buffer-reuse speedup.
    vsa_cache_tile_buf: bool = False
    model: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    pipeline_config: PipelineConfig | None = None
    model_path: str = ""
    dit_precision: str = "fp32"
