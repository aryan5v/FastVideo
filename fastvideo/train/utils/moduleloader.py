# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

import torch

from fastvideo.attention.selector import (
    NO_REQUEST,
    _component_attention_backend_scope,
    coerce_attn_backend,
    component_attention_backend,
)
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import ExecutionMode, TrainingArgs
from fastvideo.models.loader.component_loader import (
    PipelineComponentLoader, )
from fastvideo.utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from fastvideo.platforms import AttentionBackendEnum

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )



def _make_training_args(
    tc: TrainingConfig,
    *,
    model_path: str,
) -> TrainingArgs:
    """Build a TrainingArgs for PipelineComponentLoader."""
    pipeline_config = tc.pipeline_config or PipelineConfig()
    if tc.dit_precision and tc.dit_precision != pipeline_config.dit_precision:
        pipeline_config.dit_precision = tc.dit_precision
    return TrainingArgs(
        model_path=model_path,
        mode=ExecutionMode.DISTILLATION,
        inference_mode=False,
        pipeline_config=pipeline_config,
        num_gpus=tc.distributed.num_gpus,
        tp_size=tc.distributed.tp_size,
        sp_size=tc.distributed.sp_size,
        hsdp_replicate_dim=tc.distributed.hsdp_replicate_dim,
        hsdp_shard_dim=tc.distributed.hsdp_shard_dim,
        pin_cpu_memory=tc.distributed.pin_cpu_memory,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        use_fsdp_inference=False,
        enable_torch_compile=tc.model.enable_torch_compile,
        regional_compile=True,
        torch_compile_kwargs=tc.model.torch_compile_kwargs,
    )


def make_inference_args(
    tc: TrainingConfig,
    *,
    model_path: str,
) -> TrainingArgs:
    """Build a TrainingArgs for inference (validation / pipelines)."""
    args = _make_training_args(tc, model_path=model_path)
    args.inference_mode = True
    args.mode = ExecutionMode.INFERENCE
    args.dit_cpu_offload = False
    args.VSA_sparsity = tc.vsa_sparsity
    args.VSA_tile_size = tc.vsa_tile_size
    return args




def load_module_from_path(
    *,
    model_path: str,
    module_type: str,
    training_config: TrainingConfig,
    disable_custom_init_weights: bool = False,
    override_transformer_cls_name: str | None = None,
    transformer_override_safetensor: str | None = None,
    attention_backend: AttentionBackendEnum | str | None = None,
    construction_precision: str | None = None,
    pre_fsdp_transform: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """Load one pipeline component with its role-scoped attention policy.

    Accepts a ``TrainingConfig`` and internally builds the
    ``TrainingArgs`` needed by ``PipelineComponentLoader``.

    Diffusers component entries retain provider and architecture as their
    first two fields and can append modular loading metadata. Attention layers
    bind their backend during construction, so the requested backend remains
    scoped to this load call.
    """
    fastvideo_args: Any = _make_training_args(training_config, model_path=model_path)
    original_dit_precision = fastvideo_args.pipeline_config.dit_precision
    if construction_precision is not None:
        fastvideo_args.pipeline_config.dit_precision = str(construction_precision)

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in "
                         f"config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in "
                         f"config at {local_model_path}")

    transformers_or_diffusers, _architecture = module_info[:2]
    component_path = os.path.join(local_model_path, module_type)

    if override_transformer_cls_name is not None:
        fastvideo_args.override_transformer_cls_name = str(override_transformer_cls_name)

    if transformer_override_safetensor:
        fastvideo_args.init_weights_from_safetensors = str(transformer_override_safetensor)

    if pre_fsdp_transform is not None:
        if module_type != "transformer":
            raise ValueError("pre_fsdp_transform can only be set when loading "
                             f"a transformer, got module_type={module_type!r}")
        fastvideo_args._pre_fsdp_transform = pre_fsdp_transform

    if attention_backend is not None and module_type != "transformer":
        raise ValueError("attention_backend can only be set when loading "
                         f"a transformer, got module_type={module_type!r}")
    resolved_attention_backend = coerce_attn_backend(attention_backend)
    attention_context = (nullcontext() if resolved_attention_backend is None else _component_attention_backend_scope(
        resolved_attention_backend, component=module_type))

    if disable_custom_init_weights:
        fastvideo_args._loading_teacher_critic_model = True
    try:
        with attention_context:
            module = PipelineComponentLoader.load_module(
                module_name=module_type,
                component_model_path=component_path,
                transformers_or_diffusers=(transformers_or_diffusers),
                fastvideo_args=fastvideo_args,
            )
    finally:
        fastvideo_args.pipeline_config.dit_precision = original_dit_precision

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a "
                        f"torch.nn.Module: {type(module)}")
    if resolved_attention_backend is not None:
        receipt = component_attention_backend(module)
        if receipt is NO_REQUEST:
            raise RuntimeError(f"Loaded {module_type!r} from {model_path!r} did not record its "
                               f"requested attention backend {resolved_attention_backend.name}. "
                               "The component loader must stamp the construction decision on "
                               "module.config._resolved_attention_backend.")
        if receipt is not resolved_attention_backend:
            raise RuntimeError(f"Loaded {module_type!r} from {model_path!r} requested attention "
                               f"backend {resolved_attention_backend.name}, but recorded "
                               f"{receipt.name}.")
    return module
