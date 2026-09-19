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
    from fastvideo.layers.quantization.base_config import (
        QuantizationConfig, )
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

# ------------------------------------------------------------------
# TrainingArgs builders (only place that creates FastVideoArgs)
# ------------------------------------------------------------------


def _make_training_args(
    tc: TrainingConfig,
    *,
    model_path: str,
) -> TrainingArgs:
    """Build a TrainingArgs for PipelineComponentLoader."""
    pipeline_config = tc.pipeline_config or PipelineConfig()
    # Propagate dit_precision from TrainingConfig to PipelineConfig
    # so that TransformerLoader.load() picks up the correct
    # default_dtype (e.g. fp32 master weights for training).
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
        # Modular stack opts into regional fullgraph compile; the legacy
        # stack keeps whole-model torch.compile semantics (default False).
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
    # Never CPU-offload the DiT here: validation samples with the LIVE
    # training transformer, and the denoising stages' post-sampling
    # ``transformer.to("cpu")`` strands the FSDP-sharded training params on
    # CPU — the next training backward then dies assigning CUDA grads to
    # CPU tensors (no-grad forwards keep working off gathered buffers,
    # which is why it surfaces steps later).
    args.dit_cpu_offload = False
    # Validation must sample at the training attention contract: both the
    # sparsity AND the tile geometry. Leaving the tile size at its
    # FastVideoArgs default silently validates a tile-64-trained student at
    # tile 256 (v8 shipped 2400 steps of validation that way).
    args.VSA_sparsity = tc.vsa_sparsity
    args.VSA_tile_size = tc.vsa_tile_size
    return args


# ------------------------------------------------------------------
# Module loading
# ------------------------------------------------------------------

# An explicit "stay dense" request. Roles state it instead of relying on
# the absence of a key, so a teacher/critic cannot silently inherit a
# quantization request meant for the student.
_DENSE_QUANT_REQUESTS = frozenset({"none", "dense", "full_precision", "no_quant"})


def _resolve_construction_quant_config(
    request: str | QuantizationConfig,
) -> QuantizationConfig | None:
    """Resolve one role's quant request into a config instance, or None.

    Accepts a registered method name (e.g. ``"nvfp4_qat_train"``), an
    already-built :class:`QuantizationConfig`, or an explicit dense alias.
    Model construction calls ``quant_config.get_quant_method()`` inside
    ``LinearBase.__init__``, so a bare YAML string would crash there.
    """
    if isinstance(request, str):
        if request.strip().lower() in _DENSE_QUANT_REQUESTS:
            return None
        from fastvideo.layers.quantization import (
            get_quantization_config, )
        return get_quantization_config(request)()
    return request


# An explicit "no LoRA" request, so a role states it instead of relying on the
# absence of a key (mirrors _DENSE_QUANT_REQUESTS).
_NO_LORA_REQUESTS = frozenset({"none", "null", "no_lora", "disabled", "false"})

# What an H3 LoRA role adapts when it does not name its own targets: attention,
# the feed-forward branch (the token refiner's included) and the time embedder,
# which is the set HyperFlow fine-tunes on this model. Every name is an
# index-free path suffix read off the live transformer -- the checkpoint's
# packaging spelling ``to_out.0`` matches nothing, because the live module is a
# flat ``to_out``. Targets match by substring, so a bare ``fc_in`` reaches
# ``ff.fc_in``, ``token_refiner.blocks.N.ff.fc_in`` and ``time_embedder.fc_in``
# alike.
DEFAULT_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "fc_in",
    "fc_out",
)

_LORA_CONFIG_KEYS = frozenset({"rank", "alpha", "target_modules", "dropout"})


def _resolve_construction_lora_config(
    request: "LoraConfig | dict[str, Any] | str | None",
) -> "LoraConfig | None":
    """Resolve one role's LoRA request into a LoraConfig, or None.

    Accepts the shorthand a role writes in YAML
    (``{rank, alpha, target_modules, dropout}``), an already-built
    :class:`~fastvideo.train.utils.lora.LoraConfig`, or one of the explicit
    no-LoRA aliases. Unlike quantization there is no construction-time scope to
    install -- the adapter wraps a layer that already exists -- so this resolves
    the request and nothing else. A rank is required: an adapter that silently
    fell back to the normal trainable path is the full fine-tune this exists to
    avoid.
    """
    if request is None:
        return None
    if isinstance(request, str):
        if request.strip().lower() in _NO_LORA_REQUESTS:
            return None
        raise ValueError("construction_lora_config takes a mapping such as "
                         "{rank: 256, alpha: 256} or one of the explicit no-LoRA "
                         f"aliases {sorted(_NO_LORA_REQUESTS)}, got {request!r}")
    from fastvideo.train.utils.lora import LoraConfig
    if isinstance(request, LoraConfig):
        return request
    if not isinstance(request, dict):
        raise TypeError("construction_lora_config must be a mapping, a LoraConfig, "
                        f"or a no-LoRA alias, got {type(request).__name__}")
    unsupported = sorted(set(request) - _LORA_CONFIG_KEYS)
    if unsupported:
        raise ValueError(f"construction_lora_config has unsupported keys {unsupported}; "
                         f"the accepted keys are {sorted(_LORA_CONFIG_KEYS)}")
    rank = request.get("rank")
    if rank is None:
        raise ValueError("construction_lora_config requires an explicit rank")
    if float(request.get("dropout") or 0.0):
        # The training LoRA wrapper has no dropout path. Dropping the value
        # silently would leave a run looking regularized when it is not.
        raise ValueError("construction_lora_config.dropout is not supported by the "
                         "training LoRA wrapper; leave it at 0.0")
    return LoraConfig(
        enable=True,
        rank=int(rank),
        alpha=request.get("alpha"),
        target_modules=list(request.get("target_modules") or DEFAULT_LORA_TARGETS),
    )


def _verify_role_lora(transformer: torch.nn.Module) -> None:
    """Fail loudly unless an adapter request landed as a LoRA-only change.

    Two ways this plumbing could look applied while being wrong: the targets
    matched no layer, so the role quietly trains the base weights -- the full
    fine-tune this exists to prevent; or the adapters went in before
    ``apply_trainable``, which re-enables every weight and leaves the adapter
    as decoration.
    """
    trainable = [name for name, param in transformer.named_parameters() if param.requires_grad]
    if not [name for name in trainable if "lora_" in name]:
        raise RuntimeError("LoRA was requested for this role but no adapter parameter is "
                           "trainable -- the target list matched no linear layer")
    leaked = [name for name in trainable if "lora_" not in name]
    if leaked:
        raise RuntimeError(f"LoRA was requested for this role but {len(leaked)} non-adapter "
                           f"parameters are still trainable (first: {leaked[:4]}); the base "
                           "weights would move and this would not be a LoRA run")


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
    construction_quant_config: str | QuantizationConfig | None = None,
    pre_fsdp_transform: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """Load one pipeline component with its role-scoped attention policy.

    Accepts a ``TrainingConfig`` and internally builds the
    ``TrainingArgs`` needed by ``PipelineComponentLoader``.

    Diffusers component entries retain provider and architecture as their
    first two fields and can append modular loading metadata. Attention layers
    bind their backend during construction, so the requested backend remains
    scoped to this load call.

    ``construction_quant_config`` is the per-role quantization scope.
    Quantized linears bind their method inside ``LinearBase.__init__``, so
    the request must be in place while THIS role's transformer is built and
    removed before the next role builds. ``None`` means "no role-local
    request" (the pipeline-level setting, if any, stands); a dense alias
    such as ``"none"`` explicitly clears quantization for this role.
    """
    fastvideo_args: Any = _make_training_args(training_config, model_path=model_path)
    original_dit_precision = fastvideo_args.pipeline_config.dit_precision
    if construction_precision is not None:
        # A frozen role does not need FP32 optimizer masters. Its FSDP forward
        # already casts parameters to BF16, so constructing/storing that role
        # in BF16 removes memory with no change to the actual teacher compute.
        fastvideo_args.pipeline_config.dit_precision = str(construction_precision)

    resolved_quant_config = (None if construction_quant_config is None else
                             _resolve_construction_quant_config(construction_quant_config))
    if disable_custom_init_weights and resolved_quant_config is not None:
        # component_loader nulls quant_config on the ``_loading_teacher_critic_model``
        # path, so a quant request here is guaranteed to be a silent no-op --
        # exactly the failure this plumbing exists to prevent. Fail loudly.
        raise ValueError(
            "load_module_from_path: disable_custom_init_weights marks this role as the "
            "DMD teacher/critic, which is deliberately constructed full precision and "
            "has its quant_config dropped by component_loader. Remove "
            "construction_quant_config for this role, or pass an explicit dense "
            "request (e.g. 'none').")
    original_dit_quant_config = fastvideo_args.pipeline_config.dit_config.quant_config
    if construction_quant_config is not None:
        fastvideo_args.pipeline_config.dit_config.quant_config = resolved_quant_config

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in "
                         f"config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in "
                         f"config at {local_model_path}")

    # Trailing modular-manifest metadata does not change component dispatch;
    # the provider and architecture remain the first two fields.
    transformers_or_diffusers, _architecture = module_info[:2]
    component_path = os.path.join(local_model_path, module_type)

    # fastvideo_args is freshly built above and never escapes this function,
    # so overrides are plain assignments — nothing to save or restore.
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
    # Per-role request delivered as a construction scope: process-local,
    # exception-safe, and part of the selector's cache key (no global
    # mutation, no cache flushes between roles).
    attention_context = (nullcontext() if resolved_attention_backend is None else _component_attention_backend_scope(
        resolved_attention_backend, component=module_type))

    if disable_custom_init_weights:
        fastvideo_args._loading_teacher_critic_model = True
    # Attention implementations are bound while transformer layers are
    # constructed. Scope the override to this one role so student,
    # teacher, and critic can use independent backends in one process.
    try:
        with attention_context:
            module = PipelineComponentLoader.load_module(
                module_name=module_type,
                component_model_path=component_path,
                transformers_or_diffusers=(transformers_or_diffusers),
                fastvideo_args=fastvideo_args,
            )
    finally:
        # _make_training_args intentionally shares the resolved pipeline
        # config. Do not leak a role-local construction choice to later roles.
        fastvideo_args.pipeline_config.dit_precision = original_dit_precision
        fastvideo_args.pipeline_config.dit_config.quant_config = original_dit_quant_config

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
