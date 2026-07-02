# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for fastvideo.

This package contains diffusion pipelines for generating videos and images.
Heavy pipeline/registry imports are intentionally lazy so utility modules such
as ``fastvideo.pipelines.pipeline_batch_info`` can be imported by lightweight
tests without constructing the full inference stack.
"""

from __future__ import annotations

from typing import Any, cast

from fastvideo.logger import init_logger

logger = init_logger(__name__)
_PIPELINE_WITH_LORA_CLS = None


def _pipeline_with_lora_cls():
    global _PIPELINE_WITH_LORA_CLS
    if _PIPELINE_WITH_LORA_CLS is not None:
        return _PIPELINE_WITH_LORA_CLS

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
        """Type for a pipeline that has both ComposedPipelineBase and LoRAPipeline functionality."""
        pass

    _PIPELINE_WITH_LORA_CLS = PipelineWithLoRA
    return _PIPELINE_WITH_LORA_CLS


def build_pipeline(fastvideo_args: Any, pipeline_type: Any | str | None = None):
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class
    """
    from fastvideo.pipelines.pipeline_registry import PipelineType
    from fastvideo.registry import get_model_info
    from fastvideo.utils import maybe_download_model

    if pipeline_type is None:
        pipeline_type = PipelineType.BASIC

    # Get pipeline type
    model_path = fastvideo_args.model_path
    model_path = maybe_download_model(model_path)
    # fastvideo_args.downloaded_model_path = model_path
    logger.info("Model path: %s", model_path)

    logger.info("Building pipeline of type: %s",
                pipeline_type.value if isinstance(pipeline_type, PipelineType) else pipeline_type)

    model_info = get_model_info(
        model_path=model_path,
        pipeline_type=pipeline_type,
        workload_type=fastvideo_args.workload_type,
        override_pipeline_cls_name=fastvideo_args.override_pipeline_cls_name,
    )
    pipeline_cls = model_info.pipeline_cls

    # instantiate the pipelines
    pipeline = pipeline_cls(model_path, fastvideo_args)

    logger.info("Pipelines instantiated")

    return cast(_pipeline_with_lora_cls(), pipeline)


def __getattr__(name: str):
    if name == "ComposedPipelineBase":
        from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

        return ComposedPipelineBase
    if name == "ForwardBatch":
        from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

        return ForwardBatch
    if name == "LoRAPipeline":
        from fastvideo.pipelines.lora_pipeline import LoRAPipeline

        return LoRAPipeline
    if name == "PipelineWithLoRA":
        return _pipeline_with_lora_cls()
    if name == "TrainingBatch":
        from fastvideo.pipelines.pipeline_batch_info import TrainingBatch

        return TrainingBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_pipeline",
    "ComposedPipelineBase",
    "ForwardBatch",
    "LoRAPipeline",
    "PipelineWithLoRA",
    "TrainingBatch",
]
