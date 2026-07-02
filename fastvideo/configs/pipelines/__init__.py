# SPDX-License-Identifier: Apache-2.0
"""Pipeline config exports.

The concrete pipeline config modules register model-family side effects on
import. Keep this package lazy so importing one config (or the top-level
``fastvideo`` package) does not eagerly import every model family.
"""

_EXPORTS = {
    "CosmosConfig": ("fastvideo.configs.pipelines.cosmos", "CosmosConfig"),
    "Cosmos25Config": ("fastvideo.configs.pipelines.cosmos2_5", "Cosmos25Config"),
    "FastHunyuanConfig": ("fastvideo.configs.pipelines.hunyuan", "FastHunyuanConfig"),
    "Hunyuan15T2V480PConfig": ("fastvideo.configs.pipelines.hunyuan15", "Hunyuan15T2V480PConfig"),
    "Hunyuan15T2V720PConfig": ("fastvideo.configs.pipelines.hunyuan15", "Hunyuan15T2V720PConfig"),
    "HunyuanConfig": ("fastvideo.configs.pipelines.hunyuan", "HunyuanConfig"),
    "HunyuanGameCraftPipelineConfig": (
        "fastvideo.configs.pipelines.hunyuangamecraft",
        "HunyuanGameCraftPipelineConfig",
    ),
    "HYWorldConfig": ("fastvideo.configs.pipelines.hyworld", "HYWorldConfig"),
    "LucyEditDevConfig": ("fastvideo.configs.pipelines.wan", "LucyEditDevConfig"),
    "LTX2T2VConfig": ("fastvideo.pipelines.basic.ltx2.pipeline_configs", "LTX2T2VConfig"),
    "MatrixGame2I2V480PConfig": ("fastvideo.configs.pipelines.matrixgame2", "MatrixGame2I2V480PConfig"),
    "MatrixGame3I2V720PConfig": ("fastvideo.configs.pipelines.matrixgame3", "MatrixGame3I2V720PConfig"),
    "PipelineConfig": ("fastvideo.configs.pipelines.base", "PipelineConfig"),
    "SelfForcingWanT2V480PConfig": ("fastvideo.configs.pipelines.wan", "SelfForcingWanT2V480PConfig"),
    "WanI2V480PConfig": ("fastvideo.configs.pipelines.wan", "WanI2V480PConfig"),
    "WanI2V720PConfig": ("fastvideo.configs.pipelines.wan", "WanI2V720PConfig"),
    "WanT2V480PConfig": ("fastvideo.configs.pipelines.wan", "WanT2V480PConfig"),
    "WanT2V720PConfig": ("fastvideo.configs.pipelines.wan", "WanT2V720PConfig"),
    "get_pipeline_config_cls_from_name": ("fastvideo.registry", "get_pipeline_config_cls_from_name"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
