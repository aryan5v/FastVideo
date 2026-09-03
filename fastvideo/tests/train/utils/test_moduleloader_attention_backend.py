# SPDX-License-Identifier: Apache-2.0
"""Verify construction-scoped attention backends in the training loader."""

from __future__ import annotations

import json

import torch
import pytest

from fastvideo.attention.selector import _active_component_attention_backend_scope
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    TrainingConfig,
)


def test_load_transformer_scopes_attention_backend(monkeypatch, tmp_path) -> None:
    """Apply one backend while accepting trailing modular-manifest metadata."""
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    captured: list[tuple[AttentionBackendEnum | None, str | None]] = []

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path, **kwargs: {"transformer": ("diffusers", "FakeTransformer", {
            "subfolder": "transformer"
        })},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        scope = _active_component_attention_backend_scope()
        captured.append((scope.backend, scope.component) if scope else (None, None))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _fake_load_module,
    )

    result = moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
        attention_backend="ATTN_QAT_TRAIN",
    )

    assert isinstance(result, torch.nn.Module)
    assert captured == [(AttentionBackendEnum.ATTN_QAT_TRAIN, "transformer")]
    assert _active_component_attention_backend_scope() is None


def test_load_transformer_restores_backend_when_loading_fails(
    monkeypatch,
    tmp_path,
) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path, **kwargs: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _raise_during_load(**kwargs):
        del kwargs
        scope = _active_component_attention_backend_scope()
        assert scope is not None and scope.backend is AttentionBackendEnum.ATTN_QAT_TRAIN
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _raise_during_load,
    )

    with pytest.raises(RuntimeError, match="load failed"):
        moduleloader.load_module_from_path(
            model_path="fake/model",
            module_type="transformer",
            training_config=training_config,
            attention_backend="ATTN_QAT_TRAIN",
        )
    assert _active_component_attention_backend_scope() is None


def test_load_transformer_accepts_missing_unselected_modular_component(
    monkeypatch,
    tmp_path,
) -> None:
    """Validate only the component selected by a role-scoped training load."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({
            "_class_name": "MiniMaxH3ModularPipeline",
            "_diffusers_version": "0.36.0.dev0",
            "transformer": ["diffusers", "FakeTransformer"],
            "transformer_ref": ["diffusers", "FakeTransformer"],
        }))
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        lambda **kwargs: torch.nn.Linear(1, 1),
    )

    result = moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
    )

    assert isinstance(result, torch.nn.Module)
