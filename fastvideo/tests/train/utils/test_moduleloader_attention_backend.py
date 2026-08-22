# SPDX-License-Identifier: Apache-2.0
"""Verify construction-scoped attention backends in the training loader."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from fastvideo.attention.selector import (
    _active_component_attention_backend_scope,
    _component_attention_backend_scope,
)
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.models.loader import component_loader
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    ModelTrainingConfig,
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
        lambda path: {"transformer": ("diffusers", "FakeTransformer", {
            "subfolder": "transformer"
        })},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        scope = _active_component_attention_backend_scope()
        captured.append((scope.backend, scope.component) if scope else (None, None))
        module = torch.nn.Linear(1, 1)
        module.config = SimpleNamespace(  # type: ignore[attr-defined]
            _resolved_attention_backend=scope.backend if scope else None, )
        return module

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


@pytest.mark.parametrize("role", ["student", "teacher", "critic"])
def test_role_attention_backend_receipt_matches_request(
    monkeypatch,
    tmp_path,
    role: str,
) -> None:
    """Every DMD role returns a construction receipt for its explicit backend."""
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    requested = (AttentionBackendEnum.ATTN_QAT_TRAIN
                 if role == "student" else AttentionBackendEnum.FLASH_ATTN)

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        scope = _active_component_attention_backend_scope()
        assert scope is not None
        module = torch.nn.Linear(1, 1)
        module.config = SimpleNamespace(  # type: ignore[attr-defined]
            _resolved_attention_backend=scope.backend, )
        return module

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fake_load_module)

    moduleloader.load_module_from_path(
        model_path=f"fake/{role}",
        module_type="transformer",
        training_config=training_config,
        disable_custom_init_weights=(role != "student"),
        attention_backend=requested,
    )


def test_role_attention_backend_receipt_mismatch_fails(monkeypatch, tmp_path) -> None:
    """A request that gets narrowed or lost may not silently start training."""
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        module = torch.nn.Linear(1, 1)
        module.config = SimpleNamespace(  # type: ignore[attr-defined]
            _resolved_attention_backend=AttentionBackendEnum.TORCH_SDPA, )
        return module

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fake_load_module)

    with pytest.raises(RuntimeError, match="requested attention backend FLASH_ATTN.*recorded TORCH_SDPA"):
        moduleloader.load_module_from_path(
            model_path="fake/teacher",
            module_type="transformer",
            training_config=training_config,
            disable_custom_init_weights=True,
            attention_backend="FLASH_ATTN",
        )


def test_teacher_critic_preserves_explicit_flash_attention_scope() -> None:
    """disable_custom_init_weights must not erase a role-local dense backend."""
    args = SimpleNamespace(
        _loading_teacher_critic_model=True,
        attention_backend=None,
    )

    with _component_attention_backend_scope(AttentionBackendEnum.FLASH_ATTN, component="teacher"):
        with component_loader._teacher_critic_attention_context(args):
            scope = _active_component_attention_backend_scope()
            assert scope is not None
            assert scope.backend is AttentionBackendEnum.FLASH_ATTN


def test_teacher_critic_masks_generator_only_qat_attention_scope() -> None:
    """The historical student-only QAT policy still narrows teacher/critic."""
    args = SimpleNamespace(
        _loading_teacher_critic_model=True,
        attention_backend=None,
    )

    with _component_attention_backend_scope(AttentionBackendEnum.ATTN_QAT_TRAIN, component="teacher"):
        with component_loader._teacher_critic_attention_context(args):
            scope = _active_component_attention_backend_scope()
            assert scope is not None
            assert scope.backend is None


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
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
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


def test_training_args_propagate_compile_settings() -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        model=ModelTrainingConfig(
            enable_torch_compile=True,
            torch_compile_kwargs={"dynamic": False},
        ),
        pipeline_config=PipelineConfig(),
    )

    args = moduleloader._make_training_args(training_config, model_path="fake/model")

    assert args.enable_torch_compile is True
    assert args.torch_compile_kwargs == {"dynamic": False}


def test_load_transformer_forwards_pre_fsdp_transform(monkeypatch, tmp_path) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )

    def transform(module):
        return module

    captured = None

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        nonlocal captured
        captured = getattr(kwargs["fastvideo_args"], "_pre_fsdp_transform", None)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(moduleloader.PipelineComponentLoader, "load_module", _fake_load_module)

    moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
        pre_fsdp_transform=transform,
    )

    assert captured is transform


def test_pre_fsdp_transform_rejects_non_transformer(monkeypatch, tmp_path) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"vae": ("diffusers", "FakeVAE")},
    )

    with pytest.raises(ValueError, match="only be set when loading a transformer"):
        moduleloader.load_module_from_path(
            model_path="fake/model",
            module_type="vae",
            training_config=training_config,
            pre_fsdp_transform=lambda module: module,
        )
