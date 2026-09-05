# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _remove_existing_weight_files,
    _role_model_checkpoint_state,
    _strict_reload_verify,
)
from fastvideo.training.checkpointing_utils import ModelWrapper


def test_remove_existing_weight_files_removes_stale_shard_indexes(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    config = tmp_path / "config.json"
    weights.write_bytes(b"weights")
    index.write_text("{}")
    config.write_text("{}")

    _remove_existing_weight_files(tmp_path)

    assert not weights.exists()
    assert not index.exists()
    assert config.exists()


def test_role_model_checkpoint_state_excludes_optimizer_and_other_roles() -> None:
    student = SimpleNamespace(transformer=torch.nn.Linear(2, 2))
    teacher = SimpleNamespace(transformer=torch.nn.Linear(2, 2))
    method = SimpleNamespace(_role_models={"student": student, "teacher": teacher})

    states = _role_model_checkpoint_state(method, "student")

    assert list(states) == ["roles.student.transformer"]
    wrapper = states["roles.student.transformer"]
    assert isinstance(wrapper, ModelWrapper)
    assert wrapper.model is student.transformer


def test_role_model_checkpoint_state_rejects_unknown_role() -> None:
    method = SimpleNamespace(_role_models={"student": SimpleNamespace(transformer=torch.nn.Linear(2, 2))})

    with pytest.raises(KeyError, match="Unknown role 'critic'"):
        _role_model_checkpoint_state(method, "critic")


def test_strict_reload_verify_preserves_role_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_module_from_path(**kwargs: object) -> torch.nn.Module:
        calls.append(kwargs)
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(
        "fastvideo.train.utils.moduleloader.load_module_from_path",
        fake_load_module_from_path,
    )
    training_config = object()

    _strict_reload_verify(
        output_dir="/tmp/export",
        training_config=training_config,
        attention_backend="VIDEO_SPARSE_ATTN_H3",
    )

    assert calls == [{
        "model_path": "/tmp/export",
        "module_type": "transformer",
        "training_config": training_config,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    }]
