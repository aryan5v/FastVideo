# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _remove_existing_weight_files,
    _role_model_checkpoint_state,
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
