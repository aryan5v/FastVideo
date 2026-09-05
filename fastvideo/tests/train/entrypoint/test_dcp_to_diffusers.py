# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _remove_existing_weight_files, )


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
