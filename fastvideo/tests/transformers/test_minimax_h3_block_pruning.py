# SPDX-License-Identifier: Apache-2.0
"""Synthetic checkpoint tests for MiniMax-H3 block pruning and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.checkpoint_conversion.prune_minimax_h3_blocks import INDEX_NAME, prune_transformer
from scripts.fasth3_sprint.validate_h3_candidate import validate_candidate


def _write_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": {
            "time_embedder.linear_1.weight": torch.tensor([[10.0]]),
            "transformer_blocks.0.attn.weight": torch.tensor([[0.0]]),
            "transformer_blocks.1.attn.weight": torch.tensor([[1.0]]),
        },
        "diffusion_pytorch_model-00002-of-00002.safetensors": {
            "transformer_blocks.2.attn.weight": torch.tensor([[2.0]]),
            "transformer_blocks.3.attn.weight": torch.tensor([[3.0]]),
            "proj_out.weight": torch.tensor([[20.0]]),
        },
    }
    weight_map = {}
    total_size = 0
    for shard_name, state in shards.items():
        save_file(state, source / shard_name, metadata={"format": "pt"})
        for name, tensor in state.items():
            weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()
    (source / INDEX_NAME).write_text(json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }))
    (source / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3Transformer3DModel",
        "num_layers": 4,
    }))
    return source


def _read_output_tensors(component: Path) -> dict[str, torch.Tensor]:
    index = json.loads((component / INDEX_NAME).read_text())
    tensors = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        with safe_open(component / shard_name, framework="pt", device="cpu") as shard:
            tensors.update({name: shard.get_tensor(name) for name in shard.keys()})
    return tensors


def test_pruning_reindexes_selected_blocks_and_keeps_shared_tensors(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    destination = tmp_path / "student"

    manifest = prune_transformer(
        source,
        destination,
        (0, 3),
        strategy="synthetic-test",
        source_model="test/h3",
        source_revision="deadbeef",
    )

    tensors = _read_output_tensors(destination)
    assert set(tensors) == {
        "time_embedder.linear_1.weight",
        "transformer_blocks.0.attn.weight",
        "transformer_blocks.1.attn.weight",
        "proj_out.weight",
    }
    torch.testing.assert_close(tensors["transformer_blocks.0.attn.weight"], torch.tensor([[0.0]]))
    torch.testing.assert_close(tensors["transformer_blocks.1.attn.weight"], torch.tensor([[3.0]]))
    config = json.loads((destination / "config.json").read_text())
    assert config["num_layers"] == 2
    assert config["source_num_layers"] == 4
    assert config["block_map"] == [0, 3]
    assert manifest["block_map"] == [0, 3]
    assert manifest["kept_tensor_count"] == 4
    assert manifest["dropped_tensor_count"] == 2
    assert len(manifest["block_map_sha256"]) == 64

    receipt = validate_candidate(
        destination,
        expected_source_kind="dense",
        expected_layers=2,
        min_parameters=4,
        max_parameters=4,
    )
    assert receipt["valid"] is True
    assert receipt["parameter_count"] == 4
    assert receipt["vsa_gate_count"] == 0
    assert (destination / "candidate_validation_receipt.json").is_file()


@pytest.mark.parametrize("block_map", [(0, 0), (0, 4), (3, 1)])
def test_pruning_rejects_invalid_maps(tmp_path: Path, block_map: tuple[int, ...]) -> None:
    source = _write_source(tmp_path)
    with pytest.raises(ValueError):
        prune_transformer(
            source,
            tmp_path / "student",
            block_map,
            strategy="invalid-test",
            source_model="test/h3",
            source_revision="deadbeef",
        )
