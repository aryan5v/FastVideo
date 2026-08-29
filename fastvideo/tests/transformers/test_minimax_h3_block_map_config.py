# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for MiniMax H3 pruned-block checkpoint metadata."""

import json

import pytest

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config


def test_default_h3_architecture_is_unchanged_and_resolves_identity_map() -> None:
    arch = MiniMaxH3ArchConfig()

    assert arch.num_layers == 50
    assert arch.source_num_layers is None
    assert arch.block_map is None
    assert arch.source_block_indices() == tuple(range(50))


def test_pruned_block_map_is_normalized_and_serializable() -> None:
    selected = [0, 3, 7, 12, 18, 24, 31, 39, 44, 49]
    arch = MiniMaxH3ArchConfig(num_layers=len(selected), source_num_layers=50, block_map=selected)

    assert arch.block_map == tuple(selected)
    assert arch.source_block_indices() == tuple(selected)
    checkpoint_metadata = {
        "num_layers": arch.num_layers,
        "source_num_layers": arch.source_num_layers,
        "block_map": arch.block_map,
    }
    assert json.loads(json.dumps(checkpoint_metadata))["block_map"] == selected


def test_hf_config_update_preserves_pruned_block_provenance() -> None:
    config = MiniMaxH3Config()
    selected = [0, 4, 10, 17, 23, 31, 40, 49]

    config.update_model_arch({
        "num_layers": len(selected),
        "source_num_layers": 50,
        "block_map": selected,
    })

    assert config.arch_config.block_map == tuple(selected)
    assert config.arch_config.source_block_indices() == tuple(selected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_layers": 2, "source_num_layers": 50}, "must define block_map"),
        ({"num_layers": 2, "block_map": [0, 49]}, "requires source_num_layers"),
        ({"num_layers": 3, "source_num_layers": 50, "block_map": [0, 49]}, "length must equal"),
        ({"num_layers": 3, "source_num_layers": 50, "block_map": [0, 7, 7]}, "strictly increasing"),
        ({"num_layers": 2, "source_num_layers": 50, "block_map": [0, 50]}, "must be in"),
        ({"num_layers": 2, "source_num_layers": 50, "block_map": [0, True]}, "integer indices"),
    ],
)
def test_invalid_pruned_block_metadata_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MiniMaxH3ArchConfig(**kwargs)
