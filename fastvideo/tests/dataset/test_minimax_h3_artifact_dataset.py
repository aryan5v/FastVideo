# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the MiniMax H3 joint artifact loader."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.minimax_h3_artifact_dataset import (
    MiniMaxH3ArtifactDataset, is_minimax_h3_artifact_path)
from fastvideo.dataset.parquet_dataset_map_style import passthrough


def _write_sample(root, index: int, *, missing_text: bool = False) -> None:
    sample_id = f"sample_{index:02d}"
    caption = f"joint sample {index}"
    text_sha1 = hashlib.sha1(caption.encode()).hexdigest()
    (root / "latents").mkdir(parents=True, exist_ok=True)
    (root / "text").mkdir(parents=True, exist_ok=True)
    save_file({
        "video": torch.full((1, 24, 3, 4, 6), float(index), dtype=torch.float16),
        "audio": torch.arange(8 * 32, dtype=torch.float16).reshape(8, 32),
    }, str(root / "latents" / f"{sample_id}.safetensors"))
    if not missing_text:
        save_file({"embed": torch.full((index + 2, 5120), float(index), dtype=torch.float16)},
                  str(root / "text" / f"{text_sha1}.safetensors"))
    record = {
        "id": sample_id,
        "caption": caption,
        "text_sha1": text_sha1,
        "num_latent_frames": 3,
        "latent_height": 4,
        "latent_width": 6,
        "num_audio_latents": 4,
    }
    with (root / "manifest_rank00000.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _dataset(root, *, global_rank: int = 0, num_sp_groups: int = 1, sp_world_size: int = 1):
    return MiniMaxH3ArtifactDataset(
        str(root),
        batch_size=1,
        num_sp_groups=num_sp_groups,
        sp_world_size=sp_world_size,
        global_rank=global_rank,
        seed=17,
    )


def _loader(dataset):
    return StatefulDataLoader(dataset, batch_sampler=dataset.sampler, collate_fn=passthrough, num_workers=0)


def test_artifact_loader_returns_joint_training_shapes(tmp_path) -> None:
    _write_sample(tmp_path, 0)
    dataset = _dataset(tmp_path)

    assert is_minimax_h3_artifact_path(str(tmp_path))
    batch = dataset.__getitems__([0])
    assert batch["vae_latent"].shape == (1, 24, 3, 4, 6)
    assert batch["audio_latent"].shape == (1, 2, 32, 4)
    assert batch["text_embedding"].shape == (1, 2, 5120)
    assert batch["text_attention_mask"].shape == (1, 2)
    assert batch["text_attention_mask"].all()
    expected = torch.arange(8 * 32, dtype=torch.float16).reshape(2, 4, 32).permute(0, 2, 1)
    assert torch.equal(batch["audio_latent"][0], expected)
    assert batch["info_list"][0]["prompt"] == "joint sample 0"


def test_artifact_loader_fails_loudly_for_missing_tensor_file(tmp_path) -> None:
    _write_sample(tmp_path, 0, missing_text=True)
    dataset = _dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing MiniMax H3 text artifact"):
        dataset[0]


def test_sampler_repeats_document_within_sp_group_and_shards_dp_groups(tmp_path) -> None:
    for index in range(8):
        _write_sample(tmp_path, index)
    samplers = [
        list(_dataset(tmp_path, global_rank=rank, num_sp_groups=2, sp_world_size=2).sampler)
        for rank in range(4)
    ]
    assert samplers[0] == samplers[1]
    assert samplers[2] == samplers[3]
    assert samplers[0] != samplers[2]
    assert set(index for batch in samplers[0] + samplers[2] for index in batch) == set(range(8))


def test_stateful_loader_resume_continues_at_next_document(tmp_path) -> None:
    for index in range(5):
        _write_sample(tmp_path, index)
    first_loader = _loader(_dataset(tmp_path))
    first_iterator = iter(first_loader)
    first = next(first_iterator)["info_list"][0]["id"]
    second = next(first_iterator)["info_list"][0]["id"]
    state = first_loader.state_dict()
    expected_next = next(first_iterator)["info_list"][0]["id"]

    resumed_loader = _loader(_dataset(tmp_path))
    resumed_loader.load_state_dict(state)
    resumed_next = next(iter(resumed_loader))["info_list"][0]["id"]
    assert first != second
    assert resumed_next == expected_next
