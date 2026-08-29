# SPDX-License-Identifier: Apache-2.0
"""Map-style loader for MiniMax H3 precomputed joint latent artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler, _parse_data_path_specs, passthrough
from fastvideo.distributed import get_sp_world_size, get_world_rank, get_world_size
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_REQUIRED_FIELDS = (
    "id",
    "caption",
    "text_sha1",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)


def _artifact_roots(path: str | Sequence[str] | dict[str, int]) -> list[tuple[Path, int]]:
    return [(Path(root).expanduser().resolve(), repeat) for root, repeat in _parse_data_path_specs(path)]


def is_minimax_h3_artifact_path(path: str | Sequence[str] | dict[str, int]) -> bool:
    """Return whether every configured root contains H3 JSONL manifests."""
    roots = _artifact_roots(path)
    return bool(roots) and all(root.is_dir() and any(root.glob("manifest_rank*.jsonl")) for root, _ in roots)


def _safe_artifact_name(value: Any, field: str) -> str:
    name = str(value)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"MiniMax H3 manifest field {field!r} is not a safe artifact name: {name!r}")
    return name


class MiniMaxH3ArtifactDataset(Dataset):
    """Load normalized video/audio latents and Qwen embeddings from safetensors."""

    def __init__(
        self,
        path: str | Sequence[str] | dict[str, int],
        *,
        batch_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        if batch_size != 1:
            raise ValueError("MiniMax H3 artifact batches must contain exactly one joint document")
        roots = _artifact_roots(path)
        if not roots:
            raise FileNotFoundError("No MiniMax H3 artifact roots were configured")

        records: list[tuple[Path, dict[str, Any]]] = []
        for root, repeat in roots:
            manifests = sorted(root.glob("manifest_rank*.jsonl"))
            if not manifests:
                raise FileNotFoundError(f"No manifest_rank*.jsonl files found under {root}")
            root_records: list[tuple[Path, dict[str, Any]]] = []
            for manifest in manifests:
                with manifest.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        missing = [field for field in _REQUIRED_FIELDS if field not in record]
                        if missing:
                            raise ValueError(f"{manifest}:{line_number} is missing required fields: {missing}")
                        _safe_artifact_name(record["id"], "id")
                        text_sha1 = _safe_artifact_name(record["text_sha1"], "text_sha1")
                        if len(text_sha1) != 40 or any(char not in "0123456789abcdef" for char in text_sha1.lower()):
                            raise ValueError(f"{manifest}:{line_number} has an invalid text_sha1")
                        root_records.append((root, record))
            records.extend(root_records * repeat)

        if not records:
            raise RuntimeError("MiniMax H3 artifact manifests contain no samples")
        self.records = records
        self.sampler = DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=len(records),
            num_sp_groups=num_sp_groups,
            sp_world_size=sp_world_size,
            global_rank=global_rank,
            drop_last=drop_last,
            seed=seed,
        )
        logger.info("Loaded %d MiniMax H3 artifact records from %d root(s)", len(records), len(roots))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        root, record = self.records[index]
        sample_id = _safe_artifact_name(record["id"], "id")
        text_sha1 = _safe_artifact_name(record["text_sha1"], "text_sha1")
        latent_path = root / "latents" / f"{sample_id}.safetensors"
        text_path = root / "text" / f"{text_sha1}.safetensors"
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing MiniMax H3 latent artifact: {latent_path}")
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing MiniMax H3 text artifact: {text_path}")

        latents = load_file(str(latent_path), device="cpu")
        text = load_file(str(text_path), device="cpu")
        if "video" not in latents or "audio" not in latents or "embed" not in text:
            raise ValueError(f"Incomplete MiniMax H3 safetensor pair for sample {sample_id}")
        video = latents["video"]
        audio_rows = latents["audio"]
        text_embedding = text["embed"]

        expected_video_shape = (
            1,
            24,
            int(record["num_latent_frames"]),
            int(record["latent_height"]),
            int(record["latent_width"]),
        )
        expected_audio_shape = (2 * int(record["num_audio_latents"]), 32)
        if tuple(video.shape) != expected_video_shape:
            raise ValueError(
                f"Video latent shape for {sample_id} is {tuple(video.shape)}, expected {expected_video_shape}")
        if tuple(audio_rows.shape) != expected_audio_shape:
            raise ValueError(
                f"Audio latent shape for {sample_id} is {tuple(audio_rows.shape)}, expected {expected_audio_shape}")
        if text_embedding.ndim != 2 or text_embedding.shape[0] == 0 or text_embedding.shape[1] != 5120:
            raise ValueError(f"Text embedding for {sample_id} must have shape [length, 5120]")
        for name, tensor in (("video", video), ("audio", audio_rows), ("text", text_embedding)):
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"MiniMax H3 {name} artifact for {sample_id} is not finite floating-point data")

        num_audio_latents = int(record["num_audio_latents"])
        audio = audio_rows.reshape(2, num_audio_latents, 32).permute(0, 2, 1).contiguous()
        return {
            "vae_latent": video[0].contiguous(),
            "audio_latent": audio,
            "text_embedding": text_embedding.contiguous(),
            "text_attention_mask": torch.ones(text_embedding.shape[0], dtype=torch.bool),
            "info": {
                "id": sample_id,
                "prompt": str(record["caption"]),
                "source_root": str(root),
            },
        }

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        if len(indices) != 1:
            raise ValueError("MiniMax H3 artifact batches must contain exactly one index")
        sample = self[indices[0]]
        return {
            "vae_latent": sample["vae_latent"].unsqueeze(0),
            "audio_latent": sample["audio_latent"].unsqueeze(0),
            "text_embedding": sample["text_embedding"].unsqueeze(0),
            "text_attention_mask": sample["text_attention_mask"].unsqueeze(0),
            "info_list": [sample["info"]],
        }


def build_minimax_h3_artifact_dataloader(
    path: str | Sequence[str] | dict[str, int],
    *,
    batch_size: int,
    num_data_workers: int,
    seed: int = 0,
) -> tuple[MiniMaxH3ArtifactDataset, StatefulDataLoader]:
    """Build a resumable DP/SP-aware loader for H3 artifact roots."""
    sp_world_size = get_sp_world_size()
    world_size = get_world_size()
    if world_size % sp_world_size:
        raise ValueError(f"World size {world_size} must be divisible by SP size {sp_world_size}")
    dataset = MiniMaxH3ArtifactDataset(
        path,
        batch_size=batch_size,
        num_sp_groups=world_size // sp_world_size,
        sp_world_size=sp_world_size,
        global_rank=get_world_rank(),
        seed=seed,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader


__all__ = [
    "MiniMaxH3ArtifactDataset",
    "build_minimax_h3_artifact_dataloader",
    "is_minimax_h3_artifact_path",
]
