#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a densely indexed MiniMax-H3 transformer from selected source blocks.

The output stores selected blocks as ``transformer_blocks.0..N-1`` while its
config and manifest retain the exact source index for every local block.  All
shared transformer tensors are copied unchanged.  The source is never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
MANIFEST_NAME = "block_map_manifest.json"
BLOCK_KEY = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source transformer component directory.")
    parser.add_argument("--dst", type=Path, required=True, help="New transformer component directory.")
    parser.add_argument(
        "--block-map",
        required=True,
        help="Comma-separated source indices or a JSON file containing a list or a block_map field.",
    )
    parser.add_argument("--strategy", required=True, help="Selection strategy recorded in the manifest.")
    parser.add_argument("--source-model", required=True, help="Source model ID or persistent path.")
    parser.add_argument("--source-revision", required=True, help="Immutable source revision.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    return parser.parse_args()


def read_block_map(value: str) -> tuple[int, ...]:
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text())
        if isinstance(payload, dict):
            payload = payload.get("block_map")
        if not isinstance(payload, list):
            raise ValueError(f"{candidate} must contain a list or a block_map list")
        values = payload
    else:
        values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("block_map must not be empty")
    if any(isinstance(item, bool) for item in values):
        raise ValueError("block_map must contain integer source indices")
    try:
        return tuple(int(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ValueError("block_map must contain integer source indices") from error


def validate_block_map(block_map: tuple[int, ...], source_num_layers: int) -> None:
    MiniMaxH3Config().update_model_arch({
        "num_layers": len(block_map),
        "source_num_layers": source_num_layers,
        "block_map": block_map,
    })


def remap_block_key(name: str, source_to_local: dict[int, int]) -> str | None:
    match = BLOCK_KEY.match(name)
    if match is None:
        return name
    source_index = int(match.group(1))
    local_index = source_to_local.get(source_index)
    if local_index is None:
        return None
    return f"transformer_blocks.{local_index}.{match.group(2)}"


def prune_transformer(
    src: Path,
    dst: Path,
    block_map: tuple[int, ...],
    *,
    strategy: str,
    source_model: str,
    source_revision: str,
) -> dict:
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty destination: {dst}")
    source_config = json.loads((src / "config.json").read_text())
    source_num_layers = int(source_config["num_layers"])
    if source_config.get("block_map") is not None or source_config.get("source_num_layers") is not None:
        raise ValueError("source transformer is already pruned; compose block maps explicitly before conversion")
    validate_block_map(block_map, source_num_layers)

    source_index = json.loads((src / INDEX_NAME).read_text())
    source_weight_map: dict[str, str] = source_index["weight_map"]
    source_to_local = {source: local for local, source in enumerate(block_map)}
    dst.mkdir(parents=True, exist_ok=True)

    output_weight_map: dict[str, str] = {}
    total_size = 0
    kept_tensors = 0
    dropped_tensors = 0
    output_shards: list[str] = []
    for shard_name in sorted(set(source_weight_map.values())):
        output_state = {}
        with safe_open(src / shard_name, framework="pt", device="cpu") as source_file:
            for name in source_file.keys():
                output_name = remap_block_key(name, source_to_local)
                if output_name is None:
                    dropped_tensors += 1
                    continue
                if output_name in output_weight_map or output_name in output_state:
                    raise ValueError(f"duplicate output tensor after block remap: {output_name}")
                tensor = source_file.get_tensor(name)
                output_state[output_name] = tensor
                kept_tensors += 1
                total_size += tensor.numel() * tensor.element_size()
        if not output_state:
            continue
        save_file(output_state, dst / shard_name, metadata={"format": "pt"})
        output_shards.append(shard_name)
        output_weight_map.update({name: shard_name for name in output_state})

    output_config = dict(source_config)
    output_config.update({
        "num_layers": len(block_map),
        "source_num_layers": source_num_layers,
        "block_map": list(block_map),
        "block_map_strategy": strategy,
        "source_checkpoint": source_model,
        "source_revision": source_revision,
    })
    MiniMaxH3Config().update_model_arch(output_config)
    (dst / "config.json").write_text(json.dumps(output_config, indent=2, sort_keys=True) + "\n")
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    (dst / INDEX_NAME).write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")
    for extra in src.glob("*.json"):
        if extra.name not in {"config.json", INDEX_NAME, MANIFEST_NAME}:
            shutil.copy2(extra, dst / extra.name)

    map_digest = hashlib.sha256(json.dumps(list(block_map), separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "format_version": 1,
        "source_model": source_model,
        "source_revision": source_revision,
        "source_transformer": str(src.resolve()),
        "source_num_layers": source_num_layers,
        "num_layers": len(block_map),
        "block_map": list(block_map),
        "block_map_sha256": map_digest,
        "strategy": strategy,
        "kept_tensor_count": kept_tensors,
        "dropped_tensor_count": dropped_tensors,
        "total_size_bytes": total_size,
        "output_shards": output_shards,
        "output_transformer": str(dst.resolve()),
    }
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def log_wandb(manifest: dict, project: str | None, run_id: str | None, manifest_path: Path) -> None:
    if project is None and run_id is None:
        return
    if not project or not run_id:
        raise ValueError("--wandb-project and --wandb-run-id must be provided together")
    import os
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for tracked conversions")
    run = wandb.init(project=project, id=run_id, name=run_id, resume="allow", job_type="checkpoint-conversion",
                     config=manifest)
    artifact = wandb.Artifact(f"{run_id}-block-map", type="checkpoint-manifest", metadata=manifest)
    artifact.add_file(str(manifest_path))
    run.log_artifact(artifact)
    run.summary.update({
        "persistent_transformer": manifest["output_transformer"],
        "source_revision": manifest["source_revision"],
        "block_map_sha256": manifest["block_map_sha256"],
    })
    run.finish()


def main() -> None:
    args = parse_args()
    block_map = read_block_map(args.block_map)
    manifest = prune_transformer(
        args.src,
        args.dst,
        block_map,
        strategy=args.strategy,
        source_model=args.source_model,
        source_revision=args.source_revision,
    )
    log_wandb(manifest, args.wandb_project, args.wandb_run_id, args.dst / MANIFEST_NAME)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
