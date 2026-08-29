#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless a pruned MiniMax-H3 transformer is release-gate valid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
MANIFEST_NAME = "block_map_manifest.json"
RECEIPT_NAME = "candidate_validation_receipt.json"
BLOCK_KEY = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
VSA_GATE_SUFFIX = "attn.to_gate_compress.weight"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--expected-source-kind", choices=("dense", "vsa"), required=True)
    parser.add_argument("--expected-layers", type=int, default=20)
    parser.add_argument("--min-parameters", type=int, default=13_700_000_000)
    parser.add_argument("--max-parameters", type=int, default=13_900_000_000)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _tensor_sha256(component: Path, weight_map: dict[str, str], name: str) -> str:
    shard = weight_map.get(name)
    if shard is None:
        raise ValueError(f"missing tensor {name!r} from {component / INDEX_NAME}")
    with safe_open(component / shard, framework="pt", device="cpu") as source:
        tensor = source.get_tensor(name).contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _inventory(component: Path, weight_map: dict[str, str]) -> tuple[int, int]:
    parameter_count = 0
    tensor_bytes = 0
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(str(shard), []).append(name)
    for shard, expected_names in names_by_shard.items():
        with safe_open(component / shard, framework="pt", device="cpu") as source:
            for name in expected_names:
                tensor_slice = source.get_slice(name)
                numel = math.prod(tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                dtype_bytes = DTYPE_BYTES.get(dtype)
                if dtype_bytes is None:
                    raise ValueError(f"unsupported candidate tensor dtype {dtype!r} for {name}")
                parameter_count += numel
                tensor_bytes += numel * dtype_bytes
    return parameter_count, tensor_bytes


def validate_candidate(
    transformer: Path,
    *,
    expected_source_kind: str,
    expected_layers: int,
    min_parameters: int,
    max_parameters: int,
) -> dict[str, Any]:
    config = _read_json(transformer / "config.json")
    manifest = _read_json(transformer / MANIFEST_NAME)
    index = _read_json(transformer / INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("candidate index must contain a non-empty weight_map")

    block_map = manifest.get("block_map")
    if not isinstance(block_map, list) or len(block_map) != expected_layers:
        raise ValueError(f"manifest must contain exactly {expected_layers} selected blocks")
    if block_map != sorted(set(block_map)):
        raise ValueError("manifest block_map must be strictly increasing and unique")
    map_digest = hashlib.sha256(json.dumps(block_map, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("block_map_sha256") != map_digest:
        raise ValueError("manifest block_map_sha256 does not match block_map")
    for field in ("num_layers", "source_num_layers", "block_map", "source_revision"):
        if config.get(field) != manifest.get(field):
            raise ValueError(f"config and manifest disagree on {field}")
    if int(config["num_layers"]) != expected_layers:
        raise ValueError(f"candidate config does not contain {expected_layers} blocks")

    block_indices = sorted({int(match.group(1)) for name in weight_map if (match := BLOCK_KEY.match(name))})
    if block_indices != list(range(expected_layers)):
        raise ValueError(f"candidate tensor index has invalid local block indices: {block_indices}")
    missing_shards = sorted({str(shard) for shard in weight_map.values() if not (transformer / str(shard)).is_file()})
    if missing_shards:
        raise ValueError(f"candidate index references missing shards: {missing_shards}")
    shard_bytes = sum((transformer / str(shard)).stat().st_size for shard in set(weight_map.values()))
    parameter_count, tensor_bytes = _inventory(transformer, weight_map)
    indexed_tensor_bytes = int(index.get("metadata", {}).get("total_size", -1))
    if tensor_bytes != indexed_tensor_bytes or tensor_bytes != int(manifest.get("total_size_bytes", -2)):
        raise ValueError("index and manifest disagree on total tensor bytes")
    if not min_parameters <= parameter_count <= max_parameters:
        raise ValueError(f"candidate parameter count {parameter_count:,} is outside the authorized range")

    gate_names = sorted(name for name in weight_map if name.endswith(VSA_GATE_SUFFIX))
    expected_gate_count = expected_layers if expected_source_kind == "vsa" else 0
    if len(gate_names) != expected_gate_count:
        raise ValueError(
            f"{expected_source_kind} candidate has {len(gate_names)} VSA gate tensors; expected {expected_gate_count}")

    source_transformer = Path(str(manifest["source_transformer"]))
    source_index = _read_json(source_transformer / INDEX_NAME)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict):
        raise ValueError("source index must contain weight_map")
    verified_gate_hashes: dict[str, str] = {}
    for local_index, source_index_value in enumerate(block_map):
        local_name = f"transformer_blocks.{local_index}.{VSA_GATE_SUFFIX}"
        source_name = f"transformer_blocks.{source_index_value}.{VSA_GATE_SUFFIX}"
        if expected_source_kind == "dense":
            if source_name in source_weight_map:
                raise ValueError("dense source unexpectedly contains a VSA gate tensor")
            continue
        local_digest = _tensor_sha256(transformer, weight_map, local_name)
        source_digest = _tensor_sha256(source_transformer, source_weight_map, source_name)
        if local_digest != source_digest:
            raise ValueError(f"VSA gate tensor changed while mapping source block {source_index_value}")
        verified_gate_hashes[local_name] = local_digest

    receipt = {
        "format_version": 1,
        "validated_at_unix": time.time(),
        "transformer": str(transformer.resolve()),
        "source_kind": expected_source_kind,
        "source_revision": manifest["source_revision"],
        "block_map": block_map,
        "block_map_sha256": map_digest,
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "shard_bytes": shard_bytes,
        "tensor_count": len(weight_map),
        "vsa_gate_count": len(gate_names),
        "verified_vsa_gate_sha256": verified_gate_hashes,
        "valid": True,
    }
    temporary = transformer / f".{RECEIPT_NAME}.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(transformer / RECEIPT_NAME)
    return receipt


def main() -> None:
    args = parse_args()
    receipt = validate_candidate(
        args.transformer,
        expected_source_kind=args.expected_source_kind,
        expected_layers=args.expected_layers,
        min_parameters=args.min_parameters,
        max_parameters=args.max_parameters,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
