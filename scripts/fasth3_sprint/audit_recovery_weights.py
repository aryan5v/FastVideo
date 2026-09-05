#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure whether a recovery checkpoint materially changed its initializer.

The audit streams one tensor at a time from safetensors so a 14B checkpoint
does not need to be duplicated in host memory. It also reads DCP metadata to
record the optimizer-state dtypes without materializing optimizer tensors.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open


_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")


def _weight_map(model_dir: Path) -> dict[str, Path]:
    transformer = model_dir / "transformer"
    indexes = sorted(transformer.glob("*.safetensors.index.json"))
    if indexes:
        with indexes[0].open(encoding="utf-8") as stream:
            raw = json.load(stream)
        return {key: transformer / filename for key, filename in raw["weight_map"].items()}

    files = sorted(transformer.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one safetensors file or an index under {transformer}")
    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
        return {key: files[0] for key in handle.keys()}


def _dcp_dtype_receipt(checkpoint_dir: Path) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(str(checkpoint_dir / "dcp")).read_metadata()
    counters: dict[str, Counter[str]] = {
        "parameters": Counter(),
        "optimizer": Counter(),
        "ema": Counter(),
    }
    examples: dict[str, list[str]] = {name: [] for name in counters}
    for key, value in metadata.state_dict_metadata.items():
        properties = getattr(value, "properties", None)
        dtype = str(getattr(properties, "dtype", "non_tensor"))
        if key.startswith("roles.student.transformer"):
            category = "parameters"
        elif "optimizers.student" in key:
            category = "optimizer"
        elif key.startswith("callbacks.ema.student_ema"):
            category = "ema"
        else:
            continue
        counters[category][dtype] += 1
        if len(examples[category]) < 8:
            examples[category].append(key)
    return {
        name: {
            "tensor_entries_by_dtype": dict(sorted(counter.items())),
            "example_keys": examples[name],
        }
        for name, counter in counters.items()
    }


def _group_name(key: str) -> str:
    match = _BLOCK_RE.search(key)
    if match:
        return f"transformer_blocks.{int(match.group(1)):02d}"
    return key.split(".", 1)[0]


def audit(initial_dir: Path, recovered_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    initial_map = _weight_map(initial_dir)
    recovered_map = _weight_map(recovered_dir)
    missing = sorted(set(initial_map) - set(recovered_map))
    unexpected = sorted(set(recovered_map) - set(initial_map))
    if missing or unexpected:
        raise ValueError(f"Weight-key mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")

    by_file: dict[Path, set[str]] = defaultdict(set)
    for mapping in (initial_map, recovered_map):
        for key, path in mapping.items():
            by_file[path].add(key)

    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "elements": 0.0,
            "unchanged_elements": 0.0,
            "base_sq": 0.0,
            "delta_sq": 0.0,
            "delta_abs_sum": 0.0,
            "delta_abs_max": 0.0,
        })
    dtype_pairs: Counter[str] = Counter()
    tensor_rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in by_file
        }
        for index, key in enumerate(sorted(initial_map)):
            before = handles[initial_map[key]].get_tensor(key)
            after = handles[recovered_map[key]].get_tensor(key)
            if before.shape != after.shape:
                raise ValueError(f"Shape mismatch for {key}: {before.shape} != {after.shape}")
            dtype_pairs[f"{before.dtype}->{after.dtype}"] += 1
            before_f = before.float()
            after_f = after.float()
            delta = after_f - before_f
            elements = before.numel()
            unchanged = int(torch.count_nonzero(before == after).item())
            base_sq = float(torch.sum(before_f * before_f, dtype=torch.float64).item())
            delta_sq = float(torch.sum(delta * delta, dtype=torch.float64).item())
            delta_abs_sum = float(torch.sum(delta.abs(), dtype=torch.float64).item())
            delta_abs_max = float(delta.abs().max().item()) if elements else 0.0
            row = {
                "name": key,
                "shape": list(before.shape),
                "initial_dtype": str(before.dtype),
                "recovered_dtype": str(after.dtype),
                "elements": elements,
                "unchanged_fraction": unchanged / elements if elements else 1.0,
                "relative_l2_update": math.sqrt(delta_sq / base_sq) if base_sq > 0 else None,
                "mean_abs_update": delta_abs_sum / elements if elements else 0.0,
                "max_abs_update": delta_abs_max,
            }
            tensor_rows.append(row)
            for group in ("all", _group_name(key)):
                target = totals[group]
                target["elements"] += elements
                target["unchanged_elements"] += unchanged
                target["base_sq"] += base_sq
                target["delta_sq"] += delta_sq
                target["delta_abs_sum"] += delta_abs_sum
                target["delta_abs_max"] = max(target["delta_abs_max"], delta_abs_max)
            if index % 100 == 0:
                print(f"audited {index + 1}/{len(initial_map)} tensors", flush=True)

    groups: dict[str, dict[str, Any]] = {}
    for name, values in sorted(totals.items()):
        elements = int(values["elements"])
        groups[name] = {
            "elements": elements,
            "unchanged_fraction": values["unchanged_elements"] / elements,
            "relative_l2_update": math.sqrt(values["delta_sq"] / values["base_sq"])
            if values["base_sq"] > 0 else None,
            "mean_abs_update": values["delta_abs_sum"] / elements,
            "max_abs_update": values["delta_abs_max"],
        }
    return {
        "initial_model": str(initial_dir.resolve()),
        "recovered_model": str(recovered_dir.resolve()),
        "training_checkpoint": str(checkpoint_dir.resolve()),
        "weight_key_count": len(initial_map),
        "dtype_pairs_by_tensor": dict(sorted(dtype_pairs.items())),
        "groups": groups,
        "tensors": tensor_rows,
        "dcp_dtypes": _dcp_dtype_receipt(checkpoint_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--recovered-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = audit(args.initial_model, args.recovered_model, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result["groups"]["all"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
