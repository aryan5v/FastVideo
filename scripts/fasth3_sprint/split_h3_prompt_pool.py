#!/usr/bin/env python3
"""Create immutable train/validation/evaluation splits before teacher caching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--salt", default="fasth3-14b-rescue-v1")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("prompt pool is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt pool contains duplicate ids")

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "evaluation": []}
    for row in rows:
        digest = hashlib.sha256(f"{args.salt}:{row['id']}:{row['prompt_sha256']}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "evaluation"
        splits[split].append(row)
    if any(not values for values in splits.values()):
        raise RuntimeError("deterministic split unexpectedly produced an empty partition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, values in splits.items():
        values.sort(key=lambda row: hashlib.sha256(f"{args.salt}:{row['id']}".encode()).hexdigest())
        path = args.output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "salt": args.salt,
        "rule": "sha256(salt:id:prompt_sha256) modulo 100; train<80, validation<90, evaluation<100",
        "counts": {name: len(values) for name, values in splits.items()},
        "sha256": hashes,
    }
    (args.output_dir / "split_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
