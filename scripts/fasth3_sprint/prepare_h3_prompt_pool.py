#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit H3 prompt sources and select a short-form teacher-cache pool."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2va", type=Path, required=True)
    parser.add_argument("--h3ext", type=Path, required=True)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _normalized_digest(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} contains invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, row


def _existing_prompt_digests(root: Path) -> tuple[set[str], int]:
    digests: set[str] = set()
    rows = 0
    manifests = sorted(root.glob("manifest_rank*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no manifest_rank*.jsonl files under {root}")
    for manifest in manifests:
        for line_number, row in _rows(manifest):
            caption = row.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{manifest}:{line_number} has no caption")
            digests.add(_normalized_digest(caption))
            rows += 1
    return digests, rows


def prepare_prompt_pool(t2va: Path, h3ext: Path, existing_corpus: Path, output_dir: Path) -> dict[str, Any]:
    existing_digests, existing_rows = _existing_prompt_digests(existing_corpus)
    source_hashes = {
        "t2va": _file_sha256(t2va),
        "h3ext": _file_sha256(h3ext),
    }
    all_digests: set[str] = set()
    all_ids: set[str] = set()
    overlap_between_sources = 0
    overlap_with_existing = Counter()
    source_counts = Counter()
    dimension_counts: dict[str, Counter[str]] = {}
    short_rows: list[dict[str, Any]] = []

    for line_number, row in _rows(t2va):
        identifier = row.get("id")
        prompt = row.get("prompt_compiled")
        if not isinstance(identifier, str) or not identifier or identifier in all_ids:
            raise ValueError(f"{t2va}:{line_number} has a missing or duplicate id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{t2va}:{line_number} has no prompt_compiled")
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "passed":
            raise ValueError(f"{t2va}:{line_number} did not pass its source validator")
        digest = _normalized_digest(prompt)
        if digest in all_digests:
            raise ValueError(f"{t2va}:{line_number} duplicates an earlier prompt")
        all_ids.add(identifier)
        all_digests.add(digest)
        source_counts["t2va"] += 1
        overlap_with_existing["t2va"] += digest in existing_digests
        sampling = row.get("sampling")
        runtime = row.get("runtime_config")
        if not isinstance(sampling, dict) or not isinstance(sampling.get("dimensions"), dict):
            raise ValueError(f"{t2va}:{line_number} has no sampling dimensions")
        if not isinstance(runtime, dict) or runtime.get("generate_audio") is not True:
            raise ValueError(f"{t2va}:{line_number} is not an audio-enabled H3 runtime row")
        dimensions = sampling["dimensions"]
        for name, value in dimensions.items():
            dimension_counts.setdefault(str(name), Counter())[str(value)] += 1
        if dimensions.get("duration_bucket") == "five_to_six_seconds":
            short_rows.append({
                "format_version": 1,
                "id": identifier,
                "prompt": prompt,
                "prompt_sha256": digest,
                "source": "t2va-prompts-50k",
                "source_file_sha256": source_hashes["t2va"],
                "source_line": line_number,
                "runtime_config": runtime,
                "sampling": sampling,
            })

    t2va_digests = set(all_digests)
    for line_number, row in _rows(h3ext):
        identifier = row.get("sample_id")
        prompt = row.get("prompt")
        if not isinstance(identifier, str) or not identifier or identifier in all_ids:
            raise ValueError(f"{h3ext}:{line_number} has a missing or duplicate sample_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{h3ext}:{line_number} has no prompt")
        digest = _normalized_digest(prompt)
        if digest in all_digests:
            overlap_between_sources += digest in t2va_digests
            raise ValueError(f"{h3ext}:{line_number} duplicates an earlier prompt")
        all_ids.add(identifier)
        all_digests.add(digest)
        source_counts["h3ext"] += 1
        overlap_with_existing["h3ext"] += digest in existing_digests

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "short_teacher_cache_selection.jsonl"
    temporary_selection = output_dir / ".short_teacher_cache_selection.jsonl.tmp"
    with temporary_selection.open("w", encoding="utf-8") as stream:
        for row in short_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary_selection.replace(selection_path)

    receipt = {
        "format_version": 1,
        "created_at_unix": time.time(),
        "sources": {
            "t2va": {
                "path": str(t2va.resolve()),
                "sha256": source_hashes["t2va"],
                "rows": source_counts["t2va"],
            },
            "h3ext": {
                "path": str(h3ext.resolve()),
                "sha256": source_hashes["h3ext"],
                "rows": source_counts["h3ext"],
            },
        },
        "unique_ids": len(all_ids),
        "unique_normalized_prompts": len(all_digests),
        "overlap_between_sources": overlap_between_sources,
        "existing_corpus": {
            "path": str(existing_corpus.resolve()),
            "rows": existing_rows,
            "unique_normalized_prompts": len(existing_digests),
        },
        "overlap_with_existing": dict(overlap_with_existing),
        "short_teacher_cache_selection": {
            "path": str(selection_path.resolve()),
            "rows": len(short_rows),
            "effective_unique_with_existing": len(short_rows) + len(existing_digests),
            "selection_rule": "t2va validation passed and duration_bucket=five_to_six_seconds",
        },
        "t2va_dimension_counts": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(dimension_counts.items())
        },
    }
    receipt_path = output_dir / "prompt_source_audit.json"
    temporary_receipt = output_dir / ".prompt_source_audit.json.tmp"
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary_receipt.replace(receipt_path)
    return receipt


def main() -> None:
    args = parse_args()
    receipt = prepare_prompt_pool(args.t2va, args.h3ext, args.existing_corpus, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
