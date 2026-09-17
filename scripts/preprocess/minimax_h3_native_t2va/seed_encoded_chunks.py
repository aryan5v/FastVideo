# SPDX-License-Identifier: Apache-2.0
"""Seed an extended H3 freeze with exact reusable chunks from a READY base.

A chunk is reusable only when its exact native shape, ordered conditioning-ID
list, and frozen training rows match. The copied or hardlinked parquet and its
rewritten done marker then pass the ordinary finalizer; unmatched chunks are
left untouched for GPU encoding workers.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

SEED_SCHEMA_VERSION = "minimax-h3-native-t2va-seed-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="source to seed (repeatable; default: every frozen source)",
    )
    parser.add_argument("--transfer", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_harness_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_minimax_h3_native_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings_sha256(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed = {str(row["conditioning_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{path}: duplicate conditioning id")
    return indexed


def chunk_signature(chunk: dict[str, Any]) -> tuple[int, int, int, tuple[str, ...]]:
    shape = chunk["shape"]
    ids = tuple(str(record_id) for record_id in chunk["conditioning_ids"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{chunk.get('chunk_id')}: duplicate conditioning id")
    return (
        int(shape["width"]),
        int(shape["height"]),
        int(shape["num_frames"]),
        ids,
    )


def bucket_for_signature(signature: tuple[int, int, int, tuple[str, ...]]) -> str:
    width, height, num_frames, _ = signature
    return f"{width}x{height}-{num_frames}f"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.seed.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def transfer_parquet(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.seed.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        if mode in {"auto", "hardlink"}:
            try:
                os.link(source, temporary)
                used = "hardlink"
            except OSError as error:
                fallback_errors = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP}
                if mode == "hardlink" or error.errno not in fallback_errors:
                    raise
                shutil.copy2(source, temporary)
                used = "copy"
        else:
            shutil.copy2(source, temporary)
            used = "copy"
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return used


def rewritten_done(
    rows_manifest: list[dict[str, Any]],
    *,
    base_root: Path,
    source: str,
    base_chunk_id: str,
    base_parquet: Path,
    new_chunk_id: str,
    bucket: str,
    parquet_path: Path,
    parquet_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "minimax-h3-native-t2va-done-v1",
        "chunk_id": new_chunk_id,
        "bucket": bucket,
        "parquet": str(parquet_path),
        "rows": len(rows_manifest),
        "failures": {},
        "rows_manifest": rows_manifest,
        "worker": "seed-encoded-chunks",
        "seeded_from": {
            "base_root": str(base_root),
            "source": source,
            "chunk_id": base_chunk_id,
            "parquet": str(base_parquet),
            "parquet_sha256": parquet_sha256,
        },
    }


def ensure_no_active_claims(source_root: Path) -> None:
    claims_root = source_root / "claims"
    if claims_root.is_dir() and any(claims_root.iterdir()):
        raise RuntimeError(f"{claims_root}: seed before launching GPU workers; active claims exist")


def ensure_seedable_source(source_root: Path) -> None:
    ensure_no_active_claims(source_root)
    for finalized_name in ("READY.json", "MANIFEST.json", "MANIFEST_rows.jsonl"):
        if (source_root / finalized_name).exists():
            raise FileExistsError(f"{source_root / finalized_name}: seed only an unfinalized extension")


def load_base_ready_receipts(
    base_root: Path,
    frozen: dict[str, Any],
    finalizer: Any,
) -> dict[str, dict[str, Any]]:
    """Validate lightweight publication receipts without scanning every parquet."""
    ready_path = base_root / "READY.json"
    ready = json.loads(ready_path.read_text())
    finalizer.validate_root_ready(base_root, frozen, ready)
    receipts: dict[str, dict[str, Any]] = {}
    for source_summary in frozen["sources"]:
        source = str(source_summary["source"])
        source_root = base_root / source
        manifest_path = source_root / "MANIFEST.json"
        source_ready_path = source_root / "READY.json"
        manifest = json.loads(manifest_path.read_text())
        source_ready = json.loads(source_ready_path.read_text())
        expected_source_ready = {
            "schema_version": finalizer.SOURCE_READY_SCHEMA_VERSION,
            "source": source,
            "rows": int(source_summary["training_rows"]),
            "manifest_sha256": sha256_file(manifest_path),
        }
        if source_ready != expected_source_ready:
            raise ValueError(f"{source_ready_path}: does not match the finalized source manifest")
        expected_manifest_fields = {
            "schema_version": finalizer.FINAL_SCHEMA_VERSION,
            "source": source,
            "training_rows": int(source_summary["training_rows"]),
            "encoded_rows": int(source_summary["training_rows"]),
            "validation_exclusions": int(source_summary["validation_exclusions"]),
            "train_manifest_sha256": source_summary["artifacts_sha256"]["train.jsonl"],
            "frozen_manifest_sha256": sha256_file(base_root / "FROZEN_MANIFEST.json"),
            "frozen_source_manifest_sha256": sha256_file(source_root / "MANIFEST.source.json"),
        }
        for field_name, expected_value in expected_manifest_fields.items():
            if manifest.get(field_name) != expected_value:
                raise ValueError(f"{manifest_path}: {field_name} does not match the frozen source receipt")
        rows_path = source_root / "MANIFEST_rows.jsonl"
        cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
        if manifest.get("manifest_rows_sha256") != sha256_file(rows_path):
            raise ValueError(f"{manifest_path}: MANIFEST_rows.jsonl checksum mismatch")
        if manifest.get("map_style_cache_sha256") != sha256_file(cache_path):
            raise ValueError(f"{manifest_path}: map-style cache checksum mismatch")
        rows = load_jsonl(rows_path)
        rows_by_id = {str(row["conditioning_id"]): row for row in rows}
        train_ids = set(index_rows(source_root / "media" / "train.jsonl"))
        if len(rows_by_id) != len(rows) or set(rows_by_id) != train_ids:
            raise ValueError(f"{rows_path}: rows do not exactly cover the frozen training manifest")
        parquet_hashes = manifest.get("parquet_sha256")
        if not isinstance(parquet_hashes, dict) or int(manifest.get("parquet_files", -1)) != len(parquet_hashes):
            raise ValueError(f"{manifest_path}: invalid parquet checksum receipt")
        for record_id, row in rows_by_id.items():
            parquet = str(Path(str(row.get("parquet", ""))).resolve())
            if row.get("parquet_sha256") != parquet_hashes.get(parquet):
                raise ValueError(f"{rows_path}/{record_id}: parquet receipt mismatch")
        receipts[source] = {
            "manifest": manifest,
            "rows_by_id": rows_by_id,
        }
    if sum(int(receipt["manifest"]["training_rows"]) for receipt in receipts.values()) != int(ready["training_rows"]):
        raise ValueError("base source receipts do not sum to root READY training rows")
    return receipts


def seed_source(
    base_root: Path,
    new_root: Path,
    source: str,
    base_receipt: dict[str, Any],
    transfer_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    base_source = base_root / source
    new_source = new_root / source
    ensure_seedable_source(new_source)

    base_worklist = json.loads((base_source / "work" / "worklist.json").read_text())
    new_worklist = json.loads((new_source / "work" / "worklist.json").read_text())
    base_by_signature: dict[tuple[int, int, int, tuple[str, ...]], dict[str, Any]] = {}
    for chunk in base_worklist["chunks"]:
        signature = chunk_signature(chunk)
        if signature in base_by_signature:
            raise ValueError(f"{source}: base worklist has duplicate chunk signature")
        base_by_signature[signature] = chunk

    base_rows = index_rows(base_source / "media" / "train.jsonl")
    new_rows = index_rows(new_source / "media" / "train.jsonl")
    reusable: list[tuple[dict[str, Any], dict[str, Any], tuple[int, int, int, tuple[str, ...]]]] = []
    for new_chunk in new_worklist["chunks"]:
        signature = chunk_signature(new_chunk)
        base_chunk = base_by_signature.get(signature)
        if base_chunk is None:
            continue
        ids = signature[-1]
        if any(record_id not in base_rows or new_rows.get(record_id) != base_rows[record_id] for record_id in ids):
            continue
        reusable.append((base_chunk, new_chunk, signature))

    reusable_chunk_ids = [str(new_chunk["chunk_id"]) for _, new_chunk, _ in reusable]
    reusable_chunk_id_set = set(reusable_chunk_ids)
    remaining_chunk_ids = [
        str(chunk["chunk_id"])
        for chunk in new_worklist["chunks"]
        if str(chunk["chunk_id"]) not in reusable_chunk_id_set
    ]
    counts = {
        "worklist_chunks": len(new_worklist["chunks"]),
        "worklist_sha256": sha256_file(new_source / "work" / "worklist.json"),
        "reusable_chunks": len(reusable),
        "reusable_chunk_ids_sha256": strings_sha256(reusable_chunk_ids),
        "seeded_chunks": 0,
        "already_seeded_chunks": 0,
        "recovered_parquet_chunks": 0,
        "hardlinked_chunks": 0,
        "copied_chunks": 0,
        "would_seed_chunks": 0,
        "remaining_gpu_chunks": len(new_worklist["chunks"]),
        "remaining_gpu_chunk_ids_sha256": strings_sha256(remaining_chunk_ids),
        "reused_rows": 0,
    }
    for base_chunk, new_chunk, signature in reusable:
        base_chunk_id = str(base_chunk["chunk_id"])
        new_chunk_id = str(new_chunk["chunk_id"])
        bucket = bucket_for_signature(signature)
        trusted_rows = [base_receipt["rows_by_id"][record_id] for record_id in signature[-1]]
        for record_id, row in zip(signature[-1], trusted_rows, strict=True):
            expected_provenance = {
                "conditioning_id": record_id,
                "source": source,
                "bucket": bucket,
                "raw_video_path": base_rows[record_id]["raw_video_path"],
            }
            if any(row.get(name) != value for name, value in expected_provenance.items()):
                raise ValueError(f"{source}/{base_chunk_id}/{record_id}: finalized row provenance mismatch")
        base_parquet_paths = {Path(str(row["parquet"])).resolve() for row in trusted_rows}
        if len(base_parquet_paths) != 1:
            raise ValueError(f"{source}/{base_chunk_id}: finalized rows do not share one parquet")
        base_parquet = base_parquet_paths.pop()
        if not base_parquet.is_file():
            raise FileNotFoundError(base_parquet)
        base_sha256 = sha256_file(base_parquet)
        recorded_sha256 = base_receipt["manifest"]["parquet_sha256"].get(str(base_parquet))
        if base_sha256 != recorded_sha256 or any(row.get("parquet_sha256") != base_sha256 for row in trusted_rows):
            raise ValueError(f"{base_parquet}: checksum does not match the finalized base receipt")
        rows_manifest = [
            {name: value for name, value in row.items() if name not in {"parquet", "parquet_sha256"}}
            for row in trusted_rows
        ]
        destination = (new_source / "data" / f"bucket={bucket}" / f"{new_chunk_id}.parquet").resolve()
        done_path = new_source / "done" / f"{new_chunk_id}.json"
        expected_done = rewritten_done(
            rows_manifest,
            base_root=base_root,
            source=source,
            base_chunk_id=base_chunk_id,
            base_parquet=base_parquet,
            new_chunk_id=new_chunk_id,
            bucket=bucket,
            parquet_path=destination,
            parquet_sha256=base_sha256,
        )
        if destination.exists():
            if not os.path.samefile(base_parquet, destination) and sha256_file(destination) != base_sha256:
                raise FileExistsError(f"{destination}: existing parquet does not match reusable base chunk")
            if done_path.exists():
                if json.loads(done_path.read_text()) != expected_done:
                    raise FileExistsError(f"{done_path}: existing done marker does not match seed contract")
                counts["already_seeded_chunks"] += 1
            elif dry_run:
                counts["would_seed_chunks"] += 1
            else:
                atomic_write_json(done_path, expected_done)
                counts["recovered_parquet_chunks"] += 1
                counts["seeded_chunks"] += 1
        elif done_path.exists():
            raise FileExistsError(f"{done_path}: done marker exists without its parquet")
        elif dry_run:
            counts["would_seed_chunks"] += 1
        else:
            used = transfer_parquet(base_parquet, destination, transfer_mode)
            if used == "hardlink" and not os.path.samefile(base_parquet, destination):
                raise OSError(f"{destination}: hardlink does not share the base parquet inode")
            if used == "copy" and sha256_file(destination) != base_sha256:
                raise OSError(f"{destination}: transferred parquet checksum mismatch")
            atomic_write_json(done_path, expected_done)
            counts["hardlinked_chunks" if used == "hardlink" else "copied_chunks"] += 1
            counts["seeded_chunks"] += 1
        counts["reused_rows"] += len(signature[-1])

    counts["remaining_gpu_chunks"] -= len(reusable)
    return counts


def main() -> None:
    args = parse_args()
    base_root = args.base_root.resolve()
    new_root = args.new_root.resolve()
    if base_root == new_root:
        raise ValueError("--base-root and --new-root must differ")
    freezer = load_harness_module("freeze_sources")
    finalizer = load_harness_module("finalize_dataset")
    new_manifest = freezer.verify_existing(new_root, emit_summary=False)
    if (new_root / "READY.json").exists():
        raise FileExistsError(f"{new_root}: extension is already READY")
    extension = new_manifest.get("extension")
    if not isinstance(extension, dict) or Path(str(extension.get("base_root", ""))).resolve() != base_root:
        raise ValueError("new root is not a verified extension of the requested base root")
    if extension.get("base_frozen_manifest_sha256") != sha256_file(base_root / "FROZEN_MANIFEST.json"):
        raise ValueError("new extension receipt does not match the requested base freeze")
    base_manifest = json.loads((base_root / "FROZEN_MANIFEST.json").read_text())
    base_receipts = load_base_ready_receipts(base_root, base_manifest, finalizer)

    frozen_sources = [str(summary["source"]) for summary in new_manifest["sources"]]
    sources = list(dict.fromkeys(str(source) for source in args.source)) or frozen_sources
    unknown = sorted(set(sources) - set(frozen_sources))
    if unknown:
        raise ValueError(f"unknown --source values: {unknown}")
    for source in sources:
        ensure_seedable_source(new_root / source)
    receipt = {
        "schema_version": SEED_SCHEMA_VERSION,
        "base_root": str(base_root),
        "new_root": str(new_root),
        "base_frozen_manifest_sha256": sha256_file(base_root / "FROZEN_MANIFEST.json"),
        "new_frozen_manifest_sha256": sha256_file(new_root / "FROZEN_MANIFEST.json"),
        "transfer_requested": args.transfer,
        "dry_run": bool(args.dry_run),
        "sources": {},
    }
    for source in sources:
        counts = seed_source(
            base_root,
            new_root,
            source,
            base_receipts[source],
            args.transfer,
            args.dry_run,
        )
        receipt["sources"][source] = counts
        print(f"{source}: {json.dumps(counts, sort_keys=True)}")
    if not args.dry_run:
        atomic_write_json(new_root / "SEED_RECEIPT.json", receipt)
        print(f"wrote {new_root / 'SEED_RECEIPT.json'}")


if __name__ == "__main__":
    main()
