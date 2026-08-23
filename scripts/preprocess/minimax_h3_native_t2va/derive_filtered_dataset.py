# SPDX-License-Identifier: Apache-2.0
"""Derive an immutable native-H3 dataset by filtering rare resolutions.

The filter is computed from aggregate frozen-row resolution counts, while the
base frozen inventory and held-out validation artifacts remain byte-identical.
Every retained parquet is hardlinked into a new self-contained path namespace;
the ordinary ``finalize_dataset.py`` remains responsible for publishing READY.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-resolution-count", type=int, default=10)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def load_harness_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_minimax_h3_native_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def filtered_training_rows(
    rows: list[dict[str, Any]],
    excluded_resolutions: set[str],
    freezer: Any,
) -> list[dict[str, Any]]:
    return [row for row in rows if freezer.resolution_key(row) not in excluded_resolutions]


def derivation_receipt(
    *,
    base_root: Path,
    output_root: Path,
    base_manifest: dict[str, Any],
    base_ready_sha256: str,
    base_config_sha256: str,
    config_sha256: str,
    min_resolution_count: int,
    frozen_by_source: dict[str, list[dict[str, Any]]],
    base_train_by_source: dict[str, list[dict[str, Any]]],
    derived_train_by_source: dict[str, list[dict[str, Any]]],
    validation_exclusions: dict[str, int],
    validation_manifest_sha256: str,
    validation_summary_sha256: str,
    heldout64_sha256: str,
    created_utc: str,
    freezer: Any,
) -> dict[str, Any]:
    counts = freezer.aggregate_resolution_counts(frozen_by_source)
    excluded_resolutions = sorted(
        resolution
        for resolution, count in counts.items()
        if count < min_resolution_count
    )
    excluded_set = set(excluded_resolutions)
    excluded_frozen_keys = {
        (source, str(row["conditioning_id"]))
        for source, rows in frozen_by_source.items()
        for row in rows
        if freezer.resolution_key(row) in excluded_set
    }
    excluded_training_keys = {
        (source, str(row["conditioning_id"]))
        for source, rows in base_train_by_source.items()
        for row in rows
        if freezer.resolution_key(row) in excluded_set
    }
    source_receipts = {
        source: {
            "frozen_rows": len(frozen_by_source[source]),
            "base_training_rows": len(base_train_by_source[source]),
            "derived_training_rows": len(derived_train_by_source[source]),
            "validation_exclusions": int(validation_exclusions[source]),
            "filter_exclusions": len(base_train_by_source[source]) - len(derived_train_by_source[source]),
        }
        for source in frozen_by_source
    }
    return {
        "schema_version": freezer.FILTERED_DERIVATION_SCHEMA_VERSION,
        "created_utc": created_utc,
        "base_root": str(base_root),
        "output_root": str(output_root),
        "base_ready_sha256": base_ready_sha256,
        "base_frozen_manifest_sha256": freezer.sha256_file(base_root / "FROZEN_MANIFEST.json"),
        "base_config_sha256": base_config_sha256,
        "config_sha256": config_sha256,
        "filter": {
            "axis": "aggregate_frozen_resolution",
            "comparison": "count < min_resolution_count",
            "min_resolution_count": min_resolution_count,
        },
        "frozen_resolution_counts": counts,
        "excluded_resolutions": excluded_resolutions,
        "excluded_frozen_rows": len(excluded_frozen_keys),
        "excluded_frozen_keys_sha256": freezer.conditioning_keys_sha256(excluded_frozen_keys),
        "excluded_training_rows": len(excluded_training_keys),
        "excluded_training_keys_sha256": freezer.conditioning_keys_sha256(excluded_training_keys),
        "base_frozen_rows": int(base_manifest["frozen_rows"]),
        "base_training_rows": int(base_manifest["training_rows"]),
        "derived_training_rows": sum(len(rows) for rows in derived_train_by_source.values()),
        "validation_manifest_sha256": validation_manifest_sha256,
        "validation_summary_sha256": validation_summary_sha256,
        "heldout64_sha256": heldout64_sha256,
        "sources": source_receipts,
    }


def copy_validation_tree(base_root: Path, stage: Path) -> None:
    base_validation = base_root / "validation"
    output_validation = stage / "validation"
    output_validation.mkdir(parents=True)
    for name in ("manifest.jsonl", "manifest.json", "heldout64.json"):
        shutil.copy2(base_validation / name, output_validation / name)
    output_videos = output_validation / "videos"
    output_videos.mkdir()
    for base_link in sorted((base_validation / "videos").iterdir()):
        if not base_link.is_symlink():
            raise ValueError(f"{base_link}: base validation media entry is not a symlink")
        (output_videos / base_link.name).symlink_to(os.readlink(base_link))


def hardlink_source_chunks(
    *,
    base_root: Path,
    stage: Path,
    output_root: Path,
    source: str,
    new_worklist: dict[str, Any],
    base_receipt: dict[str, Any],
    seed_module: Any,
) -> dict[str, int]:
    base_source = base_root / source
    stage_source = stage / source
    base_worklist = json.loads((base_source / "work" / "worklist.json").read_text())
    base_by_signature: dict[tuple[int, int, int, tuple[str, ...]], dict[str, Any]] = {}
    for chunk in base_worklist["chunks"]:
        signature = seed_module.chunk_signature(chunk)
        if signature in base_by_signature:
            raise ValueError(f"{source}: base worklist has duplicate chunk signature")
        base_by_signature[signature] = chunk

    base_train = seed_module.index_rows(base_source / "media" / "train.jsonl")
    derived_train = seed_module.index_rows(stage_source / "media" / "train.jsonl")
    linked_chunks = 0
    linked_rows = 0
    for new_chunk in new_worklist["chunks"]:
        signature = seed_module.chunk_signature(new_chunk)
        base_chunk = base_by_signature.get(signature)
        if base_chunk is None:
            raise ValueError(
                f"{source}/{new_chunk['chunk_id']}: filtered worklist chunk has no exact base shape/ID match"
            )
        ids = signature[-1]
        if any(
            record_id not in derived_train or derived_train[record_id] != base_train.get(record_id)
            for record_id in ids
        ):
            raise ValueError(f"{source}/{new_chunk['chunk_id']}: derived rows differ from the base")

        trusted_rows = [base_receipt["rows_by_id"][record_id] for record_id in ids]
        base_parquet_paths = {Path(str(row["parquet"])).resolve() for row in trusted_rows}
        if len(base_parquet_paths) != 1:
            raise ValueError(f"{source}/{base_chunk['chunk_id']}: finalized rows do not share one parquet")
        base_parquet = base_parquet_paths.pop()
        if not base_parquet.is_file():
            raise FileNotFoundError(base_parquet)
        base_sha256 = base_receipt["manifest"]["parquet_sha256"].get(str(base_parquet))
        if not isinstance(base_sha256, str) or any(row.get("parquet_sha256") != base_sha256 for row in trusted_rows):
            raise ValueError(f"{base_parquet}: base finalized checksum receipt is inconsistent")

        new_chunk_id = str(new_chunk["chunk_id"])
        base_chunk_id = str(base_chunk["chunk_id"])
        bucket = seed_module.bucket_for_signature(signature)
        physical_destination = stage_source / "data" / f"bucket={bucket}" / f"{new_chunk_id}.parquet"
        logical_destination = output_root / source / "data" / f"bucket={bucket}" / f"{new_chunk_id}.parquet"
        physical_destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(base_parquet, physical_destination)
        if not os.path.samefile(base_parquet, physical_destination):
            raise OSError(f"{physical_destination}: retained parquet is not a hardlink to its base")

        rows_manifest = [
            {name: value for name, value in row.items() if name not in {"parquet", "parquet_sha256"}}
            for row in trusted_rows
        ]
        done = seed_module.rewritten_done(
            rows_manifest,
            base_root=base_root,
            source=source,
            base_chunk_id=base_chunk_id,
            base_parquet=base_parquet,
            new_chunk_id=new_chunk_id,
            bucket=bucket,
            parquet_path=logical_destination,
            parquet_sha256=base_sha256,
        )
        write_json(stage_source / "done" / f"{new_chunk_id}.json", done)
        linked_chunks += 1
        linked_rows += len(ids)

    if linked_rows != len(derived_train):
        raise ValueError(f"{source}: hardlinked rows {linked_rows} != derived training rows {len(derived_train)}")
    return {"hardlinked_chunks": linked_chunks, "hardlinked_rows": linked_rows}


def derive_dataset(
    base_root: Path,
    output_root: Path,
    min_resolution_count: int,
) -> None:
    if min_resolution_count <= 0:
        raise ValueError("--min-resolution-count must be positive")
    if base_root == output_root or base_root in output_root.parents or output_root in base_root.parents:
        raise ValueError("--base-root and --output-root must be disjoint trees")
    if output_root.exists():
        raise FileExistsError(f"{output_root}: derived root already exists; use --verify-only")

    freezer = load_harness_module("freeze_sources")
    finalizer = load_harness_module("finalize_dataset")
    seed_module = load_harness_module("seed_encoded_chunks")
    finalizer.verify_ready(base_root)
    base_manifest = json.loads((base_root / "FROZEN_MANIFEST.json").read_text())
    base_receipts = seed_module.load_base_ready_receipts(base_root, base_manifest, finalizer)
    source_names = [str(summary["source"]) for summary in base_manifest["sources"]]
    base_summaries = {str(summary["source"]): summary for summary in base_manifest["sources"]}
    frozen_by_source = {
        source: freezer.load_frozen_rows(base_root, source)
        for source in source_names
    }
    base_train_by_source = {
        source: [row for _, row in freezer.iter_jsonl(base_root / source / "media" / "train.jsonl")]
        for source in source_names
    }
    counts = freezer.aggregate_resolution_counts(frozen_by_source)
    excluded_resolutions = {
        resolution
        for resolution, count in counts.items()
        if count < min_resolution_count
    }
    if not excluded_resolutions:
        raise ValueError("resolution filter selects no frozen resolutions; refusing a no-op derivation")
    derived_train_by_source = {
        source: filtered_training_rows(base_train_by_source[source], excluded_resolutions, freezer)
        for source in source_names
    }
    validation_exclusions = {
        source: int(base_summaries[source]["validation_exclusions"])
        for source in source_names
    }

    config_path = freezer.resolve_frozen_config(base_root, base_manifest)
    base_config_sha256 = freezer.sha256_file(config_path)
    derived_config = json.loads(config_path.read_text())
    derived_config["output_root"] = str(output_root)
    derived_config_text = canonical_json(derived_config)
    config_sha256 = text_sha256(derived_config_text)
    created_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = derivation_receipt(
        base_root=base_root,
        output_root=output_root,
        base_manifest=base_manifest,
        base_ready_sha256=freezer.sha256_file(base_root / "READY.json"),
        base_config_sha256=base_config_sha256,
        config_sha256=config_sha256,
        min_resolution_count=min_resolution_count,
        frozen_by_source=frozen_by_source,
        base_train_by_source=base_train_by_source,
        derived_train_by_source=derived_train_by_source,
        validation_exclusions=validation_exclusions,
        validation_manifest_sha256=freezer.sha256_file(base_root / "validation" / "manifest.jsonl"),
        validation_summary_sha256=freezer.sha256_file(base_root / "validation" / "manifest.json"),
        heldout64_sha256=freezer.sha256_file(base_root / "validation" / "heldout64.json"),
        created_utc=created_utc,
        freezer=freezer,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.derive-", dir=output_root.parent))
    try:
        (stage / "CONFIG.snapshot.json").write_text(derived_config_text)
        copy_validation_tree(base_root, stage)
        source_summaries: list[dict[str, Any]] = []
        hardlink_totals = {"hardlinked_chunks": 0, "hardlinked_rows": 0}
        for source in source_names:
            base_source = base_root / source
            stage_source = stage / source
            (stage_source / "media").mkdir(parents=True)
            (stage_source / "prompts").mkdir(parents=True)
            shutil.copy2(base_source / "media" / "frozen.jsonl", stage_source / "media" / "frozen.jsonl")
            shutil.copy2(base_source / "prompts" / "source.jsonl", stage_source / "prompts" / "source.jsonl")
            shutil.copy2(base_source / "prompts" / "SOURCE.sha256", stage_source / "prompts" / "SOURCE.sha256")
            freezer.write_jsonl(stage_source / "media" / "train.jsonl", derived_train_by_source[source])

            base_worklist = json.loads((base_source / "work" / "worklist.json").read_text())
            new_worklist = freezer.build_worklist(derived_train_by_source[source], int(base_worklist["chunk_size"]))
            new_worklist.update({
                "source": source,
                "train_manifest": str(output_root / source / "media" / "train.jsonl"),
                "set_root": str(output_root / source),
            })
            write_json(stage_source / "work" / "worklist.json", new_worklist)
            hardlinks = hardlink_source_chunks(
                base_root=base_root,
                stage=stage,
                output_root=output_root,
                source=source,
                new_worklist=new_worklist,
                base_receipt=base_receipts[source],
                seed_module=seed_module,
            )
            for name, value in hardlinks.items():
                hardlink_totals[name] += value

            artifacts_sha256 = {
                "source.jsonl": freezer.sha256_file(stage_source / "prompts" / "source.jsonl"),
                "frozen.jsonl": freezer.sha256_file(stage_source / "media" / "frozen.jsonl"),
                "train.jsonl": freezer.sha256_file(stage_source / "media" / "train.jsonl"),
                "worklist.json": freezer.sha256_file(stage_source / "work" / "worklist.json"),
            }
            ignored = {
                "artifacts_sha256",
                "training_rows",
                "validation_exclusions",
                "filter_exclusions",
                "extension",
                "derivation",
                "shape_counts",
            }
            source_summary = {
                **{name: value for name, value in base_summaries[source].items() if name not in ignored},
                "training_rows": len(derived_train_by_source[source]),
                "validation_exclusions": validation_exclusions[source],
                "filter_exclusions": len(base_train_by_source[source]) - len(derived_train_by_source[source]),
                "artifacts_sha256": artifacts_sha256,
                "shape_counts": freezer.distribution(frozen_by_source[source], ("width", "height", "num_frames")),
                "derivation": receipt["sources"][source],
            }
            write_json(stage_source / "MANIFEST.source.json", source_summary)
            source_summaries.append(source_summary)

        if hardlink_totals["hardlinked_rows"] != int(receipt["derived_training_rows"]):
            raise ValueError("derived hardlink row total does not match the filtered training rows")
        write_json(stage / "DERIVATION_RECEIPT.json", receipt)
        frozen_manifest = {
            "schema_version": freezer.SCHEMA_VERSION,
            "created_utc": created_utc,
            "config_path": str(output_root / "CONFIG.snapshot.json"),
            "config_source_path": str(config_path),
            "config_sha256": config_sha256,
            "seed": int(base_manifest["seed"]),
            "sources": source_summaries,
            "frozen_rows": sum(len(rows) for rows in frozen_by_source.values()),
            "training_rows": sum(len(rows) for rows in derived_train_by_source.values()),
            "validation_rows": int(base_manifest["validation_rows"]),
            "validation_manifest_sha256": freezer.sha256_file(stage / "validation" / "manifest.jsonl"),
            "heldout64_sha256": freezer.sha256_file(stage / "validation" / "heldout64.json"),
            "derivation": receipt,
            "derivation_receipt_sha256": freezer.sha256_file(stage / "DERIVATION_RECEIPT.json"),
            "ready": False,
            "ready_policy": (
                "READY.json is created only by finalize_dataset.py after every retained training row is validated."
            ),
        }
        write_json(stage / "FROZEN_MANIFEST.json", frozen_manifest)
        os.replace(stage, output_root)
    except Exception:
        print(f"derivation failed; preserved staging tree for inspection: {stage}")
        raise

    freezer.verify_existing(output_root)
    print(
        f"derived {output_root}: {receipt['derived_training_rows']} rows, "
        f"excluded resolutions={receipt['excluded_resolutions']}; run finalize_dataset.py next"
    )


def verify_dataset(
    base_root: Path,
    output_root: Path,
    min_resolution_count: int,
) -> None:
    freezer = load_harness_module("freeze_sources")
    finalizer = load_harness_module("finalize_dataset")
    finalizer.verify_ready(output_root)
    manifest = json.loads((output_root / "FROZEN_MANIFEST.json").read_text())
    receipt = freezer.load_filtered_derivation_receipt(output_root, manifest)
    if receipt is None:
        raise ValueError(f"{output_root}: frozen root is not a filtered derivation")
    if Path(str(receipt["base_root"])).resolve() != base_root:
        raise ValueError("--base-root does not match the derived receipt")
    if int(receipt["filter"]["min_resolution_count"]) != min_resolution_count:
        raise ValueError("--min-resolution-count does not match the derived receipt")


def main() -> None:
    args = parse_args()
    base_root = args.base_root.resolve()
    output_root = args.output_root.resolve()
    if args.min_resolution_count <= 0:
        raise ValueError("--min-resolution-count must be positive")
    if args.verify_only:
        verify_dataset(base_root, output_root, args.min_resolution_count)
        return
    derive_dataset(base_root, output_root, args.min_resolution_count)


if __name__ == "__main__":
    main()
