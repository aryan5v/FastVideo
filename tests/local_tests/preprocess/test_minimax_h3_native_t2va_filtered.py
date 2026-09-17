# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


HARNESS = Path(__file__).resolve().parents[3] / "scripts" / "preprocess" / "minimax_h3_native_t2va"


def load_harness_module(name: str):
    path = HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_minimax_h3_native_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(record_id: str, width: int, height: int, source: str = "source-a") -> dict[str, Any]:
    return {
        "conditioning_id": record_id,
        "source": source,
        "width": width,
        "height": height,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in rows))


def test_receipt_filters_aggregate_frozen_resolutions_strictly_below_threshold(tmp_path: Path) -> None:
    freezer = load_harness_module("freeze_sources")
    derive = load_harness_module("derive_filtered_dataset")
    base_root = tmp_path / "base"
    output_root = tmp_path / "derived"
    base_root.mkdir()
    (base_root / "FROZEN_MANIFEST.json").write_text("base freeze\n")

    frozen = {
        "source-a": [row(f"a-keep-{index}", 10, 10) for index in range(5)]
        + [row(f"a-drop-{index}", 20, 20) for index in range(4)],
        "source-b": [row(f"b-keep-{index}", 10, 10, "source-b") for index in range(5)]
        + [row(f"b-drop-{index}", 20, 20, "source-b") for index in range(5)],
    }
    base_train = {
        "source-a": frozen["source-a"][1:],
        "source-b": frozen["source-b"][2:],
    }
    excluded = {"20x20"}
    derived_train = {
        source: derive.filtered_training_rows(rows, excluded, freezer)
        for source, rows in base_train.items()
    }
    base_validation = [frozen["source-a"][0], frozen["source-a"][-1], frozen["source-b"][0]]
    derived_validation = [base_validation[0], base_validation[2]]
    excluded_validation = [base_validation[1]]
    receipt = derive.derivation_receipt(
        base_root=base_root,
        output_root=output_root,
        base_manifest={
            "frozen_rows": 19,
            "training_rows": 16,
            "sources": [
                {"source": "source-a", "validation_exclusions": 1},
                {"source": "source-b", "validation_exclusions": 2},
            ],
        },
        base_ready_sha256="a" * 64,
        base_config_sha256="b" * 64,
        config_sha256="c" * 64,
        min_resolution_count=10,
        frozen_by_source=frozen,
        base_train_by_source=base_train,
        derived_train_by_source=derived_train,
        validation_exclusions={"source-a": 1, "source-b": 1},
        base_validation=base_validation,
        derived_validation=derived_validation,
        excluded_validation=excluded_validation,
        base_validation_manifest_sha256="d" * 64,
        base_validation_summary_sha256="e" * 64,
        base_heldout64_sha256="f" * 64,
        validation_manifest_sha256="0" * 64,
        validation_summary_sha256="1" * 64,
        validation_payload_path="validation/heldout2.json",
        validation_payload_sha256="2" * 64,
        created_utc="2026-08-23T00:00:00+00:00",
        freezer=freezer,
    )

    assert receipt["frozen_resolution_counts"] == {"10x10": 10, "20x20": 9}
    assert receipt["excluded_resolutions"] == ["20x20"]
    assert receipt["excluded_frozen_rows"] == 9
    assert receipt["excluded_training_rows"] == 9
    assert receipt["derived_training_rows"] == 7
    assert receipt["base_validation_rows"] == 3
    assert receipt["derived_validation_rows"] == 2
    assert receipt["excluded_validation_rows"] == 1
    assert receipt["excluded_validation_conditioning_ids"] == [base_validation[1]["conditioning_id"]]
    assert receipt["training_holdout_policy"] == "preserve_base_validation_conditioning_ids"
    assert receipt["sources"]["source-a"]["filter_exclusions"] == 4
    assert receipt["sources"]["source-b"]["filter_exclusions"] == 5
    assert receipt["sources"]["source-a"]["removed_validation_id_exclusions"] == 1
    assert receipt["sources"]["source-b"]["removed_validation_id_exclusions"] == 0


def test_filtered_validation_summary_distinguishes_holdouts_from_current_membership(tmp_path: Path) -> None:
    freezer = load_harness_module("freeze_sources")
    retained = [
        {
            "conditioning_id": "kept",
            "source": "source-a",
            "family": "nuva",
            "width": 768,
            "height": 768,
            "num_frames": 124,
            "fps": 24.0,
        }
    ]
    summary = freezer.filtered_validation_summary(
        {
            "seed": 7,
            "rows": 2,
            "training_exclusions_by_source": {"source-a": 2},
        },
        retained,
        {"source-a": 1},
        base_root=tmp_path / "base",
        min_resolution_count=10,
        excluded_resolutions=["576x576"],
        excluded_rows=[{"conditioning_id": "removed"}],
    )

    assert summary["training_exclusions_by_source"] == {"source-a": 2}
    assert summary["validation_membership_by_source"] == {"source-a": 1}


def test_derivation_receipt_file_is_sha_anchored(tmp_path: Path) -> None:
    freezer = load_harness_module("freeze_sources")
    receipt = {
        "schema_version": freezer.FILTERED_DERIVATION_SCHEMA_VERSION,
        "filter": {
            "axis": "aggregate_frozen_resolution",
            "comparison": "count < min_resolution_count",
            "min_resolution_count": 10,
        },
        "excluded_resolutions": ["576x576"],
        "excluded_validation_conditioning_ids": ["sample-rare"],
        "derived_validation_rows": 60,
        "validation_payload_path": "validation/heldout60.json",
    }
    receipt_path = tmp_path / "DERIVATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {
        "derivation": receipt,
        "derivation_receipt_sha256": freezer.sha256_file(receipt_path),
    }
    assert freezer.load_filtered_derivation_receipt(tmp_path, manifest) == receipt

    receipt_path.write_text(receipt_path.read_text() + " ")
    with pytest.raises(ValueError, match="sha256"):
        freezer.load_filtered_derivation_receipt(tmp_path, manifest)


@pytest.mark.parametrize("output", ["base", "base/derived", "."])
def test_derivation_rejects_same_or_nested_output_tree(tmp_path: Path, output: str) -> None:
    derive = load_harness_module("derive_filtered_dataset")
    base_root = (tmp_path / "base").resolve()
    output_root = (tmp_path / output).resolve()

    with pytest.raises(ValueError, match="disjoint trees"):
        derive.derive_dataset(base_root, output_root, 10)


def test_retained_chunk_is_hardlinked_and_done_path_is_rewritten(tmp_path: Path) -> None:
    derive = load_harness_module("derive_filtered_dataset")
    seed_module = load_harness_module("seed_encoded_chunks")
    base_root = tmp_path / "base"
    stage = tmp_path / "stage"
    output_root = tmp_path / "final"
    source = "source-a"
    shape = {"width": 768, "height": 768, "num_frames": 124}
    ids = ["sample-a", "sample-b"]
    rows = [row(record_id, 768, 768) for record_id in ids]
    write_jsonl(base_root / source / "media" / "train.jsonl", rows)
    write_jsonl(stage / source / "media" / "train.jsonl", rows)
    base_worklist = {
        "chunks": [{"chunk_id": "c00007", "shape": shape, "conditioning_ids": ids}],
    }
    (base_root / source / "work").mkdir(parents=True)
    (base_root / source / "work" / "worklist.json").write_text(json.dumps(base_worklist))
    new_worklist = {
        "chunks": [{"chunk_id": "c00003", "shape": shape, "conditioning_ids": ids}],
    }
    base_parquet = base_root / source / "data" / "bucket=768x768-124f" / "c00007.parquet"
    base_parquet.parent.mkdir(parents=True)
    base_parquet.write_bytes(b"immutable parquet payload")
    parquet_sha256 = "1" * 64
    base_receipt = {
        "manifest": {"parquet_sha256": {str(base_parquet.resolve()): parquet_sha256}},
        "rows_by_id": {
            record_id: {
                "conditioning_id": record_id,
                "parquet": str(base_parquet.resolve()),
                "parquet_sha256": parquet_sha256,
                "source": source,
                "bucket": "768x768-124f",
            }
            for record_id in ids
        },
    }

    counts = derive.hardlink_source_chunks(
        base_root=base_root,
        stage=stage,
        output_root=output_root,
        source=source,
        new_worklist=new_worklist,
        base_receipt=base_receipt,
        seed_module=seed_module,
    )

    linked = stage / source / "data" / "bucket=768x768-124f" / "c00003.parquet"
    assert counts == {"hardlinked_chunks": 1, "hardlinked_rows": 2}
    assert os.path.samefile(base_parquet, linked)
    done = json.loads((stage / source / "done" / "c00003.json").read_text())
    assert done["chunk_id"] == "c00003"
    assert done["parquet"] == str(output_root / source / "data" / "bucket=768x768-124f" / "c00003.parquet")
    assert done["seeded_from"]["chunk_id"] == "c00007"
