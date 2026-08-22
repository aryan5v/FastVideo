# SPDX-License-Identifier: Apache-2.0
"""Validate all native H3 parquet and publish immutable MANIFEST/READY files."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any

BUCKET_RE = re.compile(r"^bucket=([1-9][0-9]*)x([1-9][0-9]*)-([1-9][0-9]*)f$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_shapes(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    width = int(row["width"])
    height = int(row["height"])
    frames = int(row["num_frames"])
    if frames % 17 != 5:
        raise ValueError(f"num_frames must be 17*n+5, got {frames}")
    video_frames = (frames - 5) // 17 * 5 + 2
    audio_frames = round(frames / 24 * 40)
    return [24, video_frames, height // 16, width // 16], [
        2,
        32,
        audio_frames,
    ]


@functools.lru_cache(maxsize=1)
def load_t2va_schema():
    """Load the authoritative schema file without importing GPU backends.

    Importing the top-level ``fastvideo`` package initializes Triton on this
    environment, which makes a login-node CPU finalizer fail before it can
    inspect parquet. The schema module itself has no GPU dependencies.
    """
    schema_path = Path(__file__).resolve().parents[3] / "fastvideo" / "dataset" / "dataloader" / "schema.py"
    spec = importlib.util.spec_from_file_location("_fastvideo_parquet_schema", schema_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load schema from {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pyarrow_schema_t2va


def inspect_parquet(
    path: Path,
    expected_bucket: str,
    expected_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    pyarrow_schema_t2va = load_t2va_schema()

    match = BUCKET_RE.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"{path}: parent must match bucket=<width>x<height>-<frames>f")
    actual_bucket = path.parent.name.removeprefix("bucket=")
    if actual_bucket != expected_bucket:
        raise ValueError(f"{path}: bucket {actual_bucket} != done marker {expected_bucket}")
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != pyarrow_schema_t2va:
        raise ValueError(f"{path}: schema does not equal pyarrow_schema_t2va")
    if parquet.num_row_groups != parquet.metadata.num_rows:
        raise ValueError(f"{path}: row_group_size must be exactly one")
    columns = [
        "id",
        "vae_latent_shape",
        "audio_latent_shape",
        "text_embedding_shape",
        "width",
        "height",
        "num_frames",
        "fps",
        "audio_sample_rate",
        "caption",
        "file_name",
        "duration_sec",
    ]
    rows = parquet.read(columns=columns).to_pylist()
    width, height, frames = (int(value) for value in match.groups())
    for row in rows:
        record_id = str(row["id"])
        if record_id not in expected_rows:
            raise ValueError(f"{path}: unexpected id {record_id}")
        expected_row = expected_rows[record_id]
        geometry = (int(row["width"]), int(row["height"]), int(row["num_frames"]))
        if geometry != (width, height, frames):
            raise ValueError(f"{path}/{record_id}: row geometry {geometry} != bucket {(width, height, frames)}")
        video_shape, audio_shape = expected_shapes(row)
        if row["vae_latent_shape"] != video_shape or row["audio_latent_shape"] != audio_shape:
            raise ValueError(
                f"{path}/{record_id}: latent shapes {row['vae_latent_shape']}/{row['audio_latent_shape']} "
                f"!= {video_shape}/{audio_shape}"
            )
        if len(row["text_embedding_shape"]) != 2 or row["text_embedding_shape"][1] != 5120:
            raise ValueError(f"{path}/{record_id}: invalid text shape {row['text_embedding_shape']}")
        if float(row["fps"]) != 24.0 or int(row["audio_sample_rate"]) != 32000:
            raise ValueError(f"{path}/{record_id}: invalid media rate")
        if row["caption"] != expected_row["prompt"]:
            raise ValueError(f"{path}/{record_id}: caption does not match frozen training prompt")
        if row["file_name"] != Path(expected_row["raw_video_path"]).name:
            raise ValueError(f"{path}/{record_id}: file_name does not match frozen raw video")
        if abs(float(row["duration_sec"]) - float(expected_row["duration_sec"])) > 1e-9:
            raise ValueError(f"{path}/{record_id}: duration does not match frozen manifest")
    return rows


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(text)
    os.replace(temporary, path)


def write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def audit_source(source_root: Path, source_summary: dict[str, Any]) -> dict[str, Any]:
    source = str(source_summary["source"])
    train_path = source_root / "media" / "train.jsonl"
    train_by_id = {str(row["conditioning_id"]): row for row in iter_jsonl(train_path)}
    expected_ids = set(train_by_id)
    observed_ids: set[str] = set()
    referenced_parquet: set[Path] = set()
    parquet_lengths: dict[Path, int] = {}
    parquet_hashes: dict[Path, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    done_paths = sorted((source_root / "done").glob("*.json")) if (source_root / "done").is_dir() else []
    for done_path in done_paths:
        done = json.loads(done_path.read_text())
        failures.update(done.get("failures", {}))
        parquet_value = done.get("parquet")
        if not parquet_value:
            raise ValueError(f"{done_path}: production done marker has no parquet")
        parquet_path = Path(parquet_value).resolve()
        data_root = (source_root / "data").resolve()
        if os.path.commonpath([str(data_root), str(parquet_path)]) != str(data_root):
            raise ValueError(f"{done_path}: parquet is outside source data root: {parquet_path}")
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        if parquet_path in referenced_parquet:
            raise ValueError(f"{source}: duplicate done marker reference to {parquet_path}")
        rows = inspect_parquet(parquet_path, str(done["bucket"]), train_by_id)
        referenced_parquet.add(parquet_path)
        parquet_lengths[parquet_path] = len(rows)
        parquet_hashes[parquet_path] = sha256_file(parquet_path)
        done_manifest = {str(row["conditioning_id"]): row for row in done.get("rows_manifest", [])}
        if len(done_manifest) != len(done.get("rows_manifest", [])):
            raise ValueError(f"{done_path}: duplicate id in rows_manifest")
        for parquet_row in rows:
            record_id = str(parquet_row["id"])
            if record_id in observed_ids:
                raise ValueError(f"{source}: duplicate encoded id {record_id}")
            if record_id not in done_manifest:
                raise ValueError(f"{done_path}: no rows_manifest entry for {record_id}")
            for shape_name in ("vae_latent_shape", "audio_latent_shape", "text_embedding_shape"):
                if done_manifest[record_id].get(shape_name) != parquet_row[shape_name]:
                    raise ValueError(f"{done_path}/{record_id}: {shape_name} disagrees with parquet")
            observed_ids.add(record_id)
            manifest_rows.append({
                **done_manifest[record_id],
                "parquet": str(parquet_path),
                "parquet_sha256": parquet_hashes[parquet_path],
            })

    all_data_parquet = {path.resolve() for path in (source_root / "data").rglob("*.parquet")}
    stray = sorted(all_data_parquet - referenced_parquet)
    if stray:
        raise ValueError(f"{source}: {len(stray)} unreferenced parquet files under data/; first={stray[0]}")
    missing_files = sorted(referenced_parquet - all_data_parquet)
    if missing_files:
        raise ValueError(f"{source}: referenced parquet missing from data scan; first={missing_files[0]}")
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if failures or missing or extra:
        raise ValueError(
            f"{source}: incomplete: encoded={len(observed_ids)}/{len(expected_ids)} "
            f"failures={len(failures)} missing={len(missing)} extra={len(extra)}"
        )
    expected_chunks = {
        str(chunk["chunk_id"])
        for chunk in json.loads((source_root / "work" / "worklist.json").read_text())["chunks"]
    }
    done_chunks = {path.stem for path in done_paths}
    if done_chunks != expected_chunks:
        raise ValueError(
            f"{source}: done/worklist mismatch missing={len(expected_chunks - done_chunks)} "
            f"extra={len(done_chunks - expected_chunks)}"
        )
    return {
        "source": source,
        "training_rows": len(expected_ids),
        "encoded_rows": len(observed_ids),
        "manifest_rows": sorted(manifest_rows, key=lambda item: item["conditioning_id"]),
        "parquet_files": tuple(str(path) for path in sorted(referenced_parquet)),
        "parquet_lengths": tuple(parquet_lengths[path] for path in sorted(referenced_parquet)),
        "parquet_hashes": {str(path): parquet_hashes[path] for path in sorted(referenced_parquet)},
        "train_manifest_sha256": sha256_file(train_path),
    }


def validate_cache(source_root: Path, audit: dict[str, Any]) -> None:
    cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with cache_path.open("rb") as handle:
        file_names, lengths = pickle.load(handle)
    if tuple(file_names) != audit["parquet_files"] or tuple(lengths) != audit["parquet_lengths"]:
        raise ValueError(f"{cache_path}: cache does not exactly match audited parquet set")


def verify_ready(root: Path) -> None:
    ready_path = root / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(ready_path)
    ready = json.loads(ready_path.read_text())
    frozen = json.loads((root / "FROZEN_MANIFEST.json").read_text())
    summaries = {str(item["source"]): item for item in frozen["sources"]}
    audited_rows = 0
    for source in ready["sources"]:
        source_root = root / str(source)
        if not (source_root / "READY.json").is_file():
            raise FileNotFoundError(source_root / "READY.json")
        manifest = json.loads((source_root / "MANIFEST.json").read_text())
        audit = audit_source(source_root, summaries[str(source)])
        validate_cache(source_root, audit)
        if manifest["training_rows"] != manifest["encoded_rows"] or manifest["encoded_rows"] != audit["encoded_rows"]:
            raise ValueError(f"{source}: incomplete finalized manifest")
        rows_path = source_root / "MANIFEST_rows.jsonl"
        if sha256_file(rows_path) != manifest["manifest_rows_sha256"]:
            raise ValueError(f"{source}: MANIFEST_rows.jsonl checksum mismatch")
        if manifest.get("parquet_sha256") != audit["parquet_hashes"]:
            raise ValueError(f"{source}: MANIFEST parquet checksum map mismatch")
        cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
        if sha256_file(cache_path) != manifest.get("map_style_cache_sha256"):
            raise ValueError(f"{source}: map-style cache checksum mismatch")
        stored_rows = list(iter_jsonl(rows_path))
        if stored_rows != audit["manifest_rows"]:
            raise ValueError(f"{source}: MANIFEST_rows.jsonl does not match current parquet audit")
        if sha256_file(source_root / "MANIFEST.json") != json.loads((source_root / "READY.json").read_text())[
            "manifest_sha256"
        ]:
            raise ValueError(f"{source}: READY manifest checksum mismatch")
        audited_rows += audit["encoded_rows"]
    if audited_rows != int(ready["training_rows"]):
        raise ValueError(f"root READY row count {ready['training_rows']} != audited {audited_rows}")
    print(f"verified READY dataset {root}: {audited_rows} rows, all parquet/schema/cache/hash checks passed")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.verify_only:
        verify_ready(root)
        return
    frozen = json.loads((root / "FROZEN_MANIFEST.json").read_text())
    if (root / "READY.json").exists():
        raise FileExistsError("dataset is already READY; use --verify-only")

    audits: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for source_summary in frozen["sources"]:
        source = str(source_summary["source"])
        source_root = root / source
        for final_name in ("MANIFEST_rows.jsonl", "MANIFEST.json", "READY.json"):
            if (source_root / final_name).exists():
                raise FileExistsError(f"refusing to overwrite immutable {source_root / final_name}")
        if (source_root / "data" / "map_style_cache" / "file_info.pkl").exists():
            raise FileExistsError(f"refusing to overwrite pre-existing map-style cache for {source}")
        audits.append((source_summary, source_root, audit_source(source_root, source_summary)))

    root_rows = 0
    source_manifests: list[dict[str, Any]] = []
    for source_summary, source_root, audit in audits:
        source = audit["source"]
        rows_path = source_root / "MANIFEST_rows.jsonl"
        rows_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit["manifest_rows"])
        write_text_atomic(rows_path, rows_text)
        cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
        write_pickle_atomic(cache_path, (audit["parquet_files"], audit["parquet_lengths"]))
        validate_cache(source_root, audit)
        manifest = {
            "schema_version": "minimax-h3-native-t2va-final-v1",
            "source": source,
            "training_rows": audit["training_rows"],
            "encoded_rows": audit["encoded_rows"],
            "validation_exclusions": int(source_summary["validation_exclusions"]),
            "manifest_rows_sha256": sha256_file(rows_path),
            "train_manifest_sha256": audit["train_manifest_sha256"],
            "parquet_files": len(audit["parquet_files"]),
            "parquet_sha256": audit["parquet_hashes"],
            "map_style_cache_sha256": sha256_file(cache_path),
            "bucket_contract": "bucket=<width>x<height>-<num_frames>f",
            "parquet_schema": "fastvideo.dataset.dataloader.schema.pyarrow_schema_t2va",
            "parquet_row_group_size": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        manifest_path = source_root / "MANIFEST.json"
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        ready_payload = {
            "schema_version": "minimax-h3-native-t2va-source-ready-v1",
            "source": source,
            "rows": audit["encoded_rows"],
            "manifest_sha256": sha256_file(manifest_path),
        }
        write_text_atomic(source_root / "READY.json", json.dumps(ready_payload, indent=2, sort_keys=True) + "\n")
        source_manifests.append(manifest)
        root_rows += audit["encoded_rows"]

    root_ready = {
        "schema_version": "minimax-h3-native-t2va-ready-v1",
        "training_rows": root_rows,
        "validation_rows": 64,
        "sources": [manifest["source"] for manifest in source_manifests],
        "data_paths": [str(root / manifest["source"] / "data") for manifest in source_manifests],
        "bucket_contract": "bucket=<width>x<height>-<num_frames>f",
        "preprocessed_data_type": "t2va",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    ready_path = root / "READY.json"
    temporary = root / f".{ready_path.name}.tmp"
    temporary.write_text(json.dumps(root_ready, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, ready_path)
    verify_ready(root)


if __name__ == "__main__":
    main()
