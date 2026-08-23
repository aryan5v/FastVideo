# SPDX-License-Identifier: Apache-2.0
"""Validate all native H3 parquet and publish immutable MANIFEST/READY files."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
from typing import Any

BUCKET_RE = re.compile(r"^bucket=([1-9][0-9]*)x([1-9][0-9]*)-([1-9][0-9]*)f$")
FINAL_SCHEMA_VERSION = "minimax-h3-native-t2va-final-v1"
SOURCE_READY_SCHEMA_VERSION = "minimax-h3-native-t2va-source-ready-v1"
ROOT_READY_SCHEMA_VERSION = "minimax-h3-native-t2va-ready-v1"
TENSOR_DTYPES = {
    "vae_latent": "float32",
    "audio_latent": "float32",
    "text_embedding": "float32",
}
DTYPE_ITEM_SIZES = {"float32": 4}


def packed_audio_latent_num_frames(num_frames: int) -> int:
    """Return the H3 packed-audio length on its 40 Hz clock."""
    return (5 * num_frames + 1) // 3


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


@functools.lru_cache(maxsize=1)
def load_freeze_sources():
    """Load the stdlib-only freezer module without importing GPU backends."""
    freeze_path = Path(__file__).with_name("freeze_sources.py")
    spec = importlib.util.spec_from_file_location("_minimax_h3_native_freeze_sources", freeze_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load freeze verifier from {freeze_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_sources(root: Path) -> dict[str, Any]:
    """Validate frozen hashes, provenance, and raw-video stats."""
    return load_freeze_sources().verify_existing(root, emit_summary=False)


def expected_shapes(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    width = int(row["width"])
    height = int(row["height"])
    frames = int(row["num_frames"])
    if frames % 17 != 5:
        raise ValueError(f"num_frames must be 17*n+5, got {frames}")
    if height % 16 or width % 16:
        raise ValueError(f"source geometry {width}x{height} is not divisible by the H3 VAE spatial ratio 16")
    video_frames = (frames - 5) // 17 * 5 + 2
    audio_frames = packed_audio_latent_num_frames(frames)
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
    import pyarrow.compute as pc
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
    if parquet.num_row_groups != parquet.metadata.num_rows or any(
        parquet.metadata.row_group(index).num_rows != 1 for index in range(parquet.num_row_groups)
    ):
        raise ValueError(f"{path}: row_group_size must be exactly one")
    columns = [
        "id",
        "vae_latent_bytes",
        "vae_latent_shape",
        "vae_latent_dtype",
        "audio_latent_bytes",
        "audio_latent_shape",
        "audio_latent_dtype",
        "text_embedding_bytes",
        "text_embedding_shape",
        "text_embedding_dtype",
        "media_type",
        "width",
        "height",
        "num_frames",
        "fps",
        "audio_sample_rate",
        "caption",
        "file_name",
        "duration_sec",
    ]
    metadata_columns = [column for column in columns if not column.endswith("_bytes")]
    rows: list[dict[str, Any]] = []
    for row_group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_index, columns=columns)
        row = table.select(metadata_columns).to_pylist()[0]
        for tensor_name in TENSOR_DTYPES:
            byte_length = pc.binary_length(table.column(f"{tensor_name}_bytes"))[0].as_py()
            row[f"{tensor_name}_nbytes"] = byte_length
        rows.append(row)
    width, height, frames = (int(value) for value in match.groups())
    for row in rows:
        record_id = str(row["id"])
        if record_id not in expected_rows:
            raise ValueError(f"{path}: unexpected id {record_id}")
        expected_row = expected_rows[record_id]
        expected_geometry = (
            int(expected_row["width"]),
            int(expected_row["height"]),
            int(expected_row["num_frames"]),
        )
        if (width, height, frames) != expected_geometry:
            raise ValueError(
                f"{path}/{record_id}: bucket geometry {(width, height, frames)} != frozen {expected_geometry}"
            )
        geometry = (int(row["width"]), int(row["height"]), int(row["num_frames"]))
        if geometry != expected_geometry:
            raise ValueError(f"{path}/{record_id}: row geometry {geometry} != frozen {expected_geometry}")
        video_shape, audio_shape = expected_shapes(expected_row)
        if row["vae_latent_shape"] != video_shape or row["audio_latent_shape"] != audio_shape:
            raise ValueError(
                f"{path}/{record_id}: latent shapes {row['vae_latent_shape']}/{row['audio_latent_shape']} "
                f"!= {video_shape}/{audio_shape}"
            )
        if (
            len(row["text_embedding_shape"]) != 2
            or int(row["text_embedding_shape"][0]) <= 0
            or row["text_embedding_shape"][1] != 5120
        ):
            raise ValueError(f"{path}/{record_id}: invalid text shape {row['text_embedding_shape']}")
        for tensor_name, expected_dtype in TENSOR_DTYPES.items():
            shape = row[f"{tensor_name}_shape"]
            dtype = row[f"{tensor_name}_dtype"]
            payload_length = row[f"{tensor_name}_nbytes"]
            if dtype != expected_dtype:
                raise ValueError(f"{path}/{record_id}: {tensor_name}_dtype {dtype!r} != {expected_dtype!r}")
            if not isinstance(shape, list) or not shape or any(int(dimension) <= 0 for dimension in shape):
                raise ValueError(f"{path}/{record_id}: invalid {tensor_name}_shape {shape}")
            if payload_length is None:
                raise ValueError(f"{path}/{record_id}: {tensor_name}_bytes is not a byte payload")
            expected_bytes = math.prod(int(dimension) for dimension in shape) * DTYPE_ITEM_SIZES[expected_dtype]
            if int(payload_length) != expected_bytes:
                raise ValueError(
                    f"{path}/{record_id}: {tensor_name}_bytes length {payload_length} != "
                    f"shape*dtype {expected_bytes}"
                )
        if row["media_type"] != "video_with_audio":
            raise ValueError(f"{path}/{record_id}: media_type {row['media_type']!r} != 'video_with_audio'")
        if float(row["fps"]) != float(expected_row["fps"]):
            raise ValueError(f"{path}/{record_id}: fps does not match frozen training row")
        if int(row["audio_sample_rate"]) != int(expected_row["audio_sample_rate"]):
            raise ValueError(f"{path}/{record_id}: audio sample rate does not match frozen training row")
        if row["caption"] != expected_row["prompt"]:
            raise ValueError(f"{path}/{record_id}: caption does not match frozen training prompt")
        if row["file_name"] != Path(expected_row["raw_video_path"]).name:
            raise ValueError(f"{path}/{record_id}: file_name does not match frozen raw video")
        if float(row["duration_sec"]) != float(expected_row["duration_sec"]):
            raise ValueError(f"{path}/{record_id}: duration does not match frozen training row")
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
    train_rows = list(iter_jsonl(train_path))
    train_by_id = {str(row["conditioning_id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError(f"{train_path}: duplicate conditioning id")
    train_manifest_sha256 = sha256_file(train_path)
    if train_manifest_sha256 != source_summary["artifacts_sha256"]["train.jsonl"]:
        raise ValueError(f"{source}: train manifest checksum does not match frozen artifact")
    expected_ids = set(train_by_id)
    worklist_path = source_root / "work" / "worklist.json"
    worklist = json.loads(worklist_path.read_text())
    chunks = {str(chunk["chunk_id"]): chunk for chunk in worklist["chunks"]}
    if len(chunks) != len(worklist["chunks"]):
        raise ValueError(f"{worklist_path}: duplicate chunk id")
    observed_ids: set[str] = set()
    referenced_parquet: set[Path] = set()
    parquet_lengths: dict[Path, int] = {}
    parquet_hashes: dict[Path, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    done_paths = sorted((source_root / "done").glob("*.json")) if (source_root / "done").is_dir() else []
    for done_path in done_paths:
        done = json.loads(done_path.read_text())
        chunk_id = done_path.stem
        if done.get("chunk_id") != chunk_id or chunk_id not in chunks:
            raise ValueError(f"{done_path}: chunk id does not match frozen worklist")
        chunk = chunks[chunk_id]
        expected_bucket = (
            f"{int(chunk['shape']['width'])}x{int(chunk['shape']['height'])}-"
            f"{int(chunk['shape']['num_frames'])}f"
        )
        if done.get("bucket") != expected_bucket:
            raise ValueError(f"{done_path}: bucket does not match frozen worklist")
        done_failures = done.get("failures", {})
        if not isinstance(done_failures, dict):
            raise ValueError(f"{done_path}: failures must be an object")
        failures.update(done_failures)
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
        rows = inspect_parquet(parquet_path, expected_bucket, train_by_id)
        if int(done.get("rows", -1)) != len(rows):
            raise ValueError(f"{done_path}: row count does not match parquet")
        parquet_ids = {str(row["id"]) for row in rows}
        expected_chunk_ids = {str(record_id) for record_id in chunk["conditioning_ids"]}
        if parquet_ids != expected_chunk_ids:
            raise ValueError(f"{done_path}: parquet ids do not exactly match frozen worklist chunk")
        referenced_parquet.add(parquet_path)
        parquet_lengths[parquet_path] = len(rows)
        parquet_hashes[parquet_path] = sha256_file(parquet_path)
        rows_manifest = done.get("rows_manifest")
        if not isinstance(rows_manifest, list):
            raise ValueError(f"{done_path}: rows_manifest must be a list")
        done_manifest = {str(row["conditioning_id"]): row for row in rows_manifest}
        if len(done_manifest) != len(rows_manifest):
            raise ValueError(f"{done_path}: duplicate id in rows_manifest")
        if set(done_manifest) != parquet_ids:
            raise ValueError(f"{done_path}: rows_manifest ids do not exactly match parquet")
        for parquet_row in rows:
            record_id = str(parquet_row["id"])
            if record_id in observed_ids:
                raise ValueError(f"{source}: duplicate encoded id {record_id}")
            if record_id not in done_manifest:
                raise ValueError(f"{done_path}: no rows_manifest entry for {record_id}")
            expected_row = train_by_id[record_id]
            expected_manifest_provenance = {
                "conditioning_id": record_id,
                "source": source,
                "bucket": expected_bucket,
                "raw_video_path": expected_row["raw_video_path"],
            }
            for field_name, expected_value in expected_manifest_provenance.items():
                if done_manifest[record_id].get(field_name) != expected_value:
                    raise ValueError(f"{done_path}/{record_id}: {field_name} disagrees with frozen training row")
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
    expected_chunks = set(chunks)
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
        "train_manifest_sha256": train_manifest_sha256,
    }


def validate_cache(source_root: Path, audit: dict[str, Any]) -> None:
    cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with cache_path.open("rb") as handle:
        file_names, lengths = pickle.load(handle)
    if tuple(file_names) != audit["parquet_files"] or tuple(lengths) != audit["parquet_lengths"]:
        raise ValueError(f"{cache_path}: cache does not exactly match audited parquet set")


def _source_manifest_payload(
    source_root: Path,
    source_summary: dict[str, Any],
    audit: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    rows_path = source_root / "MANIFEST_rows.jsonl"
    cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "source": audit["source"],
        "training_rows": audit["training_rows"],
        "encoded_rows": audit["encoded_rows"],
        "validation_exclusions": int(source_summary["validation_exclusions"]),
        "manifest_rows_sha256": sha256_file(rows_path),
        "train_manifest_sha256": audit["train_manifest_sha256"],
        "frozen_manifest_sha256": sha256_file(source_root.parent / "FROZEN_MANIFEST.json"),
        "frozen_source_manifest_sha256": sha256_file(source_root / "MANIFEST.source.json"),
        "parquet_files": len(audit["parquet_files"]),
        "parquet_sha256": audit["parquet_hashes"],
        "map_style_cache_sha256": sha256_file(cache_path),
        "bucket_contract": "bucket=<width>x<height>-<num_frames>f",
        "parquet_schema": "fastvideo.dataset.dataloader.schema.pyarrow_schema_t2va",
        "parquet_row_group_size": 1,
        "created_utc": created_utc,
    }


def validate_source_publication(
    source_root: Path,
    source_summary: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    source = str(audit["source"])
    rows_path = source_root / "MANIFEST_rows.jsonl"
    expected_rows_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit["manifest_rows"])
    if rows_path.read_text() != expected_rows_text:
        raise ValueError(f"{source}: MANIFEST_rows.jsonl does not match current parquet audit")
    validate_cache(source_root, audit)

    manifest_path = source_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    created_utc = manifest.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError(f"{source}: finalized manifest has no creation timestamp")
    expected_manifest = _source_manifest_payload(source_root, source_summary, audit, created_utc)
    if manifest != expected_manifest:
        raise ValueError(f"{source}: finalized manifest does not exactly match current audit")

    ready_path = source_root / "READY.json"
    ready = json.loads(ready_path.read_text())
    expected_ready = {
        "schema_version": SOURCE_READY_SCHEMA_VERSION,
        "source": source,
        "rows": audit["encoded_rows"],
        "manifest_sha256": sha256_file(manifest_path),
    }
    if ready != expected_ready:
        raise ValueError(f"{source}: READY.json does not exactly match finalized manifest")
    return manifest


def publish_source(
    source_root: Path,
    source_summary: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Publish or resume one source without overwriting mismatched artifacts."""
    source = str(audit["source"])
    rows_path = source_root / "MANIFEST_rows.jsonl"
    rows_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit["manifest_rows"])
    if rows_path.exists():
        if rows_path.read_text() != rows_text:
            raise ValueError(f"{source}: refusing to overwrite mismatched MANIFEST_rows.jsonl")
    else:
        write_text_atomic(rows_path, rows_text)

    cache_path = source_root / "data" / "map_style_cache" / "file_info.pkl"
    if cache_path.exists():
        validate_cache(source_root, audit)
    else:
        write_pickle_atomic(cache_path, (audit["parquet_files"], audit["parquet_lengths"]))
        validate_cache(source_root, audit)

    manifest_path = source_root / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        created_utc = manifest.get("created_utc")
        if not isinstance(created_utc, str) or not created_utc:
            raise ValueError(f"{source}: existing finalized manifest has no creation timestamp")
        expected_manifest = _source_manifest_payload(source_root, source_summary, audit, created_utc)
        if manifest != expected_manifest:
            raise ValueError(f"{source}: refusing to overwrite mismatched MANIFEST.json")
    else:
        manifest = _source_manifest_payload(
            source_root,
            source_summary,
            audit,
            dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    ready_path = source_root / "READY.json"
    ready_payload = {
        "schema_version": SOURCE_READY_SCHEMA_VERSION,
        "source": source,
        "rows": audit["encoded_rows"],
        "manifest_sha256": sha256_file(manifest_path),
    }
    if ready_path.exists():
        if json.loads(ready_path.read_text()) != ready_payload:
            raise ValueError(f"{source}: refusing to overwrite mismatched READY.json")
    else:
        write_text_atomic(ready_path, json.dumps(ready_payload, indent=2, sort_keys=True) + "\n")
    return validate_source_publication(source_root, source_summary, audit)


def _root_ready_payload(
    root: Path,
    frozen: dict[str, Any],
    training_rows: int,
    created_utc: str,
) -> dict[str, Any]:
    sources = [str(summary["source"]) for summary in frozen["sources"]]
    payload = {
        "schema_version": ROOT_READY_SCHEMA_VERSION,
        "frozen_manifest_sha256": sha256_file(root / "FROZEN_MANIFEST.json"),
        "training_rows": training_rows,
        "validation_rows": int(frozen["validation_rows"]),
        "sources": sources,
        "data_paths": [str(root / source / "data") for source in sources],
        "bucket_contract": "bucket=<width>x<height>-<num_frames>f",
        "preprocessed_data_type": "t2va",
        "created_utc": created_utc,
    }
    derivation = frozen.get("derivation")
    if isinstance(derivation, dict):
        payload.update({
            "validation_payload_path": derivation["validation_payload_path"],
            "validation_manifest_sha256": derivation["validation_manifest_sha256"],
            "validation_summary_sha256": derivation["validation_summary_sha256"],
            "validation_payload_sha256": derivation["validation_payload_sha256"],
        })
    return payload


def validate_root_ready(root: Path, frozen: dict[str, Any], ready: dict[str, Any]) -> None:
    created_utc = ready.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError("root READY has no creation timestamp")
    expected = _root_ready_payload(root, frozen, int(frozen["training_rows"]), created_utc)
    if ready.get("sources") != expected["sources"]:
        raise ValueError(f"root READY sources {ready.get('sources')} != frozen source set {expected['sources']}")
    if ready.get("data_paths") != expected["data_paths"]:
        raise ValueError(f"root READY data_paths {ready.get('data_paths')} != frozen paths {expected['data_paths']}")
    if ready != expected:
        raise ValueError("root READY does not exactly match frozen dataset contract")


def audit_published_sources(root: Path, frozen: dict[str, Any]) -> int:
    audited_rows = 0
    for source_summary in frozen["sources"]:
        source = str(source_summary["source"])
        source_root = root / source
        audit = audit_source(source_root, source_summary)
        validate_source_publication(source_root, source_summary, audit)
        audited_rows += int(audit["encoded_rows"])
    return audited_rows


def verify_ready(root: Path) -> None:
    root = root.resolve()
    ready_path = root / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(ready_path)
    frozen = verify_frozen_sources(root)
    ready = json.loads(ready_path.read_text())
    validate_root_ready(root, frozen, ready)
    audited_rows = audit_published_sources(root, frozen)
    if audited_rows != int(ready["training_rows"]):
        raise ValueError(f"root READY row count {ready['training_rows']} != audited {audited_rows}")
    print(f"verified READY dataset {root}: {audited_rows} rows, all frozen/parquet/cache/hash checks passed")


def finalize_dataset(root: Path) -> None:
    root = root.resolve()
    frozen = verify_frozen_sources(root)
    if (root / "READY.json").exists():
        raise FileExistsError("dataset is already READY; use --verify-only")

    audits: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for source_summary in frozen["sources"]:
        source = str(source_summary["source"])
        source_root = root / source
        audits.append((source_summary, source_root, audit_source(source_root, source_summary)))

    root_rows = 0
    for source_summary, source_root, audit in audits:
        publish_source(source_root, source_summary, audit)
        root_rows += int(audit["encoded_rows"])
    if root_rows != int(frozen["training_rows"]):
        raise ValueError(f"audited training rows {root_rows} != frozen {frozen['training_rows']}")

    # Repeat all read-only gates after publication, so a failure never leaves
    # an aggregate READY marker behind. READY.json is the final atomic write.
    frozen = verify_frozen_sources(root)
    root_rows = audit_published_sources(root, frozen)
    if root_rows != int(frozen["training_rows"]):
        raise ValueError(f"published training rows {root_rows} != frozen {frozen['training_rows']}")
    root_ready = _root_ready_payload(
        root,
        frozen,
        root_rows,
        dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_text_atomic(root / "READY.json", json.dumps(root_ready, indent=2, sort_keys=True) + "\n")
    validate_root_ready(root, frozen, json.loads((root / "READY.json").read_text()))
    print(f"finalized READY dataset {root}: {root_rows} rows, all frozen/parquet/cache/hash checks passed")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.verify_only:
        verify_ready(root)
        return
    finalize_dataset(root)


if __name__ == "__main__":
    main()
