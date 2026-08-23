# SPDX-License-Identifier: Apache-2.0
"""Freeze H3 T2VA source inventories and build a deterministic validation split.

Only records present in all three authoritative inputs are eligible:

* at least one completed producer status record;
* a canonical ``<videos_dir>/<conditioning_id>.mp4`` file; and
* a valid, non-empty prompt record.

The resulting manifests are immutable inputs to native-shape preprocessing.
The script deliberately never searches recursively for videos, so producer
``server_tmp`` and ``_archive`` files cannot leak into training.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat as stat_module
import tempfile
from typing import Any, Iterable

SCHEMA_VERSION = "minimax-h3-native-t2va-freeze-v1"
VALIDATION_SCHEMA_VERSION = "minimax-h3-native-t2va-validation-v1"
EXTENSION_SCHEMA_VERSION = "minimax-h3-native-t2va-extension-v1"
FILTERED_DERIVATION_SCHEMA_VERSION = "minimax-h3-native-t2va-filtered-derivation-v2"
EXPECTED_FPS = 24.0
EXPECTED_AUDIO_SAMPLE_RATE = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("v10_sources.json"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify an already-frozen tree without consulting live sources",
    )
    mode.add_argument(
        "--extend-existing",
        type=Path,
        default=None,
        metavar="BASE_ROOT",
        help="create a new immutable freeze by extending a verified base root",
    )
    parser.add_argument(
        "--extend-source",
        action="append",
        default=[],
        metavar="SOURCE",
        help="source allowed to gain live eligible rows (repeatable; requires --extend-existing)",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, record


def read_jsonl_snapshot(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], str, int]:
    """Read and hash the exact same byte snapshot, including a live append log."""
    with path.open("rb") as handle:
        snapshot_size = os.fstat(handle.fileno()).st_size
        raw = handle.read(snapshot_size)
    # A live producer can be between bytes of its next append. Freeze only
    # complete newline-terminated records from the captured fd extent.
    if raw and not raw.endswith(b"\n"):
        raw = raw.rpartition(b"\n")[0] + b"\n"
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append((line_number, record))
    return rows, hashlib.sha256(raw).hexdigest(), len(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def media_value(status: dict[str, Any], name: str, default: Any = None) -> Any:
    media = status.get("media")
    if isinstance(media, dict) and media.get(name) is not None:
        return media[name]
    return status.get(name, default)


def duration_band(num_frames: int, fps: float) -> str:
    seconds = num_frames / fps
    if seconds < 7.0:
        return "05-07s"
    if seconds < 11.0:
        return "07-11s"
    return "11-15s"


def source_inventory(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = str(spec["name"])
    videos_dir = Path(spec["videos_dir"])
    status_path = Path(spec["status_jsonl"])
    prompts_path = Path(spec["prompts_jsonl"])
    for path in (videos_dir, status_path, prompts_path):
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing required input {path}")

    status_rows, status_sha256, status_snapshot_bytes = read_jsonl_snapshot(status_path)
    completed: dict[str, tuple[int, dict[str, Any]]] = {}
    completed_lines = 0
    for line_number, record in status_rows:
        if record.get("status") != "completed":
            continue
        record_id = str(record.get("id") or record.get("sample_id") or "")
        if not record_id:
            raise ValueError(f"{status_path}:{line_number}: completed row has no id")
        completed[record_id] = (line_number, record)
        completed_lines += 1

    # Deliberately non-recursive: server_tmp and _archive are not canonical.
    canonical_mp4s = {path.stem: path.resolve() for path in videos_dir.glob("*.mp4") if path.is_file()}
    eligible_ids = set(completed) & set(canonical_mp4s)

    prompt_id_field = str(spec["prompt_id_field"])
    prompt_text_field = str(spec["prompt_text_field"])
    prompts: dict[str, tuple[int, str]] = {}
    prompt_records_seen = 0
    for line_number, record in iter_jsonl(prompts_path):
        prompt_records_seen += 1
        record_id = str(record.get(prompt_id_field) or "")
        if record_id not in eligible_ids:
            continue
        if spec.get("require_prompt_validation_passed") and record.get("validation", {}).get("status") != "passed":
            continue
        prompt = record.get(prompt_text_field)
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if record_id in prompts:
            raise ValueError(f"{prompts_path}: duplicate eligible prompt id {record_id}")
        prompts[record_id] = (line_number, prompt.strip())

    rows: list[dict[str, Any]] = []
    for record_id in sorted(eligible_ids & set(prompts)):
        status_line, status = completed[record_id]
        prompt_line, prompt = prompts[record_id]
        width = int(media_value(status, "width", 0))
        height = int(media_value(status, "height", 0))
        num_frames = int(media_value(status, "frames", status.get("num_frames", 0)))
        fps = float(media_value(status, "fps", status.get("fps", 0.0)))
        audio_rate = int(media_value(status, "audio_sample_rate", 0))
        audio_channels = int(media_value(status, "audio_channels", 0))
        if width <= 0 or height <= 0 or num_frames <= 0:
            raise ValueError(f"{name}/{record_id}: invalid media geometry {width}x{height}x{num_frames}")
        if abs(fps - EXPECTED_FPS) > 1e-6:
            raise ValueError(f"{name}/{record_id}: fps {fps} != {EXPECTED_FPS}")
        if audio_rate != EXPECTED_AUDIO_SAMPLE_RATE or audio_channels != 2:
            raise ValueError(
                f"{name}/{record_id}: expected stereo {EXPECTED_AUDIO_SAMPLE_RATE} Hz audio, "
                f"got channels={audio_channels} rate={audio_rate}"
            )
        video_path = canonical_mp4s[record_id]
        stat = video_path.stat()
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "source": name,
            "family": str(spec["family"]),
            "conditioning_id": record_id,
            "prompt": prompt,
            "raw_video_path": str(video_path),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_sec": num_frames / fps,
            "audio_sample_rate": audio_rate,
            "audio_channels": audio_channels,
            "audio_samples": int(media_value(status, "audio_samples", 0) or 0),
            "audio_duration_sec": float(media_value(status, "audio_duration_s", 0.0) or 0.0),
            "bucket_id": str(status.get("bucket_id") or ""),
            "status_line": status_line,
            "prompt_line": prompt_line,
            "video_size_bytes": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
        })

    stats = {
        "source": name,
        "completed_status_ids": len(completed),
        "completed_status_lines": completed_lines,
        "canonical_mp4s": len(canonical_mp4s),
        "completed_and_canonical_mp4": len(eligible_ids),
        "prompt_records_seen": prompt_records_seen,
        "frozen_rows": len(rows),
        "missing_canonical_mp4": len(set(completed) - set(canonical_mp4s)),
        "canonical_mp4_without_completed_status": len(set(canonical_mp4s) - set(completed)),
        "eligible_without_valid_prompt": len(eligible_ids - set(prompts)),
        "status_jsonl": str(status_path.resolve()),
        "status_snapshot_sha256": status_sha256,
        "status_snapshot_bytes": status_snapshot_bytes,
        "prompts_jsonl": str(prompts_path.resolve()),
        "prompts_snapshot_sha256": sha256_file(prompts_path),
        "prompts_snapshot_bytes": prompts_path.stat().st_size,
        "videos_dir": str(videos_dir.resolve()),
    }
    return rows, stats


def select_stratified(
    rows: list[dict[str, Any]],
    quota: int,
    seed: int,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["conditioning_id"] in excluded_ids:
            continue
        key = (row["width"], row["height"], duration_band(row["num_frames"], row["fps"]))
        strata[key].append(row)
    rng = random.Random(seed)
    keys = sorted(strata)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(strata[key])

    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < quota and keys:
        key = keys[cursor % len(keys)]
        bucket = strata[key]
        while bucket and bucket[-1]["conditioning_id"] in excluded_ids:
            bucket.pop()
        if bucket:
            row = bucket.pop()
            selected.append(row)
            excluded_ids.add(row["conditioning_id"])
        keys = [candidate for candidate in keys if strata[candidate]]
        cursor += 1
    if len(selected) != quota:
        raise ValueError(f"could select only {len(selected)} of requested {quota} rows")
    return selected


def build_worklist(rows: list[dict[str, Any]], chunk_size: int) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[str]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["width"], row["height"], row["num_frames"])].append(row["conditioning_id"])
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for shape in sorted(grouped):
        ids = sorted(grouped[shape])
        for start in range(0, len(ids), chunk_size):
            width, height, num_frames = shape
            chunks.append({
                "chunk_id": f"c{chunk_index:05d}",
                "shape": {"width": width, "height": height, "num_frames": num_frames},
                "conditioning_ids": ids[start:start + chunk_size],
            })
            chunk_index += 1
    return {
        "schema_version": "minimax-h3-native-t2va-worklist-v1",
        "chunk_size": chunk_size,
        "rows": len(rows),
        "chunks": chunks,
    }


def worklist_chunk_signature(chunk: dict[str, Any]) -> tuple[int, int, int, tuple[str, ...]]:
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


def validation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "source": row["source"],
        "family": row["family"],
        "conditioning_id": row["conditioning_id"],
        "prompt": row["prompt"],
        "raw_video_path": row["raw_video_path"],
        "width": row["width"],
        "height": row["height"],
        "num_frames": row["num_frames"],
        "fps": row["fps"],
        "duration_sec": row["duration_sec"],
        "audio_sample_rate": row["audio_sample_rate"],
        "audio_channels": row["audio_channels"],
        "audio_samples": row["audio_samples"],
        "audio_duration_sec": row["audio_duration_sec"],
        "bucket_id": row["bucket_id"],
    }


def heldout_payload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "data": [
            {
                "caption": row["prompt"],
                "ref_video": row["raw_video_path"],
                "source": row["source"],
                "sample_id": row["conditioning_id"],
                "width": row["width"],
                "height": row["height"],
                "num_frames": row["num_frames"],
                "fps": row["fps"],
                "audio_sample_rate": row["audio_sample_rate"],
                "audio_channels": row["audio_channels"],
            }
            for row in rows
        ]
    }


def distribution(rows: list[dict[str, Any]], field_names: tuple[str, ...]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        key = "x".join(str(row[field]) for field in field_names)
        counts[key] += 1
    return dict(sorted(counts.items()))


def filtered_validation_summary(
    base_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    validation_membership_by_source: dict[str, int],
    *,
    base_root: Path,
    min_resolution_count: int,
    excluded_resolutions: list[str],
    excluded_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic validation receipt for a filtered derivation."""
    retained_ids = {str(row["conditioning_id"]) for row in rows}
    legacy_overlap_ids = [
        str(record_id)
        for record_id in base_summary.get("legacy_synth64_overlap_ids", [])
        if str(record_id) in retained_ids
    ]
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "seed": int(base_summary["seed"]),
        "rows": len(rows),
        "unique_conditioning_ids": len(retained_ids),
        "source_counts": dict(sorted(collections.Counter(row["source"] for row in rows).items())),
        "family_counts": dict(sorted(collections.Counter(row["family"] for row in rows).items())),
        "resolution_counts": distribution(rows, ("width", "height")),
        "frame_counts": distribution(rows, ("num_frames",)),
        "duration_band_counts": dict(
            sorted(collections.Counter(duration_band(row["num_frames"], row["fps"]) for row in rows).items())
        ),
        # Training remains anchored to the complete inherited holdout-ID set,
        # including nonrare cross-source variants of the four removed rows.
        "training_exclusions_by_source": dict(base_summary["training_exclusions_by_source"]),
        # This separately describes which frozen source rows still intersect
        # the filtered 60-ID validation manifest.
        "validation_membership_by_source": validation_membership_by_source,
        "filter": {
            "schema_version": FILTERED_DERIVATION_SCHEMA_VERSION,
            "base_root": str(base_root),
            "axis": "aggregate_frozen_resolution",
            "comparison": "count < min_resolution_count",
            "min_resolution_count": min_resolution_count,
            "excluded_resolutions": excluded_resolutions,
            "base_validation_rows": int(base_summary["rows"]),
            "excluded_validation_rows": len(excluded_rows),
        },
        "note": (
            "V3 removes validation rows at aggregate frozen resolutions below the threshold; "
            "their conditioning IDs remain excluded from training to preserve the base holdout boundary."
        ),
    }
    if "legacy_synth64_ids_path" in base_summary:
        summary["legacy_synth64_ids_path"] = base_summary["legacy_synth64_ids_path"]
        summary["legacy_synth64_overlap_count"] = len(legacy_overlap_ids)
        summary["legacy_synth64_overlap_ids"] = legacy_overlap_ids
    return summary


def conditioning_ids_sha256(ids: Iterable[str]) -> str:
    payload = "".join(f"{record_id}\n" for record_id in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def conditioning_keys_sha256(keys: Iterable[tuple[str, str]]) -> str:
    """Hash source-qualified IDs so intentional cross-source IDs stay distinct."""
    payload = "".join(f"{source}\t{record_id}\n" for source, record_id in sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


def resolution_key(row: dict[str, Any]) -> str:
    return f"{int(row['width'])}x{int(row['height'])}"


def aggregate_resolution_counts(rows_by_source: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for rows in rows_by_source.values():
        counts.update(resolution_key(row) for row in rows)
    return dict(sorted(counts.items()))


def load_frozen_rows(root: Path, source: str) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(root / source / "media" / "frozen.jsonl")]


def extension_compatible_config(base_config: dict[str, Any], config: dict[str, Any]) -> None:
    """Require every source and split input except output_root to remain fixed."""
    ignored = {"output_root"}
    base_contract = {name: value for name, value in base_config.items() if name not in ignored}
    extension_contract = {name: value for name, value in config.items() if name not in ignored}
    if extension_contract != base_contract:
        raise ValueError("extension config must equal the base config except for output_root")


def _indexed_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("conditioning_id") or "")
        if not record_id:
            raise ValueError(f"{path}: row has no conditioning_id")
        if record_id in indexed:
            raise ValueError(f"{path}: duplicate conditioning id {record_id}")
        indexed[record_id] = row
    return indexed


def _verify_sha256(path: Path, expected: Any, label: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{label}: invalid recorded sha256 {expected!r}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label}: sha256 {actual} != frozen {expected}")


def resolve_frozen_config(root: Path, root_manifest: dict[str, Any]) -> Path:
    """Resolve config provenance without trusting an unavailable login path."""
    expected = root_manifest.get("config_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"frozen config: invalid recorded sha256 {expected!r}")
    recorded_value = str(root_manifest.get("config_path", ""))
    if not recorded_value:
        raise ValueError("frozen manifest has no config_path")
    recorded = Path(recorded_value)
    candidates = [recorded, root / "CONFIG.snapshot.json", Path(__file__).with_name(recorded.name)]
    checked: list[str] = []
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        checked.append(str(candidate))
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate
    raise FileNotFoundError(
        "frozen config provenance is unavailable or has the wrong checksum; "
        f"expected sha256={expected}, checked={checked}"
    )


def _verify_frozen_video(row: dict[str, Any], videos_dir: Path) -> None:
    record_id = str(row["conditioning_id"])
    path = Path(str(row["raw_video_path"]))
    expected_path = videos_dir / f"{record_id}.mp4"
    if path != expected_path:
        raise ValueError(f"{row['source']}/{record_id}: raw video path {path} != canonical {expected_path}")
    with path.open("rb") as handle:
        file_stat = os.fstat(handle.fileno())
    if not stat_module.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{row['source']}/{record_id}: raw video is not a regular file: {path}")
    expected_stat = (int(row["video_size_bytes"]), int(row["video_mtime_ns"]))
    actual_stat = (file_stat.st_size, file_stat.st_mtime_ns)
    if actual_stat != expected_stat:
        raise ValueError(
            f"{row['source']}/{record_id}: frozen source changed: size/mtime {actual_stat} != {expected_stat}"
        )


def _verify_extension_contract(
    root: Path,
    root_manifest: dict[str, Any],
    config: dict[str, Any],
    seen_roots: set[Path],
) -> None:
    extension = root_manifest.get("extension")
    if extension is None:
        return
    if not isinstance(extension, dict) or extension.get("schema_version") != EXTENSION_SCHEMA_VERSION:
        raise ValueError("invalid frozen extension receipt")
    receipt_path = root / "EXTENSION_RECEIPT.json"
    _verify_sha256(receipt_path, root_manifest.get("extension_receipt_sha256"), "extension receipt")
    if json.loads(receipt_path.read_text()) != extension:
        raise ValueError("EXTENSION_RECEIPT.json does not match FROZEN_MANIFEST.json")
    base_root = Path(str(extension.get("base_root", ""))).resolve()
    if base_root == root:
        raise ValueError("extension base root cannot equal output root")
    base_manifest = verify_existing(base_root, emit_summary=False, _seen_roots=seen_roots)
    _verify_sha256(
        base_root / "FROZEN_MANIFEST.json",
        extension.get("base_frozen_manifest_sha256"),
        "extension base frozen manifest",
    )
    base_config_path = resolve_frozen_config(base_root, base_manifest)
    base_config = json.loads(base_config_path.read_text())
    extension_compatible_config(base_config, config)

    validation_pairs = (
        ("manifest.jsonl", "validation_manifest_sha256"),
        ("heldout64.json", "heldout64_sha256"),
    )
    for artifact_name, receipt_name in validation_pairs:
        base_path = base_root / "validation" / artifact_name
        current_path = root / "validation" / artifact_name
        expected_hash = extension.get(receipt_name)
        _verify_sha256(base_path, expected_hash, f"extension base {artifact_name}")
        _verify_sha256(current_path, expected_hash, f"extension {artifact_name}")
        if current_path.read_bytes() != base_path.read_bytes():
            raise ValueError(f"extension validation/{artifact_name} is not byte-identical to the base")

    extend_sources = extension.get("extend_sources")
    if not isinstance(extend_sources, list) or not extend_sources:
        raise ValueError("extension receipt must name at least one extended source")
    if len(extend_sources) != len(set(extend_sources)):
        raise ValueError("extension receipt has duplicate extended sources")
    configured_names = [str(spec["name"]) for spec in config["sources"]]
    if any(source not in configured_names for source in extend_sources):
        raise ValueError("extension receipt names a source outside the frozen config")
    source_receipts = extension.get("sources")
    if not isinstance(source_receipts, dict) or set(source_receipts) != set(configured_names):
        raise ValueError("extension receipt must cover every configured source")

    actual_totals = {
        "base_frozen_rows": 0,
        "combined_frozen_rows": 0,
        "base_training_rows": 0,
        "combined_training_rows": 0,
    }
    for source in configured_names:
        base_path = base_root / source / "media" / "frozen.jsonl"
        current_path = root / source / "media" / "frozen.jsonl"
        base_rows = _indexed_rows(base_path, load_frozen_rows(base_root, source))
        current_rows = _indexed_rows(current_path, load_frozen_rows(root, source))
        base_train_path = base_root / source / "media" / "train.jsonl"
        current_train_path = root / source / "media" / "train.jsonl"
        base_train = _indexed_rows(base_train_path, [row for _, row in iter_jsonl(base_train_path)])
        current_train = _indexed_rows(current_train_path, [row for _, row in iter_jsonl(current_train_path)])
        missing = set(base_rows) - set(current_rows)
        if missing:
            raise ValueError(f"{source}: extension dropped {len(missing)} base frozen rows")
        changed = [record_id for record_id, row in base_rows.items() if current_rows[record_id] != row]
        if changed:
            raise ValueError(f"{source}: extension changed {len(changed)} base frozen rows")
        added_ids = set(current_rows) - set(base_rows)
        if source not in extend_sources and added_ids:
            raise ValueError(f"{source}: non-extended source gained {len(added_ids)} rows")
        receipt = source_receipts[source]
        expected_receipt = {
            "base_frozen_rows": len(base_rows),
            "added_frozen_rows": len(added_ids),
            "combined_frozen_rows": len(current_rows),
            "base_training_rows": len(base_train),
            "added_training_rows": len(set(current_train) - set(base_train)),
            "combined_training_rows": len(current_train),
            "combined_prompt_rows": len(current_rows),
            "added_conditioning_ids_sha256": conditioning_ids_sha256(added_ids),
        }
        if receipt != expected_receipt:
            raise ValueError(f"{source}: extension receipt does not match the frozen row delta")
        for total_name in actual_totals:
            actual_totals[total_name] += expected_receipt[total_name]
    for total_name, expected_total in actual_totals.items():
        if int(extension.get(total_name, -1)) != expected_total:
            raise ValueError(f"extension receipt {total_name} does not match its source totals")


def load_filtered_derivation_receipt(
    root: Path,
    root_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    derivation = root_manifest.get("derivation")
    if derivation is None:
        return None
    if root_manifest.get("extension") is not None:
        raise ValueError("a frozen root cannot be both an extension and a filtered derivation")
    if not isinstance(derivation, dict) or derivation.get("schema_version") != FILTERED_DERIVATION_SCHEMA_VERSION:
        raise ValueError("invalid filtered derivation receipt")
    receipt_path = root / "DERIVATION_RECEIPT.json"
    _verify_sha256(
        receipt_path,
        root_manifest.get("derivation_receipt_sha256"),
        "filtered derivation receipt",
    )
    if json.loads(receipt_path.read_text()) != derivation:
        raise ValueError("DERIVATION_RECEIPT.json does not match FROZEN_MANIFEST.json")
    rule = derivation.get("filter")
    if not isinstance(rule, dict) or rule.get("axis") != "aggregate_frozen_resolution":
        raise ValueError("filtered derivation must use the aggregate frozen-resolution axis")
    if rule.get("comparison") != "count < min_resolution_count":
        raise ValueError("filtered derivation comparison is not the supported strict threshold")
    threshold = rule.get("min_resolution_count")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("filtered derivation min_resolution_count must be a positive integer")
    excluded = derivation.get("excluded_resolutions")
    if not isinstance(excluded, list) or excluded != sorted(set(excluded)):
        raise ValueError("filtered derivation excluded_resolutions must be a sorted unique list")
    for resolution in excluded:
        if not isinstance(resolution, str) or resolution.count("x") != 1:
            raise ValueError(f"invalid excluded resolution {resolution!r}")
        width, height = resolution.split("x")
        if not width.isdigit() or not height.isdigit() or int(width) <= 0 or int(height) <= 0:
            raise ValueError(f"invalid excluded resolution {resolution!r}")
    excluded_validation_ids = derivation.get("excluded_validation_conditioning_ids")
    if (
        not isinstance(excluded_validation_ids, list)
        or excluded_validation_ids != sorted(set(excluded_validation_ids))
        or any(not isinstance(record_id, str) or not record_id for record_id in excluded_validation_ids)
    ):
        raise ValueError("filtered derivation excluded validation IDs must be a sorted unique string list")
    payload_path = derivation.get("validation_payload_path")
    if (
        not isinstance(payload_path, str)
        or Path(payload_path).parts != ("validation", f"heldout{derivation.get('derived_validation_rows')}.json")
    ):
        raise ValueError("filtered derivation validation payload path does not match its row count")
    return derivation


def _verify_filtered_derivation_contract(
    root: Path,
    root_manifest: dict[str, Any],
    config: dict[str, Any],
    derivation: dict[str, Any],
    seen_roots: set[Path],
) -> None:
    base_root = Path(str(derivation.get("base_root", ""))).resolve()
    if base_root == root:
        raise ValueError("filtered derivation base root cannot equal output root")
    if Path(str(derivation.get("output_root", ""))).resolve() != root:
        raise ValueError("filtered derivation output_root does not match the frozen root")

    base_manifest = verify_existing(base_root, emit_summary=False, _seen_roots=seen_roots)
    base_frozen_sha256 = sha256_file(base_root / "FROZEN_MANIFEST.json")
    _verify_sha256(
        base_root / "FROZEN_MANIFEST.json",
        derivation.get("base_frozen_manifest_sha256"),
        "filtered derivation base frozen manifest",
    )
    _verify_sha256(
        base_root / "READY.json",
        derivation.get("base_ready_sha256"),
        "filtered derivation base READY",
    )
    base_ready = json.loads((base_root / "READY.json").read_text())
    if (
        base_ready.get("schema_version") != "minimax-h3-native-t2va-ready-v1"
        or base_ready.get("frozen_manifest_sha256") != base_frozen_sha256
        or int(base_ready.get("training_rows", -1)) != int(base_manifest["training_rows"])
    ):
        raise ValueError("filtered derivation base READY does not match its frozen manifest")

    base_config_path = resolve_frozen_config(base_root, base_manifest)
    base_config = json.loads(base_config_path.read_text())
    extension_compatible_config(base_config, config)
    if derivation.get("base_config_sha256") != sha256_file(base_config_path):
        raise ValueError("filtered derivation base config checksum does not match its frozen config")
    if derivation.get("config_sha256") != sha256_file(root / "CONFIG.snapshot.json"):
        raise ValueError("filtered derivation config checksum does not match CONFIG.snapshot.json")

    base_validation_artifacts = {
        "manifest.jsonl": "base_validation_manifest_sha256",
        "manifest.json": "base_validation_summary_sha256",
        "heldout64.json": "base_heldout64_sha256",
    }
    for artifact_name, receipt_name in base_validation_artifacts.items():
        _verify_sha256(
            base_root / "validation" / artifact_name,
            derivation.get(receipt_name),
            f"filtered derivation base {artifact_name}",
        )
    current_validation_artifacts = {
        root / "validation" / "manifest.jsonl": "validation_manifest_sha256",
        root / "validation" / "manifest.json": "validation_summary_sha256",
        root / str(derivation["validation_payload_path"]): "validation_payload_sha256",
    }
    for artifact_path, receipt_name in current_validation_artifacts.items():
        _verify_sha256(artifact_path, derivation.get(receipt_name), f"filtered derivation {artifact_path.name}")

    configured_names = [str(spec["name"]) for spec in config["sources"]]
    base_names = [str(summary["source"]) for summary in base_manifest["sources"]]
    if configured_names != base_names:
        raise ValueError("filtered derivation source order does not match the base")

    base_frozen_by_source: dict[str, list[dict[str, Any]]] = {}
    base_train_by_source: dict[str, list[dict[str, Any]]] = {}
    current_frozen_by_source: dict[str, list[dict[str, Any]]] = {}
    current_train_by_source: dict[str, list[dict[str, Any]]] = {}
    base_validation = [row for _, row in iter_jsonl(base_root / "validation" / "manifest.jsonl")]
    current_validation = [row for _, row in iter_jsonl(root / "validation" / "manifest.jsonl")]
    validation_ids = {str(row["conditioning_id"]) for row in current_validation}
    source_summaries = {str(summary["source"]): summary for summary in root_manifest["sources"]}
    base_summaries = {str(summary["source"]): summary for summary in base_manifest["sources"]}
    for source in configured_names:
        base_source = base_root / source
        current_source = root / source
        for relative in ("media/frozen.jsonl", "prompts/source.jsonl"):
            base_path = base_source / relative
            current_path = current_source / relative
            if current_path.read_bytes() != base_path.read_bytes():
                raise ValueError(f"{source}/{relative} is not byte-identical to the filtered derivation base")
        base_frozen_by_source[source] = load_frozen_rows(base_root, source)
        base_train_by_source[source] = [row for _, row in iter_jsonl(base_source / "media" / "train.jsonl")]
        current_frozen_by_source[source] = load_frozen_rows(root, source)
        current_train_by_source[source] = [row for _, row in iter_jsonl(current_source / "media" / "train.jsonl")]
        if current_frozen_by_source[source] != base_frozen_by_source[source]:
            raise ValueError(f"{source}: filtered derivation changed frozen row content")

    frozen_resolution_counts = aggregate_resolution_counts(current_frozen_by_source)
    threshold = int(derivation["filter"]["min_resolution_count"])
    excluded_resolutions = sorted(
        resolution
        for resolution, count in frozen_resolution_counts.items()
        if count < threshold
    )
    excluded_set = set(excluded_resolutions)
    excluded_frozen_keys = {
        (source, str(row["conditioning_id"]))
        for source, rows in current_frozen_by_source.items()
        for row in rows
        if resolution_key(row) in excluded_set
    }
    excluded_training_keys = {
        (source, str(row["conditioning_id"]))
        for source, rows in base_train_by_source.items()
        for row in rows
        if resolution_key(row) in excluded_set
    }
    expected_validation = [row for row in base_validation if resolution_key(row) not in excluded_set]
    excluded_validation = [row for row in base_validation if resolution_key(row) in excluded_set]
    excluded_validation_ids = sorted(str(row["conditioning_id"]) for row in excluded_validation)
    excluded_validation_keys = {
        (str(row["source"]), str(row["conditioning_id"]))
        for row in excluded_validation
    }
    if current_validation != expected_validation:
        raise ValueError("filtered validation manifest is not exactly the base manifest minus rare resolutions")
    if set(derivation["excluded_validation_conditioning_ids"]) != set(excluded_validation_ids):
        raise ValueError("filtered derivation excluded validation IDs do not match the base manifest filter")
    if int(root_manifest["validation_rows"]) != len(current_validation):
        raise ValueError("filtered derivation validation row count does not match FROZEN_MANIFEST.json")

    source_receipts: dict[str, dict[str, int]] = {}
    for source in configured_names:
        base_train = _indexed_rows(base_root / source / "media" / "train.jsonl", base_train_by_source[source])
        current_train = _indexed_rows(root / source / "media" / "train.jsonl", current_train_by_source[source])
        expected_current = {
            record_id: row
            for record_id, row in base_train.items()
            if resolution_key(row) not in excluded_set
        }
        if current_train != expected_current:
            raise ValueError(f"{source}: derived train rows are not exactly the filtered base training rows")

        base_worklist = json.loads((base_root / source / "work" / "worklist.json").read_text())
        current_worklist = json.loads((root / source / "work" / "worklist.json").read_text())
        base_chunks: dict[tuple[int, int, int, tuple[str, ...]], dict[str, Any]] = {}
        for chunk in base_worklist["chunks"]:
            signature = worklist_chunk_signature(chunk)
            if signature in base_chunks:
                raise ValueError(f"{source}: base worklist has duplicate chunk signature")
            base_chunks[signature] = chunk
        for current_chunk in current_worklist["chunks"]:
            signature = worklist_chunk_signature(current_chunk)
            base_chunk = base_chunks.get(signature)
            if base_chunk is None:
                raise ValueError(f"{source}: retained worklist chunk has no exact base chunk")
            base_done_path = base_root / source / "done" / f"{base_chunk['chunk_id']}.json"
            current_done_path = root / source / "done" / f"{current_chunk['chunk_id']}.json"
            base_done = json.loads(base_done_path.read_text())
            current_done = json.loads(current_done_path.read_text())
            base_parquet = Path(str(base_done.get("parquet", ""))).resolve()
            current_parquet = Path(str(current_done.get("parquet", ""))).resolve()
            if not base_parquet.is_file() or not current_parquet.is_file():
                raise FileNotFoundError(f"missing base/derived parquet for {source}/{current_chunk['chunk_id']}")
            if not os.path.samefile(base_parquet, current_parquet):
                raise ValueError(f"{current_parquet}: retained parquet is not hardlinked to {base_parquet}")

        frozen_ids = {str(row["conditioning_id"]) for row in current_frozen_by_source[source]}
        validation_exclusions = len(frozen_ids & validation_ids)
        base_validation_exclusions = int(base_summaries[source]["validation_exclusions"])
        removed_validation_id_exclusions = len(frozen_ids & set(excluded_validation_ids))
        filter_exclusions = len(base_train) - len(expected_current)
        source_receipts[source] = {
            "frozen_rows": len(current_frozen_by_source[source]),
            "base_training_rows": len(base_train),
            "derived_training_rows": len(current_train),
            "base_validation_exclusions": base_validation_exclusions,
            "validation_exclusions": validation_exclusions,
            "removed_validation_id_exclusions": removed_validation_id_exclusions,
            "filter_exclusions": filter_exclusions,
        }
        summary = source_summaries[source]
        if int(summary.get("filter_exclusions", -1)) != filter_exclusions:
            raise ValueError(f"{source}: source summary filter_exclusions does not match derivation")
        if summary.get("derivation") != source_receipts[source]:
            raise ValueError(f"{source}: source summary derivation receipt does not match root receipt")
        if int(base_summaries[source]["frozen_rows"]) != len(current_frozen_by_source[source]):
            raise ValueError(f"{source}: base frozen row count changed in derivation")

    base_validation_summary = json.loads((base_root / "validation" / "manifest.json").read_text())
    expected_validation_summary = filtered_validation_summary(
        base_validation_summary,
        current_validation,
        {source: receipt["validation_exclusions"] for source, receipt in source_receipts.items()},
        base_root=base_root,
        min_resolution_count=threshold,
        excluded_resolutions=excluded_resolutions,
        excluded_rows=excluded_validation,
    )
    actual_validation_summary = json.loads((root / "validation" / "manifest.json").read_text())
    if actual_validation_summary != expected_validation_summary:
        raise ValueError("filtered validation summary does not exactly derive from the base validation manifest")
    validation_payload_path = root / str(derivation["validation_payload_path"])
    if json.loads(validation_payload_path.read_text()) != heldout_payload(current_validation):
        raise ValueError("filtered validation payload does not exactly derive from its manifest")

    created_utc = derivation.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError("filtered derivation receipt has no creation timestamp")
    expected_receipt = {
        "schema_version": FILTERED_DERIVATION_SCHEMA_VERSION,
        "created_utc": created_utc,
        "base_root": str(base_root),
        "output_root": str(root),
        "base_ready_sha256": sha256_file(base_root / "READY.json"),
        "base_frozen_manifest_sha256": base_frozen_sha256,
        "base_config_sha256": sha256_file(base_config_path),
        "config_sha256": sha256_file(root / "CONFIG.snapshot.json"),
        "filter": {
            "axis": "aggregate_frozen_resolution",
            "comparison": "count < min_resolution_count",
            "min_resolution_count": threshold,
        },
        "frozen_resolution_counts": frozen_resolution_counts,
        "excluded_resolutions": excluded_resolutions,
        "excluded_frozen_rows": len(excluded_frozen_keys),
        "excluded_frozen_keys_sha256": conditioning_keys_sha256(excluded_frozen_keys),
        "excluded_training_rows": len(excluded_training_keys),
        "excluded_training_keys_sha256": conditioning_keys_sha256(excluded_training_keys),
        "base_frozen_rows": int(base_manifest["frozen_rows"]),
        "base_training_rows": int(base_manifest["training_rows"]),
        "derived_training_rows": sum(len(rows) for rows in current_train_by_source.values()),
        "training_holdout_policy": "preserve_base_validation_conditioning_ids",
        "base_validation_rows": len(base_validation),
        "derived_validation_rows": len(current_validation),
        "excluded_validation_rows": len(excluded_validation),
        "base_validation_conditioning_ids_sha256": conditioning_ids_sha256(
            str(row["conditioning_id"]) for row in base_validation
        ),
        "validation_conditioning_ids_sha256": conditioning_ids_sha256(validation_ids),
        "excluded_validation_conditioning_ids": excluded_validation_ids,
        "excluded_validation_keys_sha256": conditioning_keys_sha256(excluded_validation_keys),
        "base_validation_manifest_sha256": sha256_file(base_root / "validation" / "manifest.jsonl"),
        "base_validation_summary_sha256": sha256_file(base_root / "validation" / "manifest.json"),
        "base_heldout64_sha256": sha256_file(base_root / "validation" / "heldout64.json"),
        "validation_manifest_sha256": sha256_file(root / "validation" / "manifest.jsonl"),
        "validation_summary_sha256": sha256_file(root / "validation" / "manifest.json"),
        "validation_payload_path": f"validation/heldout{len(current_validation)}.json",
        "validation_payload_sha256": sha256_file(validation_payload_path),
        "sources": source_receipts,
    }
    if derivation != expected_receipt:
        raise ValueError("filtered derivation receipt does not exactly match the base, rule, and derived rows")


def verify_existing(
    root: Path,
    *,
    emit_summary: bool = True,
    _seen_roots: set[Path] | None = None,
) -> dict[str, Any]:
    """Verify the complete immutable freeze contract without importing FastVideo.

    The frozen root manifest is the trust anchor. Every artifact hash and
    provenance edge recorded by the freezer is checked before callers may use
    the training rows or publish derived data.
    """
    root = root.resolve()
    seen_roots = set() if _seen_roots is None else _seen_roots
    if root in seen_roots:
        raise ValueError(f"cyclic frozen extension ancestry at {root}")
    seen_roots.add(root)
    root_manifest_path = root / "FROZEN_MANIFEST.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    if root_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected frozen manifest schema")

    config_path = resolve_frozen_config(root, root_manifest)
    config = json.loads(config_path.read_text())
    configured_specs = config.get("sources")
    if not isinstance(configured_specs, list):
        raise ValueError("frozen config sources must be a list")
    configured_names = [str(spec["name"]) for spec in configured_specs]
    source_summaries = root_manifest.get("sources")
    if not isinstance(source_summaries, list):
        raise ValueError("frozen manifest sources must be a list")
    source_names = [str(summary.get("source") or "") for summary in source_summaries]
    if not all(source_names) or len(source_names) != len(set(source_names)):
        raise ValueError("frozen manifest source names must be non-empty and unique")
    if source_names != configured_names:
        raise ValueError(f"frozen manifest sources {source_names} != config sources {configured_names}")
    if int(root_manifest.get("seed", -1)) != int(config["snapshot_seed"]):
        raise ValueError("frozen manifest seed does not match frozen config")
    derivation = load_filtered_derivation_receipt(root, root_manifest)
    excluded_resolutions = (
        set(str(value) for value in derivation["excluded_resolutions"])
        if derivation is not None
        else set()
    )

    validation_manifest_path = root / "validation" / "manifest.jsonl"
    validation_payload_relative = (
        str(derivation["validation_payload_path"])
        if derivation is not None
        else "validation/heldout64.json"
    )
    if derivation is not None and root_manifest.get("heldout_path") != validation_payload_relative:
        raise ValueError("filtered root heldout_path does not match its derivation receipt")
    heldout_path = root / validation_payload_relative
    _verify_sha256(
        validation_manifest_path,
        root_manifest.get("validation_manifest_sha256"),
        "validation manifest",
    )
    heldout_checksum = (
        root_manifest.get("heldout_sha256")
        if derivation is not None
        else root_manifest.get("heldout64_sha256")
    )
    _verify_sha256(heldout_path, heldout_checksum, Path(validation_payload_relative).name)
    validation = [row for _, row in iter_jsonl(validation_manifest_path)]
    validation_by_id = _indexed_rows(validation_manifest_path, validation)
    validation_ids = set(validation_by_id)
    expected_validation_rows = int(root_manifest.get("validation_rows", -1))
    if derivation is None and expected_validation_rows != 64:
        raise ValueError(f"base/extension validation split must contain 64 rows, got {expected_validation_rows}")
    if len(validation) != expected_validation_rows or len(validation_ids) != expected_validation_rows:
        raise ValueError(
            f"validation split must have {expected_validation_rows} unique ids, "
            f"got {len(validation)}/{len(validation_ids)}"
        )
    inherited_validation_ids = validation_ids | (
        set(str(record_id) for record_id in derivation["excluded_validation_conditioning_ids"])
        if derivation is not None
        else set()
    )

    config_by_name = {str(spec["name"]): spec for spec in configured_specs}
    frozen_by_source_and_id: dict[tuple[str, str], dict[str, Any]] = {}
    total_frozen = 0
    total_training = 0
    validation_exclusions: dict[str, int] = {}
    for summary in source_summaries:
        source = str(summary["source"])
        source_root = (root / source).resolve()
        if source_root.parent != root:
            raise ValueError(f"frozen source must be a direct child of root: {source!r}")
        stored_summary = json.loads((source_root / "MANIFEST.source.json").read_text())
        if stored_summary != summary:
            raise ValueError(f"{source}: MANIFEST.source.json does not match FROZEN_MANIFEST.json")

        artifact_paths = {
            "source.jsonl": source_root / "prompts" / "source.jsonl",
            "frozen.jsonl": source_root / "media" / "frozen.jsonl",
            "train.jsonl": source_root / "media" / "train.jsonl",
            "worklist.json": source_root / "work" / "worklist.json",
        }
        hashes = summary.get("artifacts_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(artifact_paths):
            raise ValueError(f"{source}: frozen artifact checksum set is incomplete")
        for artifact_name, artifact_path in artifact_paths.items():
            _verify_sha256(artifact_path, hashes[artifact_name], f"{source}/{artifact_name}")
        expected_source_checksum = f"{hashes['source.jsonl']}  source.jsonl\n"
        if (source_root / "prompts" / "SOURCE.sha256").read_text() != expected_source_checksum:
            raise ValueError(f"{source}: SOURCE.sha256 does not match source.jsonl")

        frozen_path = artifact_paths["frozen.jsonl"]
        train_path = artifact_paths["train.jsonl"]
        frozen = [row for _, row in iter_jsonl(frozen_path)]
        training = [row for _, row in iter_jsonl(train_path)]
        frozen_by_id = _indexed_rows(frozen_path, frozen)
        train_by_id = _indexed_rows(train_path, training)
        if len(frozen) != int(summary["frozen_rows"]) or len(training) != int(summary["training_rows"]):
            raise ValueError(f"{source}: row-count mismatch")
        validation_ids_in_source = set(frozen_by_id) & validation_ids
        inherited_validation_ids_in_source = set(frozen_by_id) & inherited_validation_ids
        filtered_ids = {
            record_id
            for record_id, row in frozen_by_id.items()
            if resolution_key(row) in excluded_resolutions
        } - inherited_validation_ids_in_source
        expected_training_ids = set(frozen_by_id) - inherited_validation_ids_in_source - filtered_ids
        if set(train_by_id) != expected_training_ids:
            if derivation is None:
                raise ValueError(f"{source}: train.jsonl is not exactly frozen.jsonl minus validation ids")
            raise ValueError(
                f"{source}: train.jsonl is not exactly frozen.jsonl minus validation ids and filtered resolutions"
            )
        if any(train_by_id[record_id] != frozen_by_id[record_id] for record_id in train_by_id):
            raise ValueError(f"{source}: train.jsonl rows differ from their frozen rows")

        spec = config_by_name[source]
        videos_dir = Path(str(summary["videos_dir"]))
        expected_provenance = {
            "videos_dir": str(Path(spec["videos_dir"]).resolve()),
            "status_jsonl": str(Path(spec["status_jsonl"]).resolve()),
            "prompts_jsonl": str(Path(spec["prompts_jsonl"]).resolve()),
        }
        actual_provenance = {name: str(summary.get(name)) for name in expected_provenance}
        if actual_provenance != expected_provenance:
            raise ValueError(f"{source}: source-path provenance does not match frozen config")
        for row in frozen:
            record_id = str(row["conditioning_id"])
            if row.get("schema_version") != SCHEMA_VERSION or row.get("source") != source:
                raise ValueError(f"{source}/{record_id}: invalid frozen row provenance")
            if row.get("family") != spec["family"]:
                raise ValueError(f"{source}/{record_id}: family does not match frozen config")
            _verify_frozen_video(row, videos_dir)
            frozen_by_source_and_id[(source, record_id)] = row

        prompt_rows = [row for _, row in iter_jsonl(artifact_paths["source.jsonl"])]
        expected_prompt_rows = [
            {"conditioning_id": row["conditioning_id"], "prompt": row["prompt"]}
            for row in frozen
        ]
        if prompt_rows != expected_prompt_rows:
            raise ValueError(f"{source}: prompts/source.jsonl does not exactly match frozen rows")

        worklist = json.loads(artifact_paths["worklist.json"].read_text())
        expected_worklist = build_worklist(training, int(worklist["chunk_size"]))
        expected_worklist.update({
            "source": source,
            "train_manifest": str(root / source / "media" / "train.jsonl"),
            "set_root": str(root / source),
        })
        if worklist != expected_worklist:
            raise ValueError(f"{source}: worklist does not exactly derive from train.jsonl")
        if summary.get("shape_counts") != distribution(frozen, ("width", "height", "num_frames")):
            raise ValueError(f"{source}: frozen shape counts do not match frozen.jsonl")
        exclusions = len(validation_ids_in_source)
        if int(summary["validation_exclusions"]) != exclusions:
            raise ValueError(f"{source}: validation exclusion count mismatch")
        if derivation is not None and int(summary.get("filter_exclusions", -1)) != len(filtered_ids):
            raise ValueError(f"{source}: filtered-resolution exclusion count mismatch")
        if (
            derivation is not None
            and int(summary.get("base_validation_exclusions", -1)) != len(inherited_validation_ids_in_source)
        ):
            raise ValueError(f"{source}: inherited base-validation exclusion count mismatch")
        validation_exclusions[source] = exclusions
        total_frozen += len(frozen)
        total_training += len(training)

    if total_frozen != int(root_manifest.get("frozen_rows", -1)):
        raise ValueError("FROZEN_MANIFEST.json frozen row count mismatch")
    if total_training != int(root_manifest.get("training_rows", -1)):
        raise ValueError("FROZEN_MANIFEST.json training row count mismatch")

    for record_id, validation_row_payload in validation_by_id.items():
        source = str(validation_row_payload.get("source") or "")
        frozen_row = frozen_by_source_and_id.get((source, record_id))
        if frozen_row is None:
            raise ValueError(f"validation row {source}/{record_id} has no matching frozen row")
        if validation_row_payload != validation_row(frozen_row):
            raise ValueError(f"validation row {source}/{record_id} differs from its frozen row")

    actual_heldout = json.loads(heldout_path.read_text())
    expected_heldout = heldout_payload(validation)
    if actual_heldout != expected_heldout:
        raise ValueError(f"{heldout_path.name} does not exactly derive from the validation manifest")
    expected_links = {
        f"{row['source']}__{row['conditioning_id']}.mp4": Path(row["raw_video_path"])
        for row in validation
    }
    videos_root = root / "validation" / "videos"
    actual_links = {path.name: path for path in videos_root.iterdir()}
    if set(actual_links) != set(expected_links):
        raise ValueError("validation/videos does not exactly match the validation manifest")
    for name, target in expected_links.items():
        link = actual_links[name]
        if not link.is_symlink() or link.resolve() != target:
            raise ValueError(f"{link}: validation video link does not target frozen raw video")

    validation_summary = json.loads((root / "validation" / "manifest.json").read_text())
    expected_training_exclusions = (
        {
            str(summary["source"]): int(summary["base_validation_exclusions"])
            for summary in source_summaries
        }
        if derivation is not None
        else validation_exclusions
    )
    expected_validation_summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "seed": int(root_manifest["seed"]),
        "rows": expected_validation_rows,
        "unique_conditioning_ids": expected_validation_rows,
        "source_counts": dict(sorted(collections.Counter(row["source"] for row in validation).items())),
        "family_counts": dict(sorted(collections.Counter(row["family"] for row in validation).items())),
        "resolution_counts": distribution(validation, ("width", "height")),
        "frame_counts": distribution(validation, ("num_frames",)),
        "duration_band_counts": dict(
            sorted(collections.Counter(duration_band(row["num_frames"], row["fps"]) for row in validation).items())
        ),
        "training_exclusions_by_source": expected_training_exclusions,
    }
    if derivation is not None:
        expected_validation_summary["validation_membership_by_source"] = validation_exclusions
    for name, expected in expected_validation_summary.items():
        if validation_summary.get(name) != expected:
            raise ValueError(f"validation/manifest.json field {name} does not match frozen rows")

    _verify_extension_contract(root, root_manifest, config, seen_roots)
    if derivation is not None:
        _verify_filtered_derivation_contract(root, root_manifest, config, derivation, seen_roots)
    seen_roots.remove(root)
    if emit_summary:
        print(
            f"verified {root}: {expected_validation_rows} unique validation ids, "
            "immutable artifacts, and raw media stats"
        )
    return root_manifest


def freeze_extension(
    args: argparse.Namespace,
    config: dict[str, Any],
    root: Path,
) -> None:
    """Freeze a combined tree while retaining the verified base rows and split."""
    base_root = args.extend_existing.resolve()
    if base_root == root:
        raise ValueError("--extend-existing must differ from the output root")
    base_manifest = verify_existing(base_root, emit_summary=False)
    base_config_path = resolve_frozen_config(base_root, base_manifest)
    base_config = json.loads(base_config_path.read_text())
    extension_compatible_config(base_config, config)

    configured_names = [str(spec["name"]) for spec in config["sources"]]
    extend_sources = list(dict.fromkeys(str(name) for name in args.extend_source))
    if not extend_sources:
        raise ValueError("--extend-existing requires at least one --extend-source")
    unknown_sources = sorted(set(extend_sources) - set(configured_names))
    if unknown_sources:
        raise ValueError(f"unknown --extend-source values: {unknown_sources}")
    base_summaries = {str(summary["source"]): summary for summary in base_manifest["sources"]}
    if set(base_summaries) != set(configured_names):
        raise ValueError("base frozen sources do not match extension config")
    base_chunk_sizes = {
        int(json.loads((base_root / source / "work" / "worklist.json").read_text())["chunk_size"])
        for source in configured_names
    }
    if base_chunk_sizes != {args.chunk_size}:
        raise ValueError(
            f"--chunk-size {args.chunk_size} must equal the base freeze chunk size {sorted(base_chunk_sizes)}"
        )

    validation_manifest_path = base_root / "validation" / "manifest.jsonl"
    validation_rows = [row for _, row in iter_jsonl(validation_manifest_path)]
    validation_ids = {str(row["conditioning_id"]) for row in validation_rows}
    if len(validation_rows) != 64 or len(validation_ids) != 64:
        raise ValueError("base validation manifest must contain 64 unique conditioning ids")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    training_rows: dict[str, list[dict[str, Any]]] = {}
    all_stats: dict[str, dict[str, Any]] = {}
    source_receipts: dict[str, dict[str, Any]] = {}
    extension_specs = {str(spec["name"]): spec for spec in config["sources"]}
    for source in configured_names:
        base_rows = load_frozen_rows(base_root, source)
        base_by_id = _indexed_rows(base_root / source / "media" / "frozen.jsonl", base_rows)
        added_by_id: dict[str, dict[str, Any]] = {}
        if source in extend_sources:
            live_rows, live_stats = source_inventory(extension_specs[source])
            live_by_id = _indexed_rows(Path(live_stats["status_jsonl"]), live_rows)
            added_by_id = {record_id: row for record_id, row in live_by_id.items() if record_id not in base_by_id}
            stats = dict(live_stats)
        else:
            derived_fields = {
                "artifacts_sha256",
                "training_rows",
                "validation_exclusions",
                "shape_counts",
                "extension",
            }
            stats = {
                name: value
                for name, value in base_summaries[source].items()
                if name not in derived_fields
            }
        combined_by_id = {**base_by_id, **added_by_id}
        combined = [combined_by_id[record_id] for record_id in sorted(combined_by_id)]
        train = [row for row in combined if row["conditioning_id"] not in validation_ids]
        base_train_path = base_root / source / "media" / "train.jsonl"
        base_train_ids = {str(row["conditioning_id"]) for _, row in iter_jsonl(base_train_path)}
        combined_train_ids = {str(row["conditioning_id"]) for row in train}
        if not base_train_ids <= combined_train_ids:
            raise ValueError(f"{source}: extension would drop base training rows")
        stats["frozen_rows"] = len(combined)
        all_rows[source] = combined
        training_rows[source] = train
        all_stats[source] = stats
        source_receipts[source] = {
            "base_frozen_rows": len(base_rows),
            "added_frozen_rows": len(added_by_id),
            "combined_frozen_rows": len(combined),
            "base_training_rows": len(base_train_ids),
            "added_training_rows": len(combined_train_ids - base_train_ids),
            "combined_training_rows": len(train),
            "combined_prompt_rows": len(combined),
            "added_conditioning_ids_sha256": conditioning_ids_sha256(added_by_id),
        }
        print(
            f"{source}: base={len(base_rows)} added={len(added_by_id)} "
            f"combined={len(combined)} training={len(train)}"
        )

    base_validation_summary = json.loads((base_root / "validation" / "manifest.json").read_text())
    validation_exclusions = {
        source: len(all_rows[source]) - len(training_rows[source])
        for source in configured_names
    }
    validation_summary = {
        **base_validation_summary,
        "training_exclusions_by_source": validation_exclusions,
        "extension": {
            "schema_version": EXTENSION_SCHEMA_VERSION,
            "base_root": str(base_root),
            "extend_sources": extend_sources,
            "note": "heldout64.json and validation/manifest.jsonl are byte-identical to the verified base freeze",
        },
    }
    extension_receipt = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "base_root": str(base_root),
        "base_frozen_manifest_sha256": sha256_file(base_root / "FROZEN_MANIFEST.json"),
        "validation_manifest_sha256": sha256_file(base_root / "validation" / "manifest.jsonl"),
        "heldout64_sha256": sha256_file(base_root / "validation" / "heldout64.json"),
        "extend_sources": extend_sources,
        "sources": source_receipts,
        "base_frozen_rows": int(base_manifest["frozen_rows"]),
        "combined_frozen_rows": sum(len(rows) for rows in all_rows.values()),
        "base_training_rows": int(base_manifest["training_rows"]),
        "combined_training_rows": sum(len(rows) for rows in training_rows.values()),
    }
    print(json.dumps(extension_receipt, indent=2, sort_keys=True))
    if args.dry_run:
        return

    if root.exists():
        raise FileExistsError(f"freeze root already exists at {root}; use --verify-existing instead of overwriting")
    root.parent.mkdir(parents=True, exist_ok=True)
    # Assemble the complete tree as a sibling and publish it with one rename.
    # A killed process can leave only an unreferenced sibling staging tree;
    # the canonical root remains absent and a retry can safely start anew.
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.freeze-extension-", dir=root.parent))
    try:
        source_summaries: list[dict[str, Any]] = []
        for source in configured_names:
            source_root = stage / source
            frozen = all_rows[source]
            train = training_rows[source]
            write_jsonl(source_root / "media" / "frozen.jsonl", frozen)
            write_jsonl(source_root / "media" / "train.jsonl", train)
            write_jsonl(
                source_root / "prompts" / "source.jsonl",
                ({"conditioning_id": row["conditioning_id"], "prompt": row["prompt"]} for row in frozen),
            )
            worklist = build_worklist(train, args.chunk_size)
            worklist.update({
                "source": source,
                "train_manifest": str(root / source / "media" / "train.jsonl"),
                "set_root": str(root / source),
            })
            (source_root / "work").mkdir(parents=True, exist_ok=True)
            (source_root / "work" / "worklist.json").write_text(
                json.dumps(worklist, indent=2, sort_keys=True) + "\n"
            )
            hashes = {
                "source.jsonl": sha256_file(source_root / "prompts" / "source.jsonl"),
                "frozen.jsonl": sha256_file(source_root / "media" / "frozen.jsonl"),
                "train.jsonl": sha256_file(source_root / "media" / "train.jsonl"),
                "worklist.json": sha256_file(source_root / "work" / "worklist.json"),
            }
            (source_root / "prompts" / "SOURCE.sha256").write_text(hashes["source.jsonl"] + "  source.jsonl\n")
            source_summary = {
                **all_stats[source],
                "training_rows": len(train),
                "validation_exclusions": validation_exclusions[source],
                "artifacts_sha256": hashes,
                "shape_counts": distribution(frozen, ("width", "height", "num_frames")),
                "extension": source_receipts[source],
            }
            (source_root / "MANIFEST.source.json").write_text(
                json.dumps(source_summary, indent=2, sort_keys=True) + "\n"
            )
            source_summaries.append(source_summary)

        validation_root = stage / "validation"
        validation_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_root / "validation" / "manifest.jsonl", validation_root / "manifest.jsonl")
        shutil.copy2(base_root / "validation" / "heldout64.json", validation_root / "heldout64.json")
        (validation_root / "manifest.json").write_text(
            json.dumps(validation_summary, indent=2, sort_keys=True) + "\n"
        )
        videos_root = validation_root / "videos"
        videos_root.mkdir()
        for row in validation_rows:
            (videos_root / f"{row['source']}__{row['conditioning_id']}.mp4").symlink_to(row["raw_video_path"])

        (stage / "EXTENSION_RECEIPT.json").write_text(
            json.dumps(extension_receipt, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(args.config, stage / "CONFIG.snapshot.json")
        frozen_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_path": str(root / "CONFIG.snapshot.json"),
            "config_source_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "seed": int(config["snapshot_seed"]),
            "sources": source_summaries,
            "frozen_rows": sum(len(rows) for rows in all_rows.values()),
            "training_rows": sum(len(rows) for rows in training_rows.values()),
            "validation_rows": 64,
            "validation_manifest_sha256": sha256_file(validation_root / "manifest.jsonl"),
            "heldout64_sha256": sha256_file(validation_root / "heldout64.json"),
            "extension": extension_receipt,
            "extension_receipt_sha256": sha256_file(stage / "EXTENSION_RECEIPT.json"),
            "ready": False,
            "ready_policy": (
                "READY.json is created only by finalize_dataset.py after every training row is encoded and validated."
            ),
        }
        (stage / "FROZEN_MANIFEST.json").write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")

        os.replace(stage, root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_existing(root)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    root = (args.output_root or Path(config["output_root"])).resolve()
    if args.verify_existing:
        if args.extend_source:
            raise ValueError("--extend-source is only valid with --extend-existing")
        verify_existing(root)
        return
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.extend_existing is not None:
        freeze_extension(args, config, root)
        return
    if args.extend_source:
        raise ValueError("--extend-source requires --extend-existing")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_stats: dict[str, dict[str, Any]] = {}
    for spec in config["sources"]:
        rows, stats = source_inventory(spec)
        all_rows[spec["name"]] = rows
        all_stats[spec["name"]] = stats
        print(
            f"{spec['name']}: completed={stats['completed_status_ids']} "
            f"canonical_mp4={stats['canonical_mp4s']} joined={len(rows)}"
        )

    seed = int(config["snapshot_seed"])
    quotas = {str(name): int(value) for name, value in config["validation_quotas"].items()}
    if sum(quotas.values()) != 64:
        raise ValueError(f"validation quotas must sum to 64, got {sum(quotas.values())}")
    # Select the duplicate-bearing low-resolution NuVA source first. This
    # guarantees its quota, then prevents the same conditioning id from being
    # selected from the high-resolution variant.
    configured_names = [str(spec["name"]) for spec in config["sources"]]
    selection_order = [
        "h3_t2av_video_nuva_10k_mixed_res_len",
        *[name for name in configured_names if name != "h3_t2av_video_nuva_10k_mixed_res_len"],
    ]
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for source_index, name in enumerate(selection_order):
        selected.extend(select_stratified(all_rows[name], quotas[name], seed + source_index * 1009, selected_ids))
    selected = sorted(selected, key=lambda row: (configured_names.index(row["source"]), row["conditioning_id"]))
    if len(selected) != 64 or len(selected_ids) != 64:
        raise AssertionError("validation selection did not produce exactly 64 unique ids")
    family_counts = collections.Counter(row["family"] for row in selected)
    if family_counts != {"nuva": 32, "vidprom": 32}:
        raise ValueError(f"validation family balance is not 32/32: {dict(family_counts)}")

    legacy_path = Path(config["legacy_validation_ids"])
    legacy_ids = {line.strip() for line in legacy_path.read_text().splitlines() if line.strip()} if legacy_path.is_file() else set()
    overlap = sorted(selected_ids & legacy_ids)

    training_rows: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, int] = {}
    for name, rows in all_rows.items():
        kept = [row for row in rows if row["conditioning_id"] not in selected_ids]
        training_rows[name] = kept
        exclusions[name] = len(rows) - len(kept)

    source_counts = collections.Counter(row["source"] for row in selected)
    validation_summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "seed": seed,
        "rows": len(selected),
        "unique_conditioning_ids": len(selected_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "resolution_counts": distribution(selected, ("width", "height")),
        "frame_counts": distribution(selected, ("num_frames",)),
        "duration_band_counts": dict(sorted(collections.Counter(
            duration_band(row["num_frames"], row["fps"]) for row in selected).items())),
        "training_exclusions_by_source": exclusions,
        "legacy_synth64_ids_path": str(legacy_path),
        "legacy_synth64_overlap_count": len(overlap),
        "legacy_synth64_overlap_ids": overlap,
        "note": "This seeded mixed split replaces synth64 for v10. Only these new ids and their cross-source variants are excluded.",
    }
    print(json.dumps(validation_summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    protected = [root / "FROZEN_MANIFEST.json", root / "validation" / "manifest.jsonl"]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"freeze already exists under {root}; use --verify-existing instead of overwriting")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".freeze-", dir=root))
    try:
        source_summaries: list[dict[str, Any]] = []
        for spec in config["sources"]:
            name = str(spec["name"])
            source_root = stage / name
            frozen = all_rows[name]
            train = training_rows[name]
            write_jsonl(source_root / "media" / "frozen.jsonl", frozen)
            write_jsonl(source_root / "media" / "train.jsonl", train)
            write_jsonl(
                source_root / "prompts" / "source.jsonl",
                ({"conditioning_id": row["conditioning_id"], "prompt": row["prompt"]} for row in frozen),
            )
            worklist = build_worklist(train, args.chunk_size)
            worklist["source"] = name
            worklist["train_manifest"] = str(root / name / "media" / "train.jsonl")
            worklist["set_root"] = str(root / name)
            (source_root / "work").mkdir(parents=True, exist_ok=True)
            (source_root / "work" / "worklist.json").write_text(json.dumps(worklist, indent=2, sort_keys=True) + "\n")
            hashes = {
                "source.jsonl": sha256_file(source_root / "prompts" / "source.jsonl"),
                "frozen.jsonl": sha256_file(source_root / "media" / "frozen.jsonl"),
                "train.jsonl": sha256_file(source_root / "media" / "train.jsonl"),
                "worklist.json": sha256_file(source_root / "work" / "worklist.json"),
            }
            (source_root / "prompts" / "SOURCE.sha256").write_text(hashes["source.jsonl"] + "  source.jsonl\n")
            source_summary = {
                **all_stats[name],
                "training_rows": len(train),
                "validation_exclusions": exclusions[name],
                "artifacts_sha256": hashes,
                "shape_counts": distribution(frozen, ("width", "height", "num_frames")),
            }
            (source_root / "MANIFEST.source.json").write_text(
                json.dumps(source_summary, indent=2, sort_keys=True) + "\n"
            )
            source_summaries.append(source_summary)

        validation_rows = [validation_row(row) for row in selected]
        write_jsonl(stage / "validation" / "manifest.jsonl", validation_rows)
        (stage / "validation" / "manifest.json").write_text(
            json.dumps(validation_summary, indent=2, sort_keys=True) + "\n"
        )
        (stage / "validation" / "heldout64.json").write_text(
            json.dumps(heldout_payload(validation_rows), indent=2, ensure_ascii=False) + "\n"
        )
        videos_dir = stage / "validation" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        for row in validation_rows:
            link = videos_dir / f"{row['source']}__{row['conditioning_id']}.mp4"
            link.symlink_to(row["raw_video_path"])

        frozen_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "seed": seed,
            "sources": source_summaries,
            "frozen_rows": sum(len(rows) for rows in all_rows.values()),
            "training_rows": sum(len(rows) for rows in training_rows.values()),
            "validation_rows": 64,
            "validation_manifest_sha256": sha256_file(stage / "validation" / "manifest.jsonl"),
            "heldout64_sha256": sha256_file(stage / "validation" / "heldout64.json"),
            "ready": False,
            "ready_policy": "READY.json is created only by finalize_dataset.py after every training row is encoded and validated.",
        }
        (stage / "FROZEN_MANIFEST.json").write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")

        for child in sorted(stage.iterdir()):
            destination = root / child.name
            if child.name in configured_names:
                destination.mkdir(parents=True, exist_ok=True)
                for artifact in sorted(child.iterdir()):
                    target = destination / artifact.name
                    if target.exists() and any(target.iterdir()):
                        raise FileExistsError(f"refusing to replace non-empty {target}")
                    if target.exists():
                        target.rmdir()
                    os.replace(artifact, target)
                child.rmdir()
            else:
                os.replace(child, destination)
        stage.rmdir()
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_existing(root)


if __name__ == "__main__":
    main()
