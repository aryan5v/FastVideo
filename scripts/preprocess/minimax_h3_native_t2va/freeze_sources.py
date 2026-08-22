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
import tempfile
from typing import Any, Iterable

SCHEMA_VERSION = "minimax-h3-native-t2va-freeze-v1"
VALIDATION_SCHEMA_VERSION = "minimax-h3-native-t2va-validation-v1"
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
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify an already-frozen tree without consulting live sources",
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


def verify_existing(root: Path) -> None:
    root_manifest = json.loads((root / "FROZEN_MANIFEST.json").read_text())
    if root_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected frozen manifest schema")
    validation = list(row for _, row in iter_jsonl(root / "validation" / "manifest.jsonl"))
    ids = [row["conditioning_id"] for row in validation]
    if len(validation) != 64 or len(set(ids)) != 64:
        raise ValueError(f"validation split must have 64 unique ids, got {len(validation)}/{len(set(ids))}")
    heldout_payload = json.loads((root / "validation" / "heldout64.json").read_text())
    if not isinstance(heldout_payload, dict) or not isinstance(heldout_payload.get("data"), list):
        raise ValueError("heldout64.json must be an object containing a data list")
    heldout = heldout_payload["data"]
    required = {"caption", "ref_video", "source", "sample_id", "width", "height", "num_frames"}
    if len(heldout) != 64 or {row["sample_id"] for row in heldout} != set(ids):
        raise ValueError("heldout64.json must contain the same 64 unique conditioning ids")
    for row in heldout:
        if not required <= set(row):
            raise ValueError(f"heldout64.json row is missing fields: {sorted(required - set(row))}")
        if not Path(row["ref_video"]).is_file():
            raise FileNotFoundError(row["ref_video"])
    for source in root_manifest["sources"]:
        source_root = root / source["source"]
        frozen = list(row for _, row in iter_jsonl(source_root / "media" / "frozen.jsonl"))
        training = list(row for _, row in iter_jsonl(source_root / "media" / "train.jsonl"))
        if len(frozen) != source["frozen_rows"] or len(training) != source["training_rows"]:
            raise ValueError(f"{source['source']}: row-count mismatch")
        if set(ids) & {row["conditioning_id"] for row in training}:
            raise ValueError(f"{source['source']}: validation id leaked into training")
        for row in frozen:
            if not Path(row["raw_video_path"]).is_file():
                raise FileNotFoundError(row["raw_video_path"])
    print(f"verified {root}: 64 unique validation ids and no training leakage")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    root = (args.output_root or Path(config["output_root"])).resolve()
    if args.verify_existing:
        verify_existing(root)
        return
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

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
