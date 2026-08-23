# SPDX-License-Identifier: Apache-2.0
"""Contracts for the H3 native-shape preprocessing harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "scripts" / "preprocess" / "minimax_h3_native_t2va"


def load_script(name: str):
    path = HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_h3_native_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(record_id: str) -> dict:
    return {
        "conditioning_id": record_id,
        "source": "test-source",
        "raw_video_path": f"/raw/{record_id}.mp4",
        "prompt": "prompt",
        "width": 480,
        "height": 832,
        "num_frames": 294,
        "fps": 24.0,
        "duration_sec": 294 / 24,
        "audio_sample_rate": 32000,
    }


def chunk(*record_ids: str) -> dict:
    return {
        "chunk_id": "c00000",
        "shape": {"width": 480, "height": 832, "num_frames": 294},
        "conditioning_ids": list(record_ids),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))


def write_t2va_parquet(finalizer, path: Path, frozen_row: dict, **overrides) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    video_shape, audio_shape = finalizer.expected_shapes(frozen_row)
    text_shape = [1, 5120]
    record = {
        "id": frozen_row["conditioning_id"],
        "vae_latent_bytes": bytes(4 * 24 * video_shape[1] * video_shape[2] * video_shape[3]),
        "vae_latent_shape": video_shape,
        "vae_latent_dtype": "float32",
        "audio_latent_bytes": bytes(4 * 2 * 32 * audio_shape[2]),
        "audio_latent_shape": audio_shape,
        "audio_latent_dtype": "float32",
        "text_embedding_bytes": bytes(4 * text_shape[0] * text_shape[1]),
        "text_embedding_shape": text_shape,
        "text_embedding_dtype": "float32",
        "file_name": Path(frozen_row["raw_video_path"]).name,
        "caption": frozen_row["prompt"],
        "media_type": "video_with_audio",
        "width": frozen_row["width"],
        "height": frozen_row["height"],
        "num_frames": frozen_row["num_frames"],
        "duration_sec": frozen_row["duration_sec"],
        "fps": frozen_row["fps"],
        "audio_sample_rate": frozen_row["audio_sample_rate"],
    }
    record.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record], schema=finalizer.load_t2va_schema())
    pq.write_table(table, path, compression="zstd", row_group_size=1)
    return record


def create_frozen_tree(tmp_path: Path) -> tuple[Path, Path]:
    freezer = load_script("freeze_sources")
    root = (tmp_path / "dataset").resolve()
    source = "source"
    source_root = root / source
    inputs = (tmp_path / "inputs").resolve()
    videos_dir = inputs / "videos"
    videos_dir.mkdir(parents=True)
    status_path = inputs / "status.jsonl"
    prompts_path = inputs / "prompts.jsonl"
    status_path.write_text("")
    prompts_path.write_text("")

    frozen_rows = []
    for index in range(65):
        record_id = f"sample-{index:02d}"
        video = videos_dir / f"{record_id}.mp4"
        video.write_bytes(f"frozen-{record_id}".encode())
        stat = video.stat()
        frozen_rows.append({
            "schema_version": freezer.SCHEMA_VERSION,
            "source": source,
            "family": "nuva",
            "conditioning_id": record_id,
            "prompt": f"prompt {record_id}",
            "raw_video_path": str(video),
            "width": 16,
            "height": 16,
            "num_frames": 5,
            "fps": 24.0,
            "duration_sec": 5 / 24,
            "audio_sample_rate": 32000,
            "audio_channels": 2,
            "audio_samples": 0,
            "audio_duration_sec": 0.0,
            "bucket_id": "",
            "status_line": index + 1,
            "prompt_line": index + 1,
            "video_size_bytes": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
        })
    validation_rows = [freezer.validation_row(item) for item in frozen_rows[:64]]
    training_rows = frozen_rows[64:]
    write_jsonl(source_root / "media" / "frozen.jsonl", frozen_rows)
    write_jsonl(source_root / "media" / "train.jsonl", training_rows)
    write_jsonl(
        source_root / "prompts" / "source.jsonl",
        [{"conditioning_id": item["conditioning_id"], "prompt": item["prompt"]} for item in frozen_rows],
    )
    worklist = freezer.build_worklist(training_rows, 32)
    worklist.update({
        "source": source,
        "train_manifest": str(source_root / "media" / "train.jsonl"),
        "set_root": str(source_root),
    })
    (source_root / "work").mkdir(parents=True)
    (source_root / "work" / "worklist.json").write_text(json.dumps(worklist, indent=2, sort_keys=True) + "\n")
    artifact_paths = {
        "source.jsonl": source_root / "prompts" / "source.jsonl",
        "frozen.jsonl": source_root / "media" / "frozen.jsonl",
        "train.jsonl": source_root / "media" / "train.jsonl",
        "worklist.json": source_root / "work" / "worklist.json",
    }
    artifacts_sha256 = {name: freezer.sha256_file(path) for name, path in artifact_paths.items()}
    (source_root / "prompts" / "SOURCE.sha256").write_text(
        f"{artifacts_sha256['source.jsonl']}  source.jsonl\n"
    )
    source_summary = {
        "source": source,
        "completed_status_ids": 65,
        "completed_status_lines": 65,
        "canonical_mp4s": 65,
        "completed_and_canonical_mp4": 65,
        "prompt_records_seen": 65,
        "frozen_rows": 65,
        "missing_canonical_mp4": 0,
        "canonical_mp4_without_completed_status": 0,
        "eligible_without_valid_prompt": 0,
        "status_jsonl": str(status_path),
        "status_snapshot_sha256": freezer.sha256_file(status_path),
        "status_snapshot_bytes": 0,
        "prompts_jsonl": str(prompts_path),
        "prompts_snapshot_sha256": freezer.sha256_file(prompts_path),
        "prompts_snapshot_bytes": 0,
        "videos_dir": str(videos_dir),
        "training_rows": 1,
        "validation_exclusions": 64,
        "artifacts_sha256": artifacts_sha256,
        "shape_counts": {"16x16x5": 65},
    }
    (source_root / "MANIFEST.source.json").write_text(json.dumps(source_summary, indent=2, sort_keys=True) + "\n")

    validation_root = root / "validation"
    write_jsonl(validation_root / "manifest.jsonl", validation_rows)
    validation_summary = {
        "schema_version": freezer.VALIDATION_SCHEMA_VERSION,
        "seed": 20260822,
        "rows": 64,
        "unique_conditioning_ids": 64,
        "source_counts": {source: 64},
        "family_counts": {"nuva": 64},
        "training_exclusions_by_source": {source: 64},
    }
    (validation_root / "manifest.json").write_text(json.dumps(validation_summary, indent=2, sort_keys=True) + "\n")
    heldout = freezer.heldout_payload(validation_rows)
    (validation_root / "heldout64.json").write_text(json.dumps(heldout, indent=2) + "\n")
    validation_videos = validation_root / "videos"
    validation_videos.mkdir()
    for item in validation_rows:
        (validation_videos / f"{source}__{item['conditioning_id']}.mp4").symlink_to(item["raw_video_path"])

    config = {
        "schema_version": "minimax-h3-native-t2va-sources-v1",
        "snapshot_seed": 20260822,
        "output_root": str(root),
        "legacy_validation_ids": str(inputs / "legacy.txt"),
        "validation_quotas": {source: 64},
        "sources": [{
            "name": source,
            "family": "nuva",
            "videos_dir": str(videos_dir),
            "status_jsonl": str(status_path),
            "prompts_jsonl": str(prompts_path),
            "prompt_id_field": "id",
            "prompt_text_field": "prompt",
            "require_prompt_validation_passed": False,
        }],
    }
    config_path = inputs / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    frozen_manifest = {
        "schema_version": freezer.SCHEMA_VERSION,
        "created_utc": "2026-08-22T00:00:00+00:00",
        "config_path": str(config_path),
        "config_sha256": freezer.sha256_file(config_path),
        "seed": 20260822,
        "sources": [source_summary],
        "frozen_rows": 65,
        "training_rows": 1,
        "validation_rows": 64,
        "validation_manifest_sha256": freezer.sha256_file(validation_root / "manifest.jsonl"),
        "heldout64_sha256": freezer.sha256_file(validation_root / "heldout64.json"),
        "ready": False,
    }
    (root / "FROZEN_MANIFEST.json").write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    return root, Path(training_rows[0]["raw_video_path"])


def add_training_only_source(root: Path, source: str) -> None:
    """Add a second immutable source to the small frozen-tree fixture."""
    freezer = load_script("freeze_sources")
    root_manifest_path = root / "FROZEN_MANIFEST.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    config_path = Path(root_manifest["config_path"])
    config = json.loads(config_path.read_text())
    inputs = config_path.parent / source
    videos_dir = inputs / "videos"
    videos_dir.mkdir(parents=True)
    status_path = inputs / "status.jsonl"
    prompts_path = inputs / "prompts.jsonl"
    status_path.write_text("")
    prompts_path.write_text("")

    record_id = f"{source}-train"
    video = videos_dir / f"{record_id}.mp4"
    video.write_bytes(b"fixed-source-video")
    stat = video.stat()
    frozen_row = {
        "schema_version": freezer.SCHEMA_VERSION,
        "source": source,
        "family": "nuva",
        "conditioning_id": record_id,
        "prompt": "fixed source prompt",
        "raw_video_path": str(video.resolve()),
        "width": 16,
        "height": 16,
        "num_frames": 5,
        "fps": 24.0,
        "duration_sec": 5 / 24,
        "audio_sample_rate": 32000,
        "audio_channels": 2,
        "audio_samples": 0,
        "audio_duration_sec": 0.0,
        "bucket_id": "",
        "status_line": 1,
        "prompt_line": 1,
        "video_size_bytes": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
    }
    source_root = root / source
    write_jsonl(source_root / "media" / "frozen.jsonl", [frozen_row])
    write_jsonl(source_root / "media" / "train.jsonl", [frozen_row])
    write_jsonl(
        source_root / "prompts" / "source.jsonl",
        [{"conditioning_id": record_id, "prompt": frozen_row["prompt"]}],
    )
    worklist = freezer.build_worklist([frozen_row], 32)
    worklist.update({
        "source": source,
        "train_manifest": str(source_root / "media" / "train.jsonl"),
        "set_root": str(source_root),
    })
    (source_root / "work").mkdir(parents=True)
    (source_root / "work" / "worklist.json").write_text(json.dumps(worklist, indent=2, sort_keys=True) + "\n")
    artifact_paths = {
        "source.jsonl": source_root / "prompts" / "source.jsonl",
        "frozen.jsonl": source_root / "media" / "frozen.jsonl",
        "train.jsonl": source_root / "media" / "train.jsonl",
        "worklist.json": source_root / "work" / "worklist.json",
    }
    artifacts_sha256 = {name: freezer.sha256_file(path) for name, path in artifact_paths.items()}
    (source_root / "prompts" / "SOURCE.sha256").write_text(
        f"{artifacts_sha256['source.jsonl']}  source.jsonl\n"
    )
    source_summary = {
        "source": source,
        "completed_status_ids": 0,
        "completed_status_lines": 0,
        "canonical_mp4s": 1,
        "completed_and_canonical_mp4": 0,
        "prompt_records_seen": 0,
        "frozen_rows": 1,
        "missing_canonical_mp4": 0,
        "canonical_mp4_without_completed_status": 1,
        "eligible_without_valid_prompt": 0,
        "status_jsonl": str(status_path.resolve()),
        "status_snapshot_sha256": freezer.sha256_file(status_path),
        "status_snapshot_bytes": 0,
        "prompts_jsonl": str(prompts_path.resolve()),
        "prompts_snapshot_sha256": freezer.sha256_file(prompts_path),
        "prompts_snapshot_bytes": 0,
        "videos_dir": str(videos_dir.resolve()),
        "training_rows": 1,
        "validation_exclusions": 0,
        "artifacts_sha256": artifacts_sha256,
        "shape_counts": {"16x16x5": 1},
    }
    (source_root / "MANIFEST.source.json").write_text(json.dumps(source_summary, indent=2, sort_keys=True) + "\n")

    config["sources"].append({
        "name": source,
        "family": "nuva",
        "videos_dir": str(videos_dir.resolve()),
        "status_jsonl": str(status_path.resolve()),
        "prompts_jsonl": str(prompts_path.resolve()),
        "prompt_id_field": "id",
        "prompt_text_field": "prompt",
        "require_prompt_validation_passed": False,
    })
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    root_manifest["config_sha256"] = freezer.sha256_file(config_path)
    root_manifest["sources"].append(source_summary)
    root_manifest["frozen_rows"] += 1
    root_manifest["training_rows"] += 1
    root_manifest_path.write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n")

    validation_summary_path = root / "validation" / "manifest.json"
    validation_summary = json.loads(validation_summary_path.read_text())
    validation_summary["training_exclusions_by_source"][source] = 0
    validation_summary_path.write_text(json.dumps(validation_summary, indent=2, sort_keys=True) + "\n")


def create_seed_trees(tmp_path: Path) -> tuple[object, Path, Path, str, dict]:
    seeder = load_script("seed_encoded_chunks")
    base_root = (tmp_path / "base").resolve()
    new_root = (tmp_path / "new").resolve()
    source = "source"
    base_source = base_root / source
    new_source = new_root / source
    frozen_rows = [row("one"), row("two")]
    for frozen_row in frozen_rows:
        frozen_row["source"] = source
    write_jsonl(base_source / "media" / "train.jsonl", frozen_rows)
    write_jsonl(new_source / "media" / "train.jsonl", frozen_rows)
    base_worklist = {
        "chunks": [{
            "chunk_id": "c00000",
            "shape": {"width": 480, "height": 832, "num_frames": 294},
            "conditioning_ids": ["one", "two"],
        }],
    }
    new_worklist = {
        "chunks": [{
            "chunk_id": "c00042",
            "shape": {"width": 480, "height": 832, "num_frames": 294},
            "conditioning_ids": ["one", "two"],
        }],
    }
    (base_source / "work").mkdir(parents=True)
    (new_source / "work").mkdir(parents=True)
    (base_source / "work" / "worklist.json").write_text(json.dumps(base_worklist) + "\n")
    (new_source / "work" / "worklist.json").write_text(json.dumps(new_worklist) + "\n")

    base_parquet = (base_source / "data" / "bucket=480x832-294f" / "c00000.parquet").resolve()
    base_parquet.parent.mkdir(parents=True)
    base_parquet.write_bytes(b"trusted encoded rows")
    parquet_sha256 = seeder.sha256_file(base_parquet)
    receipt_rows = {
        frozen_row["conditioning_id"]: {
            **frozen_row,
            "parquet": str(base_parquet),
            "parquet_sha256": parquet_sha256,
        }
        for frozen_row in frozen_rows
    }
    base_receipt = {
        "manifest": {"parquet_sha256": {str(base_parquet): parquet_sha256}},
        "rows_by_id": receipt_rows,
    }
    return seeder, base_root, new_root, source, base_receipt


def test_failed_chunk_publishes_no_data_or_done(tmp_path, monkeypatch) -> None:
    worker = load_script("encode_worker")
    rows = {record_id: row(record_id) for record_id in ("ok", "retry")}

    def fake_encode(input_row, _encoders):
        if input_row["conditioning_id"] == "retry":
            raise RuntimeError("transient")
        return {"id": "ok"}, {"conditioning_id": "ok"}

    monkeypatch.setattr(worker, "encode_row", fake_encode)
    args = SimpleNamespace(worker_tag="test", stale_minutes=90.0, timing=False)
    worker.process_chunk(chunk("ok", "retry"), rows, tmp_path, object(), args)

    assert not (tmp_path / "done" / "c00000.json").exists()
    assert not list((tmp_path / "data").rglob("*.parquet")) if (tmp_path / "data").exists() else True
    assert len(list((tmp_path / "work" / "failures").glob("c00000.*.json"))) == 1
    assert len(list((tmp_path / "work" / "failed_claims").glob("c00000.*"))) == 1


def test_successful_chunk_uses_locked_bucket_path(tmp_path, monkeypatch) -> None:
    worker = load_script("encode_worker")

    def fake_encode(input_row, _encoders):
        return {"id": input_row["conditioning_id"]}, {"conditioning_id": input_row["conditioning_id"]}

    def fake_write_parquet(_records, path, _worker_tag):
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr(worker, "encode_row", fake_encode)
    monkeypatch.setattr(worker, "write_parquet", fake_write_parquet)
    args = SimpleNamespace(worker_tag="test", stale_minutes=90.0, timing=False)
    worker.process_chunk(chunk("ok"), {"ok": row("ok")}, tmp_path, object(), args)

    assert (tmp_path / "data" / "bucket=480x832-294f" / "c00000.parquet").is_file()
    assert (tmp_path / "done" / "c00000.json").is_file()


def test_finalizer_rejects_unreferenced_parquet(tmp_path) -> None:
    finalizer = load_script("finalize_dataset")
    source_root = tmp_path / "source"
    (source_root / "media").mkdir(parents=True)
    (source_root / "work").mkdir()
    (source_root / "media" / "train.jsonl").write_text(
        '{"conditioning_id":"one","width":480,"height":832,"num_frames":294}\n'
    )
    (source_root / "work" / "worklist.json").write_text(
        '{"chunks":[{"chunk_id":"c00000"}]}\n'
    )
    stray = source_root / "data" / "bucket=480x832-294f" / "stray.parquet"
    stray.parent.mkdir(parents=True)
    stray.touch()
    source_summary = {
        "source": "source",
        "artifacts_sha256": {
            "train.jsonl": finalizer.sha256_file(source_root / "media" / "train.jsonl"),
        },
    }
    with pytest.raises(ValueError, match="unreferenced parquet"):
        finalizer.audit_source(source_root, source_summary)


def test_cache_must_exactly_match_audited_files(tmp_path) -> None:
    finalizer = load_script("finalize_dataset")
    source_root = tmp_path / "source"
    cache = source_root / "data" / "map_style_cache" / "file_info.pkl"
    cache.parent.mkdir(parents=True)
    with cache.open("wb") as handle:
        pickle.dump((("/wrong.parquet", ), (1, )), handle)
    with pytest.raises(ValueError, match="does not exactly match"):
        finalizer.validate_cache(
            source_root,
            {"parquet_files": ("/right.parquet", ), "parquet_lengths": (1, )},
        )


def test_shape_and_bucket_contracts() -> None:
    finalizer = load_script("finalize_dataset")
    assert finalizer.BUCKET_RE.fullmatch("bucket=1344x768-124f")
    assert not finalizer.BUCKET_RE.fullmatch("bucket=768x1344x124")
    video, audio = finalizer.expected_shapes({"width": 480, "height": 832, "num_frames": 294})
    assert video == [24, 87, 52, 30]
    assert audio == [2, 32, 490]


def test_finalizer_rejects_one_byte_short_tensor_payload(tmp_path) -> None:
    finalizer = load_script("finalize_dataset")
    frozen_row = row("one")
    frozen_row.update({"width": 16, "height": 16, "num_frames": 5, "duration_sec": 5 / 24})
    video_shape, _ = finalizer.expected_shapes(frozen_row)
    valid_bytes = 4 * video_shape[0] * video_shape[1] * video_shape[2] * video_shape[3]
    parquet_path = tmp_path / "bucket=16x16-5f" / "one.parquet"
    write_t2va_parquet(finalizer, parquet_path, frozen_row, vae_latent_bytes=bytes(valid_bytes - 1))

    with pytest.raises(ValueError, match=r"vae_latent_bytes length .* != shape\*dtype"):
        finalizer.inspect_parquet(parquet_path, "16x16-5f", {"one": frozen_row})


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("vae_latent_dtype", "float16", "vae_latent_dtype"),
        ("audio_latent_dtype", "float16", "audio_latent_dtype"),
        ("text_embedding_dtype", "float16", "text_embedding_dtype"),
        ("media_type", "video", "media_type"),
        ("width", 32, "row geometry"),
        ("height", 32, "row geometry"),
        ("num_frames", 22, "row geometry"),
        ("caption", "changed", "caption"),
        ("file_name", "changed.mp4", "file_name"),
        ("duration_sec", 1.0, "duration"),
        ("fps", 23.0, "fps"),
        ("audio_sample_rate", 16000, "audio sample rate"),
    ],
)
def test_finalizer_rejects_parquet_fields_that_differ_from_frozen_row(
    tmp_path,
    field_name: str,
    bad_value,
    message: str,
) -> None:
    finalizer = load_script("finalize_dataset")
    frozen_row = row("one")
    frozen_row.update({"width": 16, "height": 16, "num_frames": 5, "duration_sec": 5 / 24})
    parquet_path = tmp_path / "bucket=16x16-5f" / "one.parquet"
    write_t2va_parquet(finalizer, parquet_path, frozen_row, **{field_name: bad_value})

    with pytest.raises(ValueError, match=message):
        finalizer.inspect_parquet(parquet_path, "16x16-5f", {"one": frozen_row})


def test_audio_clock_contract_covers_every_frame_count_through_supported_max() -> None:
    worker = load_script("encode_worker")
    finalizer = load_script("finalize_dataset")
    for num_frames in range(1, 363):
        target = (5 * num_frames + 1) // 3
        raw_vae_length = (5 * num_frames + 2) // 3
        assert worker.packed_audio_latent_num_frames(num_frames) == target
        assert finalizer.packed_audio_latent_num_frames(num_frames) == target
        assert raw_vae_length - target == int(num_frames % 3 == 2)


@pytest.mark.parametrize("actual_length", [602, 603, 604])
def test_audio_latents_are_reconciled_to_fastgen_clock(actual_length: int) -> None:
    worker = load_script("encode_worker")
    raw = torch.arange(2 * 3 * actual_length, dtype=torch.float32).reshape(2, 3, actual_length)
    actual = worker.reconcile_audio_latent_length(raw, num_frames=362)

    assert actual.shape == (2, 3, 603)
    if actual_length < 603:
        expected = torch.cat([raw, raw[..., -1:]], dim=-1)
    else:
        expected = raw[..., :603]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_362_frame_row_trims_raw_vae_output_before_serializing(monkeypatch) -> None:
    worker = load_script("encode_worker")
    finalizer = load_script("finalize_dataset")
    input_row = row("probe-362")
    input_row.update({"width": 16, "height": 16, "num_frames": 362, "duration_sec": 362 / 24})

    expected_video, expected_audio = worker.expected_shapes(input_row)
    assert expected_audio == (2, 32, 603)
    assert finalizer.expected_shapes(input_row)[1] == [2, 32, 603]

    class FakeEncoders:

        def encode_video(self, _frames, _seed):
            return torch.zeros(expected_video)

        def encode_audio(self, _waveform):
            return torch.arange(2 * 32 * 604, dtype=torch.float32).reshape(2, 32, 604)

        def encode_text(self, _prompt):
            return torch.zeros(1, 5120)

    monkeypatch.setattr(worker, "decode_native_media", lambda _row: (object(), object(), {}))
    record, manifest = worker.encode_row(input_row, FakeEncoders())

    assert expected_video == (24, 107, 1, 1)
    assert record["audio_latent_shape"] == [2, 32, 603]
    assert manifest["audio_latent_shape"] == [2, 32, 603]


def test_frozen_video_stat_is_enforced(tmp_path) -> None:
    worker = load_script("encode_worker")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"frozen")
    stat = video.stat()
    frozen = {
        "raw_video_path": str(video),
        "video_size_bytes": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
    }
    assert worker.verify_frozen_video(frozen) == video
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen source changed"):
        worker.verify_frozen_video(frozen)


def test_finalizer_rejects_changed_frozen_artifact(tmp_path) -> None:
    finalizer = load_script("finalize_dataset")
    root, _ = create_frozen_tree(tmp_path)
    finalizer.verify_frozen_sources(root)

    train_path = root / "source" / "media" / "train.jsonl"
    train_path.write_text(train_path.read_text() + "\n")
    with pytest.raises(ValueError, match="train.jsonl.*sha256"):
        finalizer.verify_frozen_sources(root)


def test_finalizer_rejects_changed_raw_video(tmp_path) -> None:
    finalizer = load_script("finalize_dataset")
    root, training_video = create_frozen_tree(tmp_path)
    finalizer.verify_frozen_sources(root)

    training_video.write_bytes(training_video.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen source changed"):
        finalizer.verify_frozen_sources(root)


def test_finalization_resumes_partial_source_publication_and_keeps_root_ready_last(tmp_path, monkeypatch) -> None:
    finalizer = load_script("finalize_dataset")
    root = (tmp_path / "dataset").resolve()
    summaries = [{"source": source, "validation_exclusions": 0} for source in ("source-a", "source-b")]
    frozen = {
        "sources": summaries,
        "training_rows": 0,
        "validation_rows": 64,
    }
    root.mkdir()
    (root / "FROZEN_MANIFEST.json").write_text(json.dumps(frozen, sort_keys=True) + "\n")
    audits = {}
    for summary in summaries:
        source = summary["source"]
        source_root = root / source
        source_root.mkdir()
        (source_root / "MANIFEST.source.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
        audits[source] = {
            "source": source,
            "training_rows": 0,
            "encoded_rows": 0,
            "manifest_rows": [],
            "parquet_files": (),
            "parquet_lengths": (),
            "parquet_hashes": {},
            "train_manifest_sha256": "0" * 64,
        }

    monkeypatch.setattr(finalizer, "verify_frozen_sources", lambda _root: frozen)
    monkeypatch.setattr(
        finalizer,
        "audit_source",
        lambda source_root, _summary: audits[source_root.name],
    )
    write_pickle_atomic = finalizer.write_pickle_atomic
    crashed = False

    def crash_during_second_source(path, payload):
        nonlocal crashed
        if "source-b" in path.parts and not crashed:
            crashed = True
            raise RuntimeError("simulated crash")
        write_pickle_atomic(path, payload)

    monkeypatch.setattr(finalizer, "write_pickle_atomic", crash_during_second_source)
    with pytest.raises(RuntimeError, match="simulated crash"):
        finalizer.finalize_dataset(root)

    assert (root / "source-a" / "READY.json").is_file()
    first_manifest = (root / "source-a" / "MANIFEST.json").read_bytes()
    assert (root / "source-b" / "MANIFEST_rows.jsonl").is_file()
    assert not (root / "source-b" / "READY.json").exists()
    assert not (root / "READY.json").exists()

    monkeypatch.setattr(finalizer, "write_pickle_atomic", write_pickle_atomic)
    finalizer.finalize_dataset(root)

    assert (root / "source-a" / "MANIFEST.json").read_bytes() == first_manifest
    assert (root / "source-b" / "READY.json").is_file()
    assert (root / "READY.json").is_file()
    valid_ready = json.loads((root / "READY.json").read_text())
    assert valid_ready["sources"] == ["source-a", "source-b"]
    assert valid_ready["data_paths"] == [str(root / "source-a" / "data"), str(root / "source-b" / "data")]

    wrong_sources = {**valid_ready, "sources": ["source-a"]}
    (root / "READY.json").write_text(json.dumps(wrong_sources) + "\n")
    with pytest.raises(ValueError, match="root READY sources"):
        finalizer.verify_ready(root)

    wrong_paths = {**valid_ready, "data_paths": [str(root / "wrong" / "data")] * 2}
    (root / "READY.json").write_text(json.dumps(wrong_paths) + "\n")
    with pytest.raises(ValueError, match="root READY data_paths"):
        finalizer.verify_ready(root)


def test_heldout_payload_uses_validation_dataset_data_field() -> None:
    freezer = load_script("freeze_sources")
    validation = {
        "prompt": "caption",
        "raw_video_path": "/raw/ref.mp4",
        "source": "source",
        "conditioning_id": "sample",
        "width": 1344,
        "height": 768,
        "num_frames": 124,
        "fps": 24.0,
        "audio_sample_rate": 32000,
        "audio_channels": 2,
    }
    payload = freezer.heldout_payload([validation])
    assert list(payload) == ["data"]
    assert payload["data"][0]["caption"] == "caption"
    assert payload["data"][0]["ref_video"] == "/raw/ref.mp4"


def test_extension_preserves_base_rows_and_heldout_and_only_grows_allowed_source(tmp_path) -> None:
    freezer = load_script("freeze_sources")
    base_root, _ = create_frozen_tree(tmp_path)
    fixed_source = "fixed-source"
    add_training_only_source(base_root, fixed_source)
    freezer.verify_existing(base_root, emit_summary=False)

    base_frozen = {
        source: [item for _, item in freezer.iter_jsonl(base_root / source / "media" / "frozen.jsonl")]
        for source in ("source", fixed_source)
    }
    base_heldout = (base_root / "validation" / "heldout64.json").read_bytes()
    base_validation = (base_root / "validation" / "manifest.jsonl").read_bytes()
    base_fixed_frozen = (base_root / fixed_source / "media" / "frozen.jsonl").read_bytes()

    base_manifest = json.loads((base_root / "FROZEN_MANIFEST.json").read_text())
    base_config = json.loads(Path(base_manifest["config_path"]).read_text())
    growing_spec = base_config["sources"][0]
    status_rows = []
    prompt_rows = []
    for index in range(2):
        record_id = f"new-{index}"
        video = Path(growing_spec["videos_dir"]) / f"{record_id}.mp4"
        video.write_bytes(f"new-video-{index}".encode())
        status_rows.append({
            "status": "completed",
            "id": record_id,
            "media": {
                "width": 16,
                "height": 16,
                "frames": 5,
                "fps": 24.0,
                "audio_sample_rate": 32000,
                "audio_channels": 2,
                "audio_samples": 0,
                "audio_duration_s": 0.0,
            },
        })
        prompt_rows.append({"id": record_id, "prompt": f"new prompt {index}"})
    write_jsonl(Path(growing_spec["status_jsonl"]), status_rows)
    write_jsonl(Path(growing_spec["prompts_jsonl"]), prompt_rows)

    new_root = (tmp_path / "extended").resolve()
    extension_config = {**base_config, "output_root": str(new_root)}
    extension_config_path = tmp_path / "extension-config.json"
    extension_config_path.write_text(json.dumps(extension_config, indent=2, sort_keys=True) + "\n")
    freezer.freeze_extension(
        SimpleNamespace(
            extend_existing=base_root,
            extend_source=["source"],
            chunk_size=32,
            dry_run=False,
            config=extension_config_path,
        ),
        extension_config,
        new_root,
    )

    extended_manifest = freezer.verify_existing(new_root, emit_summary=False)
    extension = extended_manifest["extension"]
    assert (new_root / "validation" / "heldout64.json").read_bytes() == base_heldout
    assert (new_root / "validation" / "manifest.jsonl").read_bytes() == base_validation
    assert (new_root / fixed_source / "media" / "frozen.jsonl").read_bytes() == base_fixed_frozen
    for source in ("source", fixed_source):
        combined_rows = {
            item["conditioning_id"]: item
            for _, item in freezer.iter_jsonl(new_root / source / "media" / "frozen.jsonl")
        }
        for base_row in base_frozen[source]:
            assert combined_rows[base_row["conditioning_id"]] == base_row
    assert extension["extend_sources"] == ["source"]
    assert extension["sources"]["source"]["added_frozen_rows"] == 2
    assert extension["sources"]["source"]["added_training_rows"] == 2
    assert extension["sources"][fixed_source]["added_frozen_rows"] == 0
    assert extension["sources"][fixed_source]["added_training_rows"] == 0


def test_frozen_config_uses_matching_local_snapshot_when_recorded_path_is_unavailable(tmp_path) -> None:
    freezer = load_script("freeze_sources")
    root = (tmp_path / "frozen").resolve()
    root.mkdir()
    snapshot = root / "CONFIG.snapshot.json"
    snapshot.write_text('{"snapshot_seed":20260822}\n')
    manifest = {
        "config_path": "/home/removed-login-user/unique-v10-source-config.json",
        "config_sha256": freezer.sha256_file(snapshot),
    }

    assert freezer.resolve_frozen_config(root, manifest) == snapshot

    snapshot.write_text('{"snapshot_seed":20260823}\n')
    with pytest.raises(FileNotFoundError, match="unavailable or has the wrong checksum"):
        freezer.resolve_frozen_config(root, manifest)


def test_seed_reuses_exact_signature_and_rewrites_destination_receipt(tmp_path) -> None:
    seeder, base_root, new_root, source, base_receipt = create_seed_trees(tmp_path)

    counts = seeder.seed_source(base_root, new_root, source, base_receipt, "hardlink", False)

    base_parquet = Path(next(iter(base_receipt["manifest"]["parquet_sha256"])))
    destination = (new_root / source / "data" / "bucket=480x832-294f" / "c00042.parquet").resolve()
    done_path = new_root / source / "done" / "c00042.json"
    done = json.loads(done_path.read_text())
    assert counts["seeded_chunks"] == 1
    assert counts["reused_rows"] == 2
    assert counts["remaining_gpu_chunks"] == 0
    assert destination.is_file()
    assert destination.samefile(base_parquet)
    assert done["chunk_id"] == "c00042"
    assert done["bucket"] == "480x832-294f"
    assert done["parquet"] == str(destination)
    assert done["seeded_from"]["chunk_id"] == "c00000"
    assert done["seeded_from"]["parquet"] == str(base_parquet)
    assert all("parquet" not in item and "parquet_sha256" not in item for item in done["rows_manifest"])

    repeated = seeder.seed_source(base_root, new_root, source, base_receipt, "hardlink", False)
    assert repeated["already_seeded_chunks"] == 1
    assert repeated["seeded_chunks"] == 0


@pytest.mark.parametrize("mismatch", ["shape", "id-order", "frozen-row"])
def test_seed_leaves_non_exact_chunks_for_gpu_encoding(tmp_path, mismatch: str) -> None:
    seeder, base_root, new_root, source, base_receipt = create_seed_trees(tmp_path)
    new_source = new_root / source
    if mismatch in {"shape", "id-order"}:
        worklist_path = new_source / "work" / "worklist.json"
        worklist = json.loads(worklist_path.read_text())
        chunk_payload = worklist["chunks"][0]
        if mismatch == "shape":
            chunk_payload["shape"]["width"] = 832
        else:
            chunk_payload["conditioning_ids"].reverse()
        worklist_path.write_text(json.dumps(worklist) + "\n")
    else:
        train_path = new_source / "media" / "train.jsonl"
        frozen_rows = [item for _, item in load_script("freeze_sources").iter_jsonl(train_path)]
        frozen_rows[0]["prompt"] = "not the frozen base prompt"
        write_jsonl(train_path, frozen_rows)

    counts = seeder.seed_source(base_root, new_root, source, base_receipt, "hardlink", False)

    assert counts["reusable_chunks"] == 0
    assert counts["remaining_gpu_chunks"] == 1
    assert not (new_source / "done" / "c00042.json").exists()
    assert not list((new_source / "data").rglob("c00042.parquet")) if (new_source / "data").exists() else True


def test_seed_rejects_corrupt_base_parquet_before_publishing(tmp_path) -> None:
    seeder, base_root, new_root, source, base_receipt = create_seed_trees(tmp_path)
    base_parquet = Path(next(iter(base_receipt["manifest"]["parquet_sha256"])))
    base_parquet.write_bytes(b"corrupt after finalization")

    with pytest.raises(ValueError, match="checksum does not match the finalized base receipt"):
        seeder.seed_source(base_root, new_root, source, base_receipt, "hardlink", False)

    new_source = new_root / source
    assert not (new_source / "done" / "c00042.json").exists()
    assert not list((new_source / "data").rglob("c00042.parquet")) if (new_source / "data").exists() else True
