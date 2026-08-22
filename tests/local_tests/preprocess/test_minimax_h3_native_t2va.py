# SPDX-License-Identifier: Apache-2.0
"""Contracts for the H3 native-shape preprocessing harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

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
    with pytest.raises(ValueError, match="unreferenced parquet"):
        finalizer.audit_source(source_root, {"source": "source"})


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
