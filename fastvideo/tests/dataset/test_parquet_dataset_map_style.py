# SPDX-License-Identifier: Apache-2.0
"""Unit tests for parquet map-style dataset path parsing."""

from __future__ import annotations

import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastvideo.dataset import parquet_dataset_map_style as parquet_dataset
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2va, pyarrow_schema_text_only
from fastvideo.dataset.parquet_dataset_map_style import (
    _parse_data_path_specs,
    LatentsParquetMapStyleDataset,
    read_row_from_parquet_file,
)


def test_parse_data_path_specs_accepts_old_repeat_string() -> None:
    assert _parse_data_path_specs("data/path1:2,data/path2:1") == [
        ("data/path1", 2),
        ("data/path2", 1),
    ]


def test_parse_data_path_specs_accepts_yaml_mapping() -> None:
    assert _parse_data_path_specs({
        "data/path1": 1,
        "data/path2": 2,
    }) == [
        ("data/path1", 1),
        ("data/path2", 2),
    ]


def test_parse_data_path_specs_accepts_path_list() -> None:
    assert _parse_data_path_specs(["data/a", "data/b"]) == [
        ("data/a", 1),
        ("data/b", 1),
    ]


def test_parse_data_path_specs_drops_non_positive_repeats() -> None:
    assert _parse_data_path_specs({
        "data/a": 0,
        "data/b": -1,
        "data/c": 2
    }) == [
        ("data/c", 2),
    ]
    assert _parse_data_path_specs("data/a:0,data/b:2") == [("data/b", 2)]


def test_parse_data_path_specs_rejects_malformed_repeats() -> None:
    with pytest.raises(ValueError):
        _parse_data_path_specs("data/a:abc")
    with pytest.raises(ValueError):
        _parse_data_path_specs("data/a:1.5")
    with pytest.raises(ValueError):
        _parse_data_path_specs({"data/a": "abc"})  # type: ignore[dict-item]


class _DummyWorldGroup:

    def barrier(self) -> None:
        return None


def _write_root_cache(dataset_root, filename: str, length: int) -> str:
    cache_dir = dataset_root / "map_style_cache"
    cache_dir.mkdir(parents=True)
    parquet_file = dataset_root / filename
    parquet_file.touch()
    with (cache_dir / "file_info.pkl").open("wb") as f:
        pickle.dump(((str(parquet_file), ), (length, )), f)
    return str(parquet_file)


def test_get_parquet_files_and_length_repeats_single_path(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    parquet_file = _write_root_cache(dataset_root, "sample.parquet", 7)

    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: 0)
    monkeypatch.setattr(parquet_dataset, "get_world_group", _DummyWorldGroup)

    file_names, lengths = parquet_dataset.get_parquet_files_and_length({
        str(dataset_root): 2,
    })

    assert file_names == (parquet_file, parquet_file)
    assert lengths == (7, 7)


def test_get_parquet_files_and_length_mixes_roots_and_resorts(tmp_path, monkeypatch) -> None:
    root_a = tmp_path / "dataset_a"
    root_b = tmp_path / "dataset_b"
    file_a = _write_root_cache(root_a, "a.parquet", 5)
    file_b = _write_root_cache(root_b, "b.parquet", 9)

    monkeypatch.setattr(parquet_dataset, "get_world_rank", lambda: 0)
    monkeypatch.setattr(parquet_dataset, "get_world_group", _DummyWorldGroup)

    file_names, lengths = parquet_dataset.get_parquet_files_and_length({
        str(root_b): 1,
        str(root_a): 2,
    })

    assert file_names == (file_a, file_a, file_b)
    assert lengths == (5, 5, 9)


def test_get_parquet_files_and_length_raises_when_all_repeats_dropped() -> None:
    with pytest.raises(FileNotFoundError):
        parquet_dataset.get_parquet_files_and_length({"data/a": 0})


def test_read_row_projects_text_columns_from_t2va_superset(tmp_path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    row = {
        "id": ["sample-0"],
        "vae_latent_bytes": [b"video-must-not-be-read"],
        "vae_latent_shape": [[24, 2, 4, 4]],
        "vae_latent_dtype": ["float32"],
        "audio_latent_bytes": [b"audio-must-not-be-read"],
        "audio_latent_shape": [[2, 32, 8]],
        "audio_latent_dtype": ["float32"],
        "text_embedding_bytes": [b"text"],
        "text_embedding_shape": [[1, 4]],
        "text_embedding_dtype": ["float32"],
        "file_name": ["sample.mp4"],
        "caption": ["prompt"],
        "media_type": ["video"],
        "width": [64],
        "height": [64],
        "num_frames": [5],
        "duration_sec": [5.0 / 24.0],
        "fps": [24.0],
        "audio_sample_rate": [32_000],
    }
    pq.write_table(pa.Table.from_pydict(row, schema=pyarrow_schema_t2va), parquet_path)
    text_columns = [
        "id",
        "text_embedding_bytes",
        "text_embedding_shape",
        "text_embedding_dtype",
        "caption",
    ]

    projected = read_row_from_parquet_file([str(parquet_path)], 0, [1], columns=text_columns)

    assert projected == {
        "id": "sample-0",
        "text_embedding_bytes": b"text",
        "text_embedding_shape": [1, 4],
        "text_embedding_dtype": "float32",
        "caption": "prompt",
    }


def test_dataset_projects_its_declared_schema_columns(monkeypatch) -> None:
    observed = {}

    def fake_read(parquet_files, global_row_idx, lengths, columns=None):
        observed["columns"] = columns
        return {"id": "sample-0"}

    monkeypatch.setattr(parquet_dataset, "read_row_from_parquet_file", fake_read)
    monkeypatch.setattr(parquet_dataset, "collate_rows_from_parquet_schema", lambda rows, *args, **kwargs: rows[0])
    dataset = LatentsParquetMapStyleDataset.__new__(LatentsParquetMapStyleDataset)
    dataset.parquet_files = ("unused.parquet", )
    dataset.lengths = (1, )
    dataset.parquet_schema = pyarrow_schema_text_only
    dataset.text_padding_length = 512
    dataset.cfg_rate = 0.0
    dataset.seed = 42
    dataset.sample_bucket_ids = None

    assert dataset.__getitems__([0]) == {"id": "sample-0", "_sample_index": 0}
    assert observed["columns"] == pyarrow_schema_text_only.names
