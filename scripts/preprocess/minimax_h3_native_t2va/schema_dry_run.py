# SPDX-License-Identifier: Apache-2.0
"""CPU-only round-trip check for one dynamic-shape H3 T2VA parquet row."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq



def load_t2va_schema():
    schema_path = Path(__file__).resolve().parents[3] / "fastvideo" / "dataset" / "dataloader" / "schema.py"
    spec = importlib.util.spec_from_file_location("_fastvideo_parquet_schema", schema_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load schema from {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pyarrow_schema_t2va


def main() -> None:
    width, height, frames = 480, 832, 294
    video_frames = (frames - 5) // 17 * 5 + 2
    audio_frames = round(frames / 24 * 40)
    arrays = {
        "vae_latent": np.zeros((24, video_frames, height // 16, width // 16), dtype=np.float32),
        "audio_latent": np.zeros((2, 32, audio_frames), dtype=np.float32),
        "text_embedding": np.zeros((3, 5120), dtype=np.float32),
    }
    row = {"id": "schema-dry-run"}
    for name, array in arrays.items():
        row[f"{name}_bytes"] = array.tobytes()
        row[f"{name}_shape"] = list(array.shape)
        row[f"{name}_dtype"] = "float32"
    row.update({
        "file_name": "schema-dry-run.mp4",
        "caption": "schema dry run",
        "media_type": "video_with_audio",
        "width": width,
        "height": height,
        "num_frames": frames,
        "duration_sec": frames / 24,
        "fps": 24.0,
        "audio_sample_rate": 32000,
    })
    pyarrow_schema_t2va = load_t2va_schema()
    table = pa.table({name: [row[name]] for name in pyarrow_schema_t2va.names}, schema=pyarrow_schema_t2va)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"bucket={width}x{height}-{frames}f" / "probe.parquet"
        path.parent.mkdir()
        pq.write_table(table, path, compression="zstd", row_group_size=1)
        parquet = pq.ParquetFile(path)
        assert parquet.schema_arrow == pyarrow_schema_t2va
        assert parquet.num_row_groups == parquet.metadata.num_rows == 1
        loaded = parquet.read().to_pylist()[0]
        for name, array in arrays.items():
            restored = np.frombuffer(loaded[f"{name}_bytes"], dtype=np.float32).reshape(loaded[f"{name}_shape"])
            assert restored.shape == array.shape
    print(
        f"schema dry-run passed: bucket={width}x{height}-{frames}f, "
        f"video={arrays['vae_latent'].shape}, audio={arrays['audio_latent'].shape}, text={arrays['text_embedding'].shape}"
    )


if __name__ == "__main__":
    main()
