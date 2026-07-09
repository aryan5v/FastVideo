# SPDX-License-Identifier: Apache-2.0
"""Write a tiny synthetic T2V parquet in the ``pyarrow_schema_t2v`` format.

For the SF+QAD training smoke: the values are random, but the schema/shapes are
exactly what ``collate_rows_from_parquet_schema`` and the Wan T2V dataloader
expect, so the training loop runs end-to-end. Not for quality — only to validate
that the recipe assembles, ``mlx_qat`` arms, and a few steps produce finite loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v


def _tensor_fields(name: str, array: np.ndarray) -> dict:
    return {
        f"{name}_bytes": array.tobytes(),
        f"{name}_shape": list(array.shape),
        f"{name}_dtype": str(array.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a tiny T2V parquet for smoke training.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--latent-t", type=int, default=3)
    parser.add_argument("--latent-h", type=int, default=60)
    parser.add_argument("--latent-w", type=int, default=104)
    parser.add_argument("--text-seqlen", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=4096)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for i in range(args.rows):
        latent = (rng.standard_normal((args.channels, args.latent_t, args.latent_h, args.latent_w)) * 0.5).astype(
            np.float32)
        text = (rng.standard_normal((args.text_seqlen, args.text_dim)) * 0.1).astype(np.float32)
        row = {
            "id": f"synth-{i:04d}",
            "file_name": f"synth-{i:04d}.mp4",
            "caption": "a synthetic smoke-test clip",
            "media_type": "video",
            "width": args.latent_w * 8,
            "height": args.latent_h * 8,
            "num_frames": (args.latent_t - 1) * 4 + 1,
            "duration_sec": 5.0,
            "fps": 16.0,
        }
        row.update(_tensor_fields("vae_latent", latent))
        row.update(_tensor_fields("text_embedding", text))
        rows.append(row)

    table = pa.Table.from_pylist(rows, schema=pyarrow_schema_t2v)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "synth_00000.parquet"
    pq.write_table(table, out_path)
    print(f"wrote {out_path} ({args.rows} rows, latent {args.channels}x{args.latent_t}x{args.latent_h}x{args.latent_w}, "
          f"text {args.text_seqlen}x{args.text_dim})", flush=True)


if __name__ == "__main__":
    main()
