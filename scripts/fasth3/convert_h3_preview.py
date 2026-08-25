# SPDX-License-Identifier: Apache-2.0
"""Convert the FastH3-Preview 33B into pre-quantized MLX checkpoints (one per
format), streamed so a 36 GB Mac never holds the 66 GB bf16 resident.

Each output dir is a self-contained, shippable artifact in the
`mlx_h3_dit.safetensors` format: weights already cast/quantized, optional
AdaLN tables dropped at load time by the runtime. After this, the survey and
all repeat runs load the compact checkpoints instead of the diffusers files.

Usage (on the target Mac — 36 GB is plenty):

    python scripts/fasth3/convert_h3_preview.py \\
        --model-root ~/models/FastH3-Preview-v0.2/transformer \\
        --out ~/models/FastH3-MLX \\
        --formats "int8 int6 int4 mxfp8 mxfp4 fp16"

Output layout: `~/models/FastH3-MLX/<format>/mlx_h3_dit.safetensors` + manifest.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported
from fastvideo.mlx_runtime.minimax_h3 import (
    mlx_h3_dit_from_diffusers_safetensors,
    save_mlx_h3_checkpoint,
)

logger = init_logger(__name__)

DEFAULT_FORMATS = "int8 int6 int4 mxfp8 mxfp4 fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="transformer/ dir of the diffusers snapshot")
    parser.add_argument("--out", required=True, help="base dir for the per-format MLX checkpoints")
    parser.add_argument("--formats", default=DEFAULT_FORMATS)
    return parser.parse_args()


def main() -> None:
    import mlx.core as mx

    args = parse_args()
    formats = args.formats.split()
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == "fp16":
            spec, dtype = None, "fp16"
        else:
            spec = MLXQuantizationSpec.from_name(fmt)
            ensure_quantization_supported(spec)
            dtype = "fp16"
        out_dir = out_base / fmt
        if (out_dir / "mlx_h3_dit.json").exists():
            print(f"[skip] {fmt} already converted at {out_dir}", flush=True)
            continue
        print(f"[convert] {fmt} (dtype={dtype}, spec={spec})", flush=True)
        t0 = time.perf_counter()
        dit = mlx_h3_dit_from_diffusers_safetensors(
            args.model_root,
            quantization=spec,
            dtype=dtype,
        )
        mx.eval()
        save_mlx_h3_checkpoint(dit, out_dir)
        peak_gb = float(getattr(mx, "get_peak_memory", lambda: 0)()) / 1e9
        print(f"[done] {fmt} in {time.perf_counter() - t0:.1f}s | peak {peak_gb:.1f} GB -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()