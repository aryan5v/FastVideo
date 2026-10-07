# SPDX-License-Identifier: Apache-2.0
"""Check how a preprocessed H3 parquet stores its video latents before distilling on it.

Prints the parquet schema and per-channel latent statistics, then decodes the
first row with the teacher VAE twice (as stored, and denormalized with the
VAE's latent mean/std) and reports PSNR of each against the source video named
in ``MANIFEST_rows.jsonl``. The interpretation with the higher PSNR is the one
to pass to ``train_qad.py --latents-normalized``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "minimax_h3_vae"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_decoder import load_h3_vae, psnr, read_frames  # noqa: E402
from train_qad import LATENT_COLUMNS, row_latent  # noqa: E402


def source_frames(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    frames = read_frames(path, num_frames).float() / 255  # T, C, H, W
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
    return frames.permute(1, 0, 2, 3).unsqueeze(0).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="one source dir with MANIFEST_rows.jsonl and data/")
    parser.add_argument("--teacher-vae-dir", required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    manifest = [json.loads(line) for line in open(Path(args.dataset_dir) / "MANIFEST_rows.jsonl")]
    first = manifest[0]
    parquet = pq.ParquetFile(first["parquet"])
    print("schema:", parquet.schema_arrow.names)
    print("manifest row:", {k: first[k] for k in ("bucket", "vae_latent_shape", "raw_video_path") if k in first})
    row = parquet.read_row_group(0).slice(0, 1).to_pylist()[0]
    print("first row scalars:", {k: v for k, v in row.items() if k not in LATENT_COLUMNS and not isinstance(v, bytes)
                                 and not (isinstance(v, list) and len(v) > 8)})
    latent = row_latent(row).unsqueeze(0)

    device = torch.device("cuda")
    vae = load_h3_vae(args.teacher_vae_dir, device)
    mean, std = vae.latents_mean.cpu(), vae.latents_std.cpu()
    print("stored per-channel mean:", [round(v, 3) for v in latent.mean(dim=(0, 2, 3, 4)).tolist()])
    print("stored per-channel std: ", [round(v, 3) for v in latent.std(dim=(0, 2, 3, 4)).tolist()])
    print("vae latents_mean:       ", [round(v, 3) for v in mean.flatten().tolist()])
    print("vae latents_std:        ", [round(v, 3) for v in std.flatten().tolist()])

    results = {}
    for name, z in (("as_stored", latent), ("denormalized", latent * std + mean)):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            sample = vae.decode(z.to(device), return_dict=False)[0]
        video = vae.denormalize_pixels(sample.float()).clamp(0, 1).cpu()
        reference = source_frames(first["raw_video_path"], video.shape[2], video.shape[3], video.shape[4])
        frames = min(video.shape[2], reference.shape[2])
        results[name] = psnr(video[:, :, :frames], reference[:, :, :frames])
        print(f"{name}: decoded {tuple(video.shape)} PSNR vs source {results[name]:.2f} dB", flush=True)
    verdict = "no" if results["as_stored"] >= results["denormalized"] else "yes"
    print(f"VERDICT --latents-normalized {verdict}")


if __name__ == "__main__":
    main()
