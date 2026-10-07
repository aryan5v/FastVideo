# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware distillation of an NVFP4 MiniMax-H3 video decoder.

Teacher: the full 36-layer H3 decoder (frozen). Student: a decoder (by default
the 26-layer light one) whose transformer-block linears run NVFP4 through
``NVFP4DecoderLinear``. Each step decodes the exact unit the release decode
uses: one temporal chunk (``tokens_chunk_size + token_overlap`` latent frames)
of one 256 px spatial tile, sampled from preprocessed H3 latents. The student
matches the teacher in pixel space (L1 + LPIPS).

Evaluation decodes whole held-out clips through ``vae.decode`` (real tiling and
temporal blending, deployment autocast) and reports PSNR / SSIM / LPIPS against
the teacher's full decode.

Launch with torchrun, one process per GPU::

    torchrun --nproc-per-node 4 scripts/distill/minimax_h3_nvfp4_decoder/train_qad.py \\
        --teacher-vae-dir $FULL/vae --student-vae-dir $LIGHT/vae \\
        --data-root /path/to/h3_t2av_preprocessed/v10_mixed_native_v3 \\
        --latents-normalized yes --output-dir runs/qad --wandb-project fasth3-nvfp4-decoder
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "minimax_h3_vae"))
from bench_decoder import Fidelity, load_h3_vae  # noqa: E402

from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (  # noqa: E402
    NVFP4DecoderLinear, convert_decoder_to_nvfp4, nvfp4_decoder_metadata,
)

STUDENT_AUTOCAST = torch.bfloat16
SPATIAL_TILE_LATENT = 16
LPIPS_FRAMES_PER_TILE = 8
LATENT_COLUMNS = ["vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype"]


# ----------------------------------------------------------------------------- data


def list_parquets(roots: list[str]) -> list[str]:
    files = sorted(f for root in roots for f in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
                   if "map_style_cache" not in f)
    if not files:
        raise FileNotFoundError(f"No parquet files under {roots}")
    return files


def row_latent(row: dict) -> torch.Tensor:
    """Rebuild one ``[C, T, H, W]`` latent from FastVideo's parquet record columns."""
    dtype_name = str(row["vae_latent_dtype"]).replace("torch.", "")
    shape = row["vae_latent_shape"]
    if dtype_name == "bfloat16":
        raw = np.frombuffer(row["vae_latent_bytes"], dtype=np.uint16).reshape(shape)
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).float()
    array = np.frombuffer(row["vae_latent_bytes"], dtype=np.dtype(dtype_name)).reshape(shape)
    return torch.from_numpy(array.copy()).float()


def chunk_starts(latent_frames: int, chunk: int, overlap: int, token_drop: int) -> tuple[int, list[int]]:
    """Padding and chunk start indices of ``AutoencoderKLMiniMaxH3._decode_chunks``."""
    num_tokens = latent_frames + token_drop
    pad = (-num_tokens) % chunk
    num_chunks = (num_tokens + pad) // chunk - int(token_drop > 0)
    if num_chunks < 1:
        pad += chunk
        num_chunks = 1
    starts = [index * chunk for index in range(num_chunks)]
    assert starts[-1] + chunk + overlap <= latent_frames + pad, "decode plan does not cover the last chunk"
    return pad, starts


class LatentTileStream(IterableDataset):
    """Endless stream of decoder-input tiles ``[C, chunk + overlap, 16, 16]`` (raw, denormalized latents)."""

    def __init__(self, files: list[str], *, rank: int, world: int, seed: int, tiles_per_clip: int,
                 chunk: int, overlap: int, token_drop: int, latent_mean: torch.Tensor, latent_std: torch.Tensor,
                 normalized: bool, holdout: set[tuple[str, int]]) -> None:
        self.files = files
        self.rank, self.world, self.seed = rank, world, seed
        self.tiles_per_clip = tiles_per_clip
        self.chunk, self.overlap, self.token_drop = chunk, overlap, token_drop
        self.mean, self.std = latent_mean.view(-1, 1, 1, 1), latent_std.view(-1, 1, 1, 1)
        self.normalized = normalized
        self.holdout = holdout

    def __iter__(self) -> Iterator[torch.Tensor]:
        import pyarrow.parquet as pq

        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        stream_id = self.rank * workers + worker
        rng = random.Random(self.seed * 100003 + stream_id)
        files = self.files[stream_id::self.world * workers] or self.files
        while True:
            path = rng.choice(files)
            parquet = pq.ParquetFile(path)
            group_index = rng.randrange(parquet.num_row_groups)
            offset = sum(parquet.metadata.row_group(g).num_rows for g in range(group_index))
            group = parquet.read_row_group(group_index, columns=LATENT_COLUMNS)
            for index, row in enumerate(group.to_pylist()):
                if (path, offset + index) in self.holdout:
                    continue
                yield from self._tiles(row_latent(row), rng)

    def _tiles(self, latent: torch.Tensor, rng: random.Random) -> Iterator[torch.Tensor]:
        if self.normalized:
            latent = latent * self.std + self.mean
        _, frames, height, width = latent.shape
        if height < SPATIAL_TILE_LATENT or width < SPATIAL_TILE_LATENT:
            return
        pad, starts = chunk_starts(frames, self.chunk, self.overlap, self.token_drop)
        if pad:
            latent = torch.cat([latent, latent[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        span = self.chunk + self.overlap
        for _ in range(self.tiles_per_clip):
            start = rng.choice(starts)
            top = rng.randrange(height - SPATIAL_TILE_LATENT + 1)
            left = rng.randrange(width - SPATIAL_TILE_LATENT + 1)
            yield latent[:, start:start + span, top:top + SPATIAL_TILE_LATENT,
                         left:left + SPATIAL_TILE_LATENT].contiguous()


def load_holdout(files: list[str], count: int, normalized: bool, mean: torch.Tensor,
                 std: torch.Tensor) -> tuple[list[torch.Tensor], set[tuple[str, int]]]:
    """Row 0 of ``count`` files spread across the sorted list; excluded from training by (file, row)."""
    import pyarrow.parquet as pq

    stride = max(1, len(files) // max(count, 1))
    clips, held = [], set()
    for path in files[::stride][:count]:
        row = pq.ParquetFile(path).read_row_group(0, columns=LATENT_COLUMNS).slice(0, 1).to_pylist()[0]
        latent = row_latent(row)
        if normalized:
            latent = latent * std.view(-1, 1, 1, 1) + mean.view(-1, 1, 1, 1)
        clips.append(latent.unsqueeze(0))
        held.add((path, 0))
    return clips, held


# ----------------------------------------------------------------------------- models


class DecoderTile(nn.Module):
    """``post_quant_conv`` + ViT decoder: the per-tile computation of ``_decode_clip``."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent_tile: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latent_tile))


def to_unit_pixels(vae: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    return vae.denormalize_pixels(sample.float()).clamp(0, 1)


def build_student(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[str]]:
    vae = load_h3_vae(args.student_vae_dir, device)
    names = convert_decoder_to_nvfp4(vae.decoder,
                                     rotation_group=args.rotation_group or None,
                                     skip_blocks=tuple(args.skip_blocks),
                                     compute_dtype=STUDENT_AUTOCAST)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device)
        vae.decoder.load_state_dict(state["decoder"], strict=True)
    vae.requires_grad_(False)
    for parameter in list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters()):
        parameter.requires_grad_(True)
    vae.decoder.gradient_checkpointing = args.grad_checkpointing
    return vae, names


def invalidate_packed(module: nn.Module) -> None:
    for sub in module.modules():
        if isinstance(sub, NVFP4DecoderLinear):
            sub.invalidate()


# ----------------------------------------------------------------------------- train / eval


def teacher_context(precision: str) -> contextlib.AbstractContextManager:
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def lpips_frames(pixels: torch.Tensor, frame_index: torch.Tensor) -> torch.Tensor:
    """``[B, 3, T, H, W]`` in [0, 1] -> ``[B * k, 3, H, W]`` in [-1, 1] for the given frames."""
    picked = pixels.index_select(2, frame_index)
    return picked.permute(0, 2, 1, 3, 4).flatten(0, 1) * 2 - 1


@torch.no_grad()
def evaluate(student_vae: nn.Module, teacher_videos: list[torch.Tensor], clips: list[torch.Tensor],
             fidelity: Fidelity, device: torch.device) -> dict[str, float]:
    student_vae.eval()
    totals: dict[str, list[float]] = {}
    for clip, reference in zip(clips, teacher_videos, strict=True):
        with torch.autocast(device_type="cuda", dtype=STUDENT_AUTOCAST):
            sample = student_vae.decode(clip.to(device), return_dict=False)[0]
        video = to_unit_pixels(student_vae, sample)
        for key, value in fidelity(video, reference.to(device)).items():
            totals.setdefault(key, []).append(value)
    student_vae.train()
    return {f"eval/{key}": float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def teacher_decodes(teacher_vae: nn.Module, clips: list[torch.Tensor], precision: str,
                    device: torch.device) -> list[torch.Tensor]:
    videos = []
    for clip in clips:
        with teacher_context(precision):
            sample = teacher_vae.decode(clip.to(device), return_dict=False)[0]
        videos.append(to_unit_pixels(teacher_vae, sample).cpu())
    return videos


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, student_vae: nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "decoder": student_vae.decoder.state_dict(),
            "post_quant_conv": student_vae.post_quant_conv.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        }, tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-vae-dir", required=True)
    p.add_argument("--student-vae-dir", required=True)
    p.add_argument("--data-root", action="append", required=True)
    p.add_argument("--latents-normalized", choices=("yes", "no"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rotation-group", type=int, default=0, help="Hadamard group (0 = no rotation)")
    p.add_argument("--skip-blocks", type=int, nargs="*", default=[], help="decoder blocks kept in high precision")
    p.add_argument("--init-checkpoint")
    p.add_argument("--resume", action="store_true", help="resume from output-dir/last.pt if present")
    p.add_argument("--teacher-precision", choices=("fp32", "fp16-autocast"), default="fp16-autocast")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-tiles", type=int, default=4, help="tiles per GPU per step")
    p.add_argument("--tiles-per-clip", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.5)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-clips", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb-project")
    p.add_argument("--run-name")
    return p.parse_args()


def main() -> None:  # noqa: C901 - one linear training script
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output_dir)
    is_main = rank == 0

    teacher_vae = load_h3_vae(args.teacher_vae_dir, device)
    teacher = DecoderTile(teacher_vae).eval().requires_grad_(False)
    student_vae, names = build_student(args, device)
    student = DecoderTile(student_vae).train()
    metadata = {
        **nvfp4_decoder_metadata(names, args.rotation_group or None, tuple(args.skip_blocks)),
        "student_vae_dir": args.student_vae_dir,
        "teacher_vae_dir": args.teacher_vae_dir,
        "student_layers": len(student_vae.decoder.transformer_blocks),
    }
    ddp = DistributedDataParallel(student, device_ids=[local_rank], broadcast_buffers=False)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    start_step = 0
    last_path = out_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        student_vae.decoder.load_state_dict(state["decoder"])
        student_vae.post_quant_conv.load_state_dict(state["post_quant_conv"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]

    import lpips
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    fidelity = Fidelity(device)

    config = teacher_vae.config
    files = list_parquets(args.data_root)
    normalized = args.latents_normalized == "yes"
    holdout_clips, holdout_ids = load_holdout(files, args.eval_clips, normalized, teacher_vae.latents_mean.cpu(),
                                              teacher_vae.latents_std.cpu())
    reference_videos = teacher_decodes(teacher_vae, holdout_clips, args.teacher_precision, device) if is_main else []
    stream = LatentTileStream(files,
                              rank=rank,
                              world=world,
                              seed=args.seed + start_step,
                              tiles_per_clip=args.tiles_per_clip,
                              chunk=teacher_vae.tokens_chunk_size,
                              overlap=teacher_vae.token_overlap,
                              token_drop=config.token_drop,
                              latent_mean=teacher_vae.latents_mean.cpu(),
                              latent_std=teacher_vae.latents_std.cpu(),
                              normalized=normalized,
                              holdout=holdout_ids)
    loader = iter(
        DataLoader(stream, batch_size=args.batch_tiles, num_workers=args.num_workers, pin_memory=True,
                   persistent_workers=args.num_workers > 0))

    run = None
    if is_main and args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=args.run_name,
                         id=args.run_name,
                         resume="allow",
                         job_type="qad",
                         config={
                             **vars(args), "world_size": world,
                             "nvfp4": metadata
                         })
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if start_step == 0:
            baseline = evaluate(student_vae, reference_videos, holdout_clips, fidelity, device)
            print(json.dumps({"step": 0, **baseline}), flush=True)
            if run is not None:
                run.log(baseline, step=0)

    best_lpips = float("inf")
    tic = time.perf_counter()
    for step in range(start_step, args.steps):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tiles = next(loader).to(device, non_blocking=True)
        with torch.no_grad(), teacher_context(args.teacher_precision):
            target = to_unit_pixels(teacher_vae, teacher(tiles))
        with torch.autocast(device_type="cuda", dtype=STUDENT_AUTOCAST):
            prediction = to_unit_pixels(student_vae, ddp(tiles))
        l1 = F.l1_loss(prediction, target)
        frame_index = torch.randperm(prediction.shape[2], device=device)[:LPIPS_FRAMES_PER_TILE]
        perceptual = lpips_net(lpips_frames(prediction, frame_index), lpips_frames(target, frame_index)).mean()
        loss = l1 + args.lpips_weight * perceptual
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        invalidate_packed(student)

        if is_main and (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tic
            tic = time.perf_counter()
            record = {
                "train/loss": loss.item(),
                "train/l1": l1.item(),
                "train/lpips": perceptual.item(),
                "train/psnr": -10 * math.log10(max(F.mse_loss(prediction, target).item(), 1e-12)),
                "train/grad_norm": float(grad_norm),
                "train/lr": lr,
                "perf/step_s": elapsed / args.log_every,
                "perf/tiles_per_s": args.log_every * args.batch_tiles * world / elapsed,
                "perf/peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps({"step": step + 1, **record}), flush=True)
            if run is not None:
                run.log(record, step=step + 1)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            if is_main:
                save_checkpoint(last_path, student_vae, optimizer, step + 1, metadata)
            dist.barrier()
        if is_main and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            metrics = evaluate(student_vae, reference_videos, holdout_clips, fidelity, device)
            print(json.dumps({"step": step + 1, **metrics}), flush=True)
            if run is not None:
                run.log(metrics, step=step + 1)
            if metrics.get("eval/lpips", float("inf")) < best_lpips:
                best_lpips = metrics["eval/lpips"]
                save_checkpoint(out_dir / "best.pt", student_vae, optimizer, step + 1, metadata)
        if (step + 1) % args.eval_every == 0:
            dist.barrier()

    if run is not None:
        run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
