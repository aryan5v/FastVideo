# SPDX-License-Identifier: Apache-2.0
"""Speed and fidelity of MiniMax-H3 video decoders on the same latents.

Every variant decodes identical latents. Fidelity is measured against the full
36-layer decoder run in FP32 without autocast (``full_fp32``), and against the
source pixels when the latents come from ``--video``. Timing covers the decoder
call only, after one warmup, synchronized on the device.

Variants:
  full_fp32     full VAE, FP32 weights, no autocast (fidelity reference)
  full_prod     full VAE, FP32 weights, fp16 autocast (what the H3 decode stage runs)
  full_fp16w    full VAE, decoder linears stored fp16, fp16 autocast
  light_prod    26-layer light VAE, dense weights, fp16 autocast
  light_int8    light VAE with the INT8 ConvRot overlay (V2 / Trim release decode)
  light_fp16w   light VAE, decoder linears stored fp16, fp16 autocast
  taeh3         TAEH3 preview decoder

Example:
  python scripts/benchmarks/minimax_h3_vae/bench_decoder.py \\
      --full-vae-dir $FULL/vae --light-vae-dir $V2/vae --video assets/videos/robot_pouring.mp4 \\
      --variants full_fp32,full_prod,light_prod,light_int8,taeh3 --tile-batch 1,12
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REFERENCE = "full_fp32"
LPIPS_FRAME_STRIDE = 4


def load_h3_vae(vae_dir: str, device: torch.device, *, int8_overlay: bool = False) -> nn.Module:
    from safetensors.torch import load_file

    from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from fastvideo.models.vaes.minimax_h3_int8_convrot import (
        dense_vae_safetensors,
        find_int8_convrot_vae_path,
        overlay_minimax_h3_int8_convrot_decoder,
    )
    from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

    config = MiniMaxH3VideoVAEConfig()
    with open(os.path.join(vae_dir, "config.json")) as handle:
        config.update_model_arch(json.load(handle))
    vae = AutoencoderKLMiniMaxH3(config).to(device)
    state: dict[str, torch.Tensor] = {}
    for path in dense_vae_safetensors(sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")))):
        state.update(load_file(path))
    vae.load_state_dict(state, strict=True)
    del state
    if int8_overlay:
        overlay_path = find_int8_convrot_vae_path(vae_dir)
        if overlay_path is None:
            raise FileNotFoundError(f"No INT8 ConvRot overlay next to {vae_dir}")
        overlay_minimax_h3_int8_convrot_decoder(vae, overlay_path)
    return vae.eval().requires_grad_(False)


def store_decoder_linears_fp16(vae: nn.Module) -> None:
    """Keep decoder GEMM weights in fp16 so autocast does not recast them per call."""
    for module in vae.decoder.modules():
        if type(module) is nn.Linear:
            nn.Module.to(module, dtype=torch.float16)


def read_video(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Return ``[1, 3, T, H, W]`` uint8 on CPU, center-cropped and resized."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    available = len(decoder)
    if available < num_frames:
        raise ValueError(f"{path} has {available} frames, need {num_frames}")
    frames = decoder.get_frames_in_range(0, num_frames).data  # T, C, H, W uint8
    _, _, src_h, src_w = frames.shape
    scale = max(height / src_h, width / src_w)
    resized = F.interpolate(frames.float(),
                            size=(round(src_h * scale), round(src_w * scale)),
                            mode="bilinear",
                            antialias=True,
                            align_corners=False)
    top = (resized.shape[-2] - height) // 2
    left = (resized.shape[-1] - width) // 2
    cropped = resized[..., top:top + height, left:left + width].clamp(0, 255).round().to(torch.uint8)
    return cropped.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def timed(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[torch.Tensor, list[float]]:
    out = fn()  # warmup
    times = []
    for _ in range(repeats):
        del out
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return out, times


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float(), b.float()).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def frames_for_metrics(video: torch.Tensor) -> torch.Tensor:
    """``[1, 3, T, H, W]`` in [0, 1] -> ``[T', 3, H, W]`` every ``LPIPS_FRAME_STRIDE`` frames."""
    return video[0, :, ::LPIPS_FRAME_STRIDE].permute(1, 0, 2, 3)


class Fidelity:

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except ImportError:
            self.lpips = None
        try:
            from pytorch_msssim import ssim
            self.ssim = ssim
        except ImportError:
            self.ssim = None

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        if pred.shape != target.shape:
            return {"shape_mismatch": 1.0}
        result = {"psnr": psnr(pred, target)}
        p, t = frames_for_metrics(pred).float(), frames_for_metrics(target).float()
        if self.ssim is not None:
            result["ssim"] = float(self.ssim(p, t, data_range=1.0).item())
        if self.lpips is not None:
            result["lpips"] = float(self.lpips(p * 2 - 1, t * 2 - 1).mean().item())
        return result


def decode_fn(vae: nn.Module, z: torch.Tensor, *, autocast: bool) -> Callable[[], torch.Tensor]:
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if autocast else contextlib.nullcontext())

    @torch.no_grad()
    def run() -> torch.Tensor:
        with ctx:
            sample = vae.decode(z, return_dict=False)[0]
        return vae.denormalize_pixels(sample.float()).clamp_(0, 1)

    return run


def taeh3_fn(vae: nn.Module, z: torch.Tensor) -> Callable[[], torch.Tensor]:
    from fastvideo.models.vaes.minimax_h3_taeh3 import decode_ncthw_latents_taeh3

    normalized = vae.normalize_latents(z)

    @torch.no_grad()
    def run() -> torch.Tensor:
        return decode_ncthw_latents_taeh3(normalized, device=z.device).float()

    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-vae-dir", required=True, help="vae/ folder of the full 36-layer H3 VAE")
    parser.add_argument("--light-vae-dir", help="vae/ folder of the 26-layer light VAE (+ INT8 overlay)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="encode this clip with the full VAE encoder")
    source.add_argument("--latents", help="raw (denormalized) NCTHW latents saved with torch.save")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--variants", default="full_fp32,full_prod,full_fp16w,light_prod,light_int8,taeh3")
    parser.add_argument("--tile-batch", default="1", help="comma list of FASTVIDEO_H3_VAE_TILE_BATCH values")
    parser.add_argument("--overlap", default="64", help="comma list of spatial tile overlaps in pixels")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--wandb-project", help="log every row and a summary table to this W&B project")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def flatten(row: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}/"))
        elif not isinstance(value, list):
            flat[f"{prefix}{key}"] = value
    return flat


def build_variant(name: str, args: argparse.Namespace, device: torch.device,
                  cache: dict[str, nn.Module]) -> tuple[nn.Module, bool]:
    """Return (vae, autocast) for a decoder variant; VAEs are cached by weight set."""
    family, _, mode = name.partition("_")
    if family == "full":
        key, vae_dir, int8 = "full", args.full_vae_dir, False
    elif family == "light":
        if not args.light_vae_dir:
            raise ValueError(f"{name} needs --light-vae-dir")
        key, vae_dir, int8 = ("light_int8" if mode == "int8" else "light"), args.light_vae_dir, mode == "int8"
    else:
        raise ValueError(f"Unknown variant {name}")
    if mode == "fp16w":
        key += "_fp16w"
    if key not in cache:
        vae = load_h3_vae(vae_dir, device, int8_overlay=int8)
        if mode == "fp16w":
            store_decoder_linears_fp16(vae)
        cache[key] = vae
    return cache[key], mode != "fp32"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, nn.Module] = {}
    reference_vae, _ = build_variant(REFERENCE, args, device, cache)

    source_pixels = None
    if args.video:
        pixels = read_video(args.video, args.num_frames, args.height, args.width)
        with torch.no_grad():
            z = reference_vae.encode_pixels(pixels).latent_dist.mode()
        source_pixels = pixels.float().div(255).to(device)
        torch.save(z.cpu(), out_dir / "latents.pt")
    else:
        z = torch.load(args.latents, map_location="cpu")
    z = z.to(device=device, dtype=torch.float32)
    print(f"latents {tuple(z.shape)}", flush=True)

    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, job_type="decoder-bench", config=vars(args))

    fidelity = Fidelity(device)
    rows = []
    reference = None
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if REFERENCE in variants:
        variants.remove(REFERENCE)
    variants.insert(0, REFERENCE)
    for name in variants:
        for tile_batch in (int(v) for v in args.tile_batch.split(",")):
            for overlap in (int(v) for v in args.overlap.split(",")):
                if name == "taeh3" and (tile_batch, overlap) != (1, 64):
                    continue
                os.environ["FASTVIDEO_H3_VAE_TILE_BATCH"] = str(tile_batch)
                if name == "taeh3":
                    fn = taeh3_fn(reference_vae, z)
                else:
                    vae, autocast = build_variant(name, args, device, cache)
                    vae.tile_sample_min_overlap_height = overlap
                    vae.tile_sample_min_overlap_width = overlap
                    fn = decode_fn(vae, z, autocast=autocast)
                torch.cuda.reset_peak_memory_stats()
                video, times = timed(fn, args.repeats)
                row = {
                    "variant": name,
                    "tile_batch": tile_batch,
                    "overlap": overlap,
                    "median_s": sorted(times)[len(times) // 2],
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "shape": list(video.shape),
                }
                if reference is None:
                    reference = video
                else:
                    row["vs_full_fp32"] = fidelity(video, reference)
                if source_pixels is not None:
                    row["vs_source"] = fidelity(video, source_pixels)
                if args.save_videos:
                    torch.save((video * 255).round().to(torch.uint8).cpu(),
                               out_dir / f"{name}_tb{tile_batch}_ov{overlap}.pt")
                rows.append(row)
                print(json.dumps(row), flush=True)
                if run is not None:
                    run.log(flatten(row))
                del video
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    if run is not None:
        flat_rows = [flatten(row) for row in rows]
        columns = sorted({key for row in flat_rows for key in row})
        run.log({"results": wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in flat_rows])})
        run.finish()


if __name__ == "__main__":
    main()
