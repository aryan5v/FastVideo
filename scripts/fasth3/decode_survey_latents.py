# SPDX-License-Identifier: Apache-2.0
"""Decode MLX-survey latents with torch VAEs on a CUDA node -> mp4 eyeballs.

Pairs with fastvideo/benchmarks/mlx_h3_quant_survey.py: the survey dumps the
denoised (normalized) video/audio latents per quant format; this script
denormalizes and decodes them through the H3 video/audio VAEs so the bake-off
can be judged by eye without an MLX VAE port.

Usage (cluster, 1 node):
    python scripts/fasth3/decode_survey_latents.py \\
        --latents-dir ~/survey_latents --vae-root <H3 snapshot> --out ~/survey_mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import AudioDecoderLoader, VAELoader

FPS = 24
AUDIO_FPS = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--vae-root", required=True, help="H3 snapshot (vae/ + audio_vae/)")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _loader_args(model_root: str) -> FastVideoArgs:
    return FastVideoArgs.from_kwargs(
        model_path=model_root, num_gpus=1, use_fsdp_inference=True,
        training_mode=False, inference_mode=True, trust_remote_code=False, revision="main",
        hsdp_replicate_dim=1, hsdp_shard_dim=1, vae_cpu_offload=False, dit_cpu_offload=False)


def decode_video(vae: torch.nn.Module, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(1, C, T, H, W) normalized -> (T, H, W, 3) uint8."""
    latents = latents.to(device)
    z = latents * vae.latents_std.to(device) + vae.latents_mean.to(device)
    with torch.no_grad():
        decoded = vae.decode(z).sample  # (1, 3, T, H, W) in [0, 1]
    frames = (decoded[0].permute(1, 2, 3, 0).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    return frames


def decode_audio(audio_vae: torch.nn.Module, audio_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(2n, 32) normalized rows -> (2, S) waveform in [-1, 1]."""
    rows = audio_rows.to(device)
    latents = rows.reshape(2, -1, 32).transpose(1, 2)  # (2, 32, n)
    mean = audio_vae.latents_mean.detach().to(device).transpose(1, 2)
    std = audio_vae.latents_std.detach().to(device).transpose(1, 2)
    z = latents * std + mean
    with torch.no_grad():
        waveform = audio_vae.decode(z).sample  # (2, 1, S)
    return waveform[:, 0].clamp(-1, 1).cpu()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = VAELoader().load(f"{args.vae_root}/vae", _loader_args(args.vae_root)).to(device).eval()
    audio_vae = AudioDecoderLoader().load(f"{args.vae_root}/audio_vae",
                                         _loader_args(args.vae_root)).to(device).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from torchvision.io import write_video

    for path in sorted(Path(args.latents_dir).glob("latents_*.npz")):
        data = np.load(path, allow_pickle=True)
        layout = data["layout"].item()
        video_latents = torch.from_numpy(data["video_rows"]).reshape(
            1, 24, layout["num_latent_frames"], layout["latent_height"], layout["latent_width"])
        audio_rows = torch.from_numpy(data["audio_rows"])

        frames = decode_video(vae, video_latents, device)
        waveform = decode_audio(audio_vae, audio_rows, device)
        out_path = out_dir / f"{path.stem}.mp4"
        write_video(
            str(out_path), torch.from_numpy(frames), fps=FPS,
            video_codec="libx264", options={"crf": "18", "pix_fmt": "yuv420p"},
            audio_array=waveform.unsqueeze(0), audio_fps=AUDIO_FPS, audio_codec="aac",
        )
        print(f"wrote {out_path} ({frames.shape[0]} frames, {waveform.shape[1]/AUDIO_FPS:.1f}s audio)", flush=True)
    print("DECODE DONE")


if __name__ == "__main__":
    main()