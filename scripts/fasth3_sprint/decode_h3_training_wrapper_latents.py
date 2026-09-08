#!/usr/bin/env python3
"""Decode terminal H3 training-wrapper latents into verified AV MP4 files."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.train.entrypoint.dcp_to_diffusers import _ensure_distributed, _run_config_from_raw
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.train.utils.validation_media import write_validation_mp4


def _rgb_frames(pixels: torch.Tensor) -> list[np.ndarray]:
    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(f"decoded pixels must have shape [1, 3, T, H, W], got {tuple(pixels.shape)}")
    array = pixels[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return [np.ascontiguousarray(frame) for frame in array]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    rollout = json.loads((input_dir / "rollout_receipt.json").read_text())
    metadata = json.loads((Path(args.source_checkpoint).resolve() / "metadata.json").read_text())
    raw = copy.deepcopy(metadata["config"])
    raw["training"]["distributed"].update({
        "num_gpus": 1,
        "sp_size": 1,
        "tp_size": 1,
        "hsdp_replicate_dim": 1,
        "hsdp_shard_dim": 1,
    })
    raw["training"]["dit_precision"] = "bf16"
    cfg = _run_config_from_raw(raw)
    tc = cfg.training
    _ensure_distributed()
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    model_path = str(Path(args.model_path).resolve())
    vae = load_module_from_path(model_path=model_path, module_type="vae", training_config=tc)
    audio_vae = load_module_from_path(model_path=model_path, module_type="audio_vae", training_config=tc)
    device = torch.device("cuda")
    vae.to(device)
    audio_vae.to(device)

    decoded: list[dict[str, object]] = []
    for item in rollout["samples"]:
        name = item["name"]
        tensors = load_file(str(input_dir / f"{name}.safetensors"), device="cpu")
        video = tensors["video"].permute(0, 2, 1, 3, 4).to(device)
        video = vae.denormalize_latents(video.float())
        pixels = torch.empty(vae.decoded_pixel_shape(video.shape), dtype=torch.float32, device="cpu")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            vae.decode_to_pixels(video, pixels)

        audio = audio_vae.denormalize_latents(tensors["audio"][0].to(device).float())
        with torch.inference_mode():
            waveform = audio_vae.decode(audio).sample.float()
        if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (2, 1):
            raise ValueError(f"decoded audio must have shape [2, 1, samples], got {tuple(waveform.shape)}")
        waveform = waveform[:, 0].transpose(0, 1).contiguous().cpu()
        destination = input_dir / f"{name}.mp4"
        write_validation_mp4(
            str(destination),
            _rgb_frames(pixels),
            fps=args.fps,
            audio=waveform,
            audio_sample_rate=int(audio_vae.sampling_rate),
        )
        decoded.append({
            "name": name,
            "path": str(destination),
            "frames": int(pixels.shape[2]),
            "audio_samples": int(waveform.shape[0]),
            "audio_sample_rate": int(audio_vae.sampling_rate),
        })
    receipt = {
        "passed": len(decoded) == len(rollout["samples"]),
        "rollout_receipt": str(input_dir / "rollout_receipt.json"),
        "decoded": decoded,
    }
    (input_dir / "decode_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
