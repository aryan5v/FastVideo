# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 corpus preprocessor: real video+audio+prompt -> H3 training artifacts.

Two phases (SLURM: 1 node, 8 GPUs; ``--smoke`` runs one clip + one prompt):

  Phase A -- latents (per-clip, parallel across GPUs)
    decode clip (torchvision.io.read_video), 5s / 124-frame window (17n+5),
    canvas short edge 512 (multiple of 32), H3 video VAE encode:
    pixels/255 -> normalize_pixels -> encode -> posterior.sample(seed=42)
    -> fp16 -> (x - mean)/std   [exact upstream recipe]
    H3 audio VAE encode: waveform [-1,1] -> encode -> posterior.mode()
    -> transpose(1,2) -> (x - mean)/std (transposed stats)
    -> one .safetensors per clip

  Phase B -- text embeddings (batched, FSDP over the node)
    Qwen3-VL-32B hidden_states[50] (unnormalized), verbatim prompt, no chat
    template. Tokenization + encoder call replicate the upstream H3
    conditioning stage exactly (fastvideo/pipelines/basic/minimax_h3/stages/
    minimax_h3_conditioning.py::_encode_tokens).

Output:
  <out>/latents/<clip_id>.safetensors   video: (1,24,T',H',W') fp16; audio: (2*n,32) fp16
  <out>/text/<sha1>.safetensors         embed: (L,5120) fp16
  <out>/manifest_rank{r}.jsonl          {id, caption, text_sha1, geometry}

Usage (inside the nvcr pytorch container):
  torchrun --nproc_per_node=8 scripts/fasth3/preprocess_h3_corpus.py \
      --manifest <clip_manifest.csv> --out <out> --mode both
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV: clip_id,video,audio,caption")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="both", choices=("latents", "text", "both"))
    parser.add_argument("--smoke", action="store_true", help="one clip + one prompt only")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--short-edge", type=int, default=512)
    parser.add_argument("--max-long-edge", type=int, default=1344)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-root", default="MiniMaxAI/MiniMax-H3")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def read_clip(video_path: str, num_frames: int, short_edge: int, max_long_edge: int, sample_rate: int = 32000,
              seconds: float = 5.0, video_fps: float = 24.0):
    """Decode video+audio. Returns (frames (T,C,H,W) fp32 [0,1], waveform (2,S) fp32 [-1,1]).

    Audio is cropped to the same time window as the retained video frames
    (``num_frames / video_fps`` seconds) and resampled to ``sample_rate``.
    """
    from torchvision.io import read_video

    video, audio, info = read_video(video_path, pts_unit="sec")
    video = video[:num_frames]
    if video.shape[0] < num_frames:
        raise ValueError(f"clip too short: {video.shape[0]} < {num_frames}")
    _, h, w, _ = video.shape
    scale = short_edge / min(h, w)
    new_h = max(32, (round(h * scale) + 15) // 32 * 32)
    new_w = max(32, (round(w * scale) + 15) // 32 * 32)
    while new_h * new_w > short_edge * max_long_edge and new_h > 32 and new_w > 32:
        new_h -= 32
        new_w -= 32
    frames = video.permute(0, 3, 1, 2).float().div_(255.0)
    frames = torch.nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # audio: (C, S) fp32 [-1, 1]; crop to the video window, resample to 32 kHz
    window_seconds = num_frames / video_fps
    n_target = int(seconds * sample_rate)
    if audio is None or audio.numel() == 0:
        waveform = torch.zeros(2, n_target, dtype=torch.float32)
    else:
        rate = float(info.get("audio_fps", 0) or 0)
        data = audio  # (C, S) fp32 [-1,1]
        if rate > 0:
            n_window = int(round(window_seconds * rate))
            data = data[:, :n_window]
            try:
                import torchaudio.functional as taf  # noqa: PLC0415

                data = taf.resample(data, int(rate), sample_rate)
            except Exception:  # noqa: BLE001 - fall back to linear interpolation
                data = torch.nn.functional.interpolate(data[None], size=n_target, mode="linear",
                                                       align_corners=False)[0]
        else:
            data = torch.nn.functional.interpolate(data[None], size=n_target, mode="linear", align_corners=False)[0]
        waveform = data[:2]
        if waveform.shape[0] < 2:
            waveform = torch.cat([waveform, waveform[:1]], dim=0)
        if waveform.shape[1] < n_target:
            waveform = torch.nn.functional.pad(waveform, (0, n_target - waveform.shape[1]))
        else:
            waveform = waveform[:, :n_target]
    return frames, waveform


def read_audio(audio_path: str, sample_rate: int = 32000, seconds: float = 5.0) -> torch.Tensor:
    """Fallback: decode a standalone wav to (2, S) float in [-1, 1]."""
    import wave

    with wave.open(audio_path, "rb") as wf:
        if wf.getnchannels() != 2:
            raise ValueError(f"expected stereo, got {wf.getnchannels()} ch")
        rate = wf.getframerate()
        n = int(seconds * rate)
        data = np.frombuffer(wf.readframes(n), dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    data = torch.from_numpy(data).t().contiguous()  # (2, n)
    if data.shape[1] < n:
        data = torch.nn.functional.pad(data, (0, n - data.shape[1]))
    if rate != sample_rate:
        target = int(seconds * sample_rate)
        data = torch.nn.functional.interpolate(data[None], size=target, mode="linear", align_corners=False)[0]
    return data


def _loader_args(model_root: str) -> object:
    """Real FastVideoArgs for the component loaders (resolves the H3 pipeline
    config via the registry, which the loaders require for config wiring)."""
    from fastvideo.fastvideo_args import FastVideoArgs

    return FastVideoArgs.from_kwargs(
        model_path=model_root,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        training_mode=False,
        inference_mode=True,
        trust_remote_code=False,
        revision="main",
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
    )


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def phase_latents(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> None:
    import safetensors.torch

    from fastvideo.models.loader.component_loader import VAELoader

    out_dir = Path(args.out)
    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.smoke:
        rows = rows[:1]
    rows = rows[rank::world_size]

    vae = VAELoader().load(f"{args.model_root}/vae", _loader_args(args.model_root))
    audio_vae = VAELoader().load(f"{args.model_root}/audio_vae", _loader_args(args.model_root))
    vae.to(device).eval()
    audio_vae.to(device).eval()

    video_mean = vae.latents_mean.detach().float().cpu()
    video_std = vae.latents_std.detach().float().cpu()
    audio_mean = audio_vae.latents_mean.detach().float().cpu().transpose(1, 2)
    audio_std = audio_vae.latents_std.detach().float().cpu().transpose(1, 2)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    done: list[dict] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            try:
                frames, waveform = read_clip(row["video"], args.num_frames, args.short_edge, args.max_long_edge)

                pixels = frames.permute(1, 0, 2, 3)[None].to(device)  # (1, C, T, H, W)
                posterior = vae.encode(vae.normalize_pixels(pixels)).latent_dist
                latents = posterior.sample(generator=generator).to(torch.float16).float().cpu()
                latents = (latents - video_mean[None, :, None, None, None]) / video_std[None, :, None, None, None]

                posterior = audio_vae.encode(waveform.to(device)[:, None]).latent_dist
                audio = posterior.mode().float().cpu().transpose(1, 2)
                audio = ((audio - audio_mean) / audio_std).reshape(-1, audio_vae.latent_channels)

                safetensors.torch.save_file(
                    {"video": latents.to(torch.float16), "audio": audio.to(torch.float16)},
                    str(latents_dir / f"{row['clip_id']}.safetensors"),
                )
                done.append({
                    "id": row["clip_id"],
                    "caption": row["caption"],
                    "text_sha1": hashlib.sha1(row["caption"].encode("utf-8")).hexdigest(),
                    "num_latent_frames": int(latents.shape[2]),
                    "latent_height": int(latents.shape[3]),
                    "latent_width": int(latents.shape[4]),
                    "num_audio_latents": int(audio.shape[0] // 2),
                })
            except Exception as exc:  # noqa: BLE001 - skip bad clips, log them
                print(f"[rank {rank}] clip {row['clip_id']} failed: {exc!r}", flush=True)
            if index % 25 == 0:
                print(f"[rank {rank}] latents {index}/{len(rows)}", flush=True)

    manifest_path = out_dir / f"manifest_rank{rank}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for item in done:
            fh.write(json.dumps(item) + "\n")
    print(f"[rank {rank}] phase A done: {len(done)} clips -> {manifest_path}", flush=True)


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def phase_text(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> None:
    import safetensors.torch

    from fastvideo.models.loader.component_loader import ProcessorLoader, TextEncoderLoader

    out_dir = Path(args.out)
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    conditioner = TextEncoderLoader().load(f"{args.model_root}/text_encoder", _loader_args(args.model_root))
    processor = ProcessorLoader().load(f"{args.model_root}/processor", _loader_args(args.model_root))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    conditioner.eval()

    captions: dict[str, str] = {}
    for path in sorted(out_dir.glob("manifest_rank*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            captions[row["text_sha1"]] = row["caption"]
    if args.smoke:
        captions = dict(list(captions.items())[:1])
    # Each rank encodes its own prompt shard; FSDP wraps the 64 GB encoder
    # across the node, so every rank runs its own forward concurrently.
    caption_items = list(captions.items())[rank::world_size]

    hidden_state_index = 50
    with torch.no_grad():
        for index, (sha1, caption) in enumerate(caption_items):
            token_ids = _token_ids(tokenizer(caption, add_special_tokens=False))
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            mm_token_type_ids = _mm_token_type_ids(processor, token_ids, device)
            outputs = conditioner(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            if outputs.hidden_states is None or len(outputs.hidden_states) <= hidden_state_index:
                raise RuntimeError(f"hidden_states[{hidden_state_index}] unavailable")
            embed = outputs.hidden_states[hidden_state_index][0].float().cpu()
            safetensors.torch.save_file({"embed": embed.to(torch.float16)}, str(text_dir / f"{sha1}.safetensors"))
            if index % 100 == 0:
                print(f"[rank {rank}] text {index}/{len(captions)}", flush=True)
    print(f"[rank {rank}] phase B done: {len(caption_items)} prompts", flush=True)


def _token_ids(tokenized) -> list[int]:
    if isinstance(tokenized, dict):
        return tokenized["input_ids"]
    return tokenized.input_ids


def _mm_token_type_ids(processor, token_ids: list[int], device: torch.device) -> torch.Tensor:
    create_ids = getattr(processor, "create_mm_token_type_ids", None)
    if create_ids is not None:
        try:
            return torch.as_tensor(create_ids([token_ids]), dtype=torch.long, device=device)
        except Exception:  # noqa: BLE001 - fall back to text-only modality ids
            pass
    return torch.zeros((1, len(token_ids)), dtype=torch.long, device=device)


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed)
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    if args.mode in ("latents", "both"):
        phase_latents(args, rank, world_size, device)
    if world_size > 1:
        torch.distributed.barrier()
    if args.mode in ("text", "both"):
        phase_text(args, rank, world_size, device)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
