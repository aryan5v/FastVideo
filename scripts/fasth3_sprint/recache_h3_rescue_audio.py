#!/usr/bin/env python3
"""Build a temporally aligned H3 recovery-pilot artifact subset.

The inherited VGGSound cache already has correct 124-frame/37-latent video
artifacts and text embeddings, but its audio was cropped to 5.0 seconds. This
script preserves those verified video/text artifacts and re-encodes audio for
the full 124/24-second window, which must produce 207 H3 audio latents.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file

NUM_FRAMES = 124
VIDEO_FPS = 24.0
SAMPLE_RATE = 32_000
EXPECTED_VIDEO_LATENTS = 37
EXPECTED_AUDIO_LATENTS = 207


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def select_records(
    *,
    source_root: Path,
    raw_manifest: Path,
    output: Path,
    max_clips: int,
) -> None:
    cached = {record["id"]: record for record in _read_jsonl(sorted(source_root.glob("manifest_rank*.jsonl")))}
    raw = {row["clip_id"]: row for row in csv.DictReader(raw_manifest.open(encoding="utf-8"))}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip_id, record in cached.items():
        row = raw.get(clip_id)
        if row is None or not Path(row["video"]).is_file():
            continue
        if int(record["num_latent_frames"]) != EXPECTED_VIDEO_LATENTS:
            continue
        groups[str(record["caption"])].append({**record, "video": row["video"]})
    for rows in groups.values():
        rows.sort(key=lambda value: value["id"])

    # Reserve one real clip per label as a never-trained rescue validation
    # split. Selection then round-robins labels instead of overrepresenting
    # whichever VGGSound classes happen to sort first.
    holdout: list[dict[str, Any]] = []
    train_groups: dict[str, list[dict[str, Any]]] = {}
    for caption, rows in sorted(groups.items()):
        if len(rows) > 1:
            holdout.append(rows[-1])
            train_groups[caption] = rows[:-1]
        else:
            train_groups[caption] = rows
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < max_clips:
        added = 0
        for caption in sorted(train_groups):
            rows = train_groups[caption]
            if offset < len(rows):
                selected.append(rows[offset])
                added += 1
                if len(selected) == max_clips:
                    break
        if added == 0:
            break
        offset += 1
    if not selected:
        raise RuntimeError("no aligned rescue candidates were selected")

    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selection_train.jsonl"
    holdout_path = output / "selection_holdout.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    holdout_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in holdout))
    receipt = {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "raw_manifest": str(raw_manifest.resolve()),
        "train_count": len(selected),
        "train_caption_count": len({row["caption"] for row in selected}),
        "holdout_count": len(holdout),
        "holdout_caption_count": len({row["caption"] for row in holdout}),
        "selection_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "geometry": {
            "frames": NUM_FRAMES,
            "video_fps": VIDEO_FPS,
            "video_latents": EXPECTED_VIDEO_LATENTS,
            "audio_sample_rate": SAMPLE_RATE,
            "audio_samples": round(NUM_FRAMES / VIDEO_FPS * SAMPLE_RATE),
            "audio_latents": EXPECTED_AUDIO_LATENTS,
        },
    }
    (output / "selection_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _loader_args(model_root: str) -> Any:
    from fastvideo.fastvideo_args import FastVideoArgs

    return FastVideoArgs.from_kwargs(
        model_path=model_root,
        num_gpus=1,
        use_fsdp_inference=False,
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


def _decode_aligned_audio(path: str) -> torch.Tensor:
    from torchvision.io import read_video

    _video, audio, info = read_video(path, pts_unit="sec")
    target_samples = round(NUM_FRAMES / VIDEO_FPS * SAMPLE_RATE)
    if audio is None or audio.numel() == 0:
        raise ValueError("clip has no audio")
    rate = int(float(info.get("audio_fps", 0) or 0))
    if rate <= 0:
        raise ValueError("clip has no audio sample rate")
    source_samples = round(NUM_FRAMES / VIDEO_FPS * rate)
    audio = audio[:2, :source_samples].float()
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    try:
        import torchaudio.functional as taf

        audio = taf.resample(audio, rate, SAMPLE_RATE)
    except Exception:
        audio = torch.nn.functional.interpolate(audio[None], size=target_samples, mode="linear", align_corners=False)[0]
    if audio.shape[1] < target_samples:
        audio = torch.nn.functional.pad(audio, (0, target_samples - audio.shape[1]))
    return audio[:, :target_samples].contiguous()


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source.resolve(), destination)


def recache(
    *,
    selection: Path,
    source_root: Path,
    output: Path,
    model_root: str,
) -> None:
    from fastvideo.models.loader.component_loader import AudioDecoderLoader

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    records = _read_jsonl([selection])[rank::world]
    latent_dir = output / "latents"
    text_dir = output / "text"
    latent_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    audio_vae = AudioDecoderLoader().load(f"{model_root}/audio_vae", _loader_args(model_root)).to(device).eval()
    mean = audio_vae.latents_mean.detach().float().cpu().transpose(1, 2)
    std = audio_vae.latents_std.detach().float().cpu().transpose(1, 2)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            sample_id = str(record["id"])
            destination = latent_dir / f"{sample_id}.safetensors"
            try:
                old = load_file(str(source_root / "latents" / f"{sample_id}.safetensors"), device="cpu")
                video = old["video"]
                if tuple(video.shape[:3]) != (1, 24, EXPECTED_VIDEO_LATENTS):
                    raise ValueError(f"bad retained video shape {tuple(video.shape)}")
                waveform = _decode_aligned_audio(str(record["video"])).to(device)
                posterior = audio_vae.encode(waveform[:, None]).latent_dist
                audio = posterior.mode().float().cpu().transpose(1, 2)
                audio = ((audio - mean) / std).reshape(-1, audio_vae.latent_channels)
                if tuple(audio.shape) != (2 * EXPECTED_AUDIO_LATENTS, 32):
                    raise ValueError(f"aligned audio encoded to {tuple(audio.shape)}, expected (414, 32)")
                save_file({"video": video.half().contiguous(), "audio": audio.half().contiguous()}, str(destination))
                text_sha1 = str(record["text_sha1"])
                _link_or_copy(source_root / "text" / f"{text_sha1}.safetensors", text_dir / f"{text_sha1}.safetensors")
                completed.append({
                    **{key: record[key] for key in ("id", "caption", "text_sha1", "latent_height", "latent_width")},
                    "num_latent_frames": EXPECTED_VIDEO_LATENTS,
                    "num_audio_latents": EXPECTED_AUDIO_LATENTS,
                    "source_video": str(record["video"]),
                    "temporal_seconds": NUM_FRAMES / VIDEO_FPS,
                })
            except Exception as error:  # keep other ranks useful, but fail the final gate
                failures.append({"id": sample_id, "error": repr(error)})
            if index % 10 == 0:
                print(f"rank={rank} progress={index}/{len(records)} complete={len(completed)} failures={len(failures)}", flush=True)
    (output / f"manifest_rank{rank}.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in completed))
    (output / f"failures_rank{rank}.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in failures))
    print(f"rank={rank} finished complete={len(completed)} failures={len(failures)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--source-root", type=Path, required=True)
    select.add_argument("--raw-manifest", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--max-clips", type=int, default=512)
    encode = subparsers.add_parser("recache")
    encode.add_argument("--selection", type=Path, required=True)
    encode.add_argument("--source-root", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--model-root", required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_records(source_root=args.source_root, raw_manifest=args.raw_manifest, output=args.output, max_clips=args.max_clips)
    else:
        recache(selection=args.selection, source_root=args.source_root, output=args.output, model_root=args.model_root)


if __name__ == "__main__":
    main()
