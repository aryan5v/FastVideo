# SPDX-License-Identifier: Apache-2.0
"""Encode frozen MiniMax-H3 T2VA rows at their native shape and duration.

Production chunks are homogeneous in exact ``width x height x num_frames``
and land below ``data/bucket=<width>x<height>-<num_frames>f``. A probe writes
only below ``work/probe`` and never creates a production done marker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any
import zlib

AUDIO_SAMPLE_RATE = 32000
AUDIO_PAD_TOLERANCE_S = 0.25


def packed_audio_latent_num_frames(num_frames: int) -> int:
    """Return the H3 packed-audio length on its 40 Hz clock.

    The audio VAE right-pads to its 800-sample hop and emits
    ``ceil(5 * num_frames / 3)`` latents. H3's joint packed sequence instead
    uses the nearest 40 Hz grid point, ``round(5 * num_frames / 3)``. Since
    thirds cannot tie, the latter is exactly this integer expression.
    """
    return (5 * num_frames + 1) // 3


def reconcile_audio_latent_length(audio_latents: Any, num_frames: int):
    """Trim or edge-pad raw audio-VAE output to H3's packed clock."""
    import torch

    target = packed_audio_latent_num_frames(num_frames)
    actual = int(audio_latents.shape[-1])
    if actual > target:
        audio_latents = audio_latents[..., :target]
    elif actual < target:
        if actual == 0:
            raise ValueError("cannot edge-pad an empty audio latent sequence")
        audio_latents = torch.cat(
            [audio_latents, audio_latents[..., -1:].expand(*audio_latents.shape[:-1], target - actual)],
            dim=-1,
        )
    return audio_latents.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/lustre/vlm-k1kong/models/MiniMax-H3"))
    parser.add_argument("--worker-tag", required=True)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--record-id", default=None, help="specific training id for --probe-only")
    parser.add_argument("--stale-minutes", type=float, default=90.0)
    parser.add_argument("--linger-minutes", type=float, default=60.0)
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def apply_env() -> None:
    for name in list(os.environ):
        if "VSA" in name or name == "FASTVIDEO_FA4" or name.startswith("FASTVIDEO_DMD"):
            os.environ.pop(name, None)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def log(tag: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}][{tag}] {message}", flush=True)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def bucket_id(row: dict[str, Any]) -> str:
    return f"{int(row['width'])}x{int(row['height'])}-{int(row['num_frames'])}f"


def load_work(worklist_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    worklist = json.loads(worklist_path.read_text())
    rows = {str(row["conditioning_id"]): row for row in iter_jsonl(Path(worklist["train_manifest"]))}
    if len(rows) != int(worklist["rows"]):
        raise ValueError(f"worklist says {worklist['rows']} rows but train manifest has {len(rows)} unique ids")
    for chunk in worklist["chunks"]:
        shape = chunk["shape"]
        for record_id in chunk["conditioning_ids"]:
            row = rows[record_id]
            actual = (row["width"], row["height"], row["num_frames"])
            expected = (shape["width"], shape["height"], shape["num_frames"])
            if actual != expected:
                raise ValueError(f"{record_id}: worklist shape {expected} != manifest shape {actual}")
    return worklist, rows


def verify_frozen_video(row: dict[str, Any]) -> Path:
    path = Path(row["raw_video_path"])
    stat = path.stat()
    expected_size = int(row["video_size_bytes"])
    expected_mtime_ns = int(row["video_mtime_ns"])
    if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime_ns:
        raise ValueError(
            f"frozen source changed: size/mtime {(stat.st_size, stat.st_mtime_ns)} "
            f"!= {(expected_size, expected_mtime_ns)} for {path}"
        )
    return path


def decode_native_media(row: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    path = verify_frozen_video(row)
    import numpy as np
    import torch
    from fastvideo.pipelines.basic.minimax_h3.reference import (
        _decode_audio_stream,
        _import_av,
        prepare_reference_waveform,
        resample_reference_frames,
    )

    need_frames = int(row["num_frames"])
    target_fps = float(row["fps"])
    av_module = _import_av()
    with av_module.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        rate_value = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate_value is None:
            raise ValueError("video stream has no frame rate")
        source_fps = float(rate_value)
        max_frames = need_frames if abs(source_fps - target_fps) < 1e-6 else math.ceil(
            need_frames * source_fps / target_fps) + 2
        frames = []
        rotation = 0.0
        for frame in container.decode(stream):
            rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
        if not frames:
            raise ValueError("no decoded video frames")
        soundtrack = None
        if container.streams.audio:
            container.seek(0)
            soundtrack = _decode_audio_stream(av_module, container, container.streams.audio[0])
    frames_array = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        frames_array = np.ascontiguousarray(np.rot90(frames_array, k=-turns, axes=(1, 2)))
    frames_array = resample_reference_frames(frames_array, source_fps)[:need_frames]
    expected_shape = (need_frames, int(row["height"]), int(row["width"]), 3)
    if tuple(frames_array.shape) != expected_shape:
        raise ValueError(f"decoded video shape {tuple(frames_array.shape)} != {expected_shape}")
    if soundtrack is None:
        raise ValueError("source has no audio stream")
    waveform, source_sample_rate = soundtrack
    waveform = prepare_reference_waveform(
        waveform,
        int(source_sample_rate),
        AUDIO_SAMPLE_RATE,
        max_duration=need_frames / target_fps,
    )
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError(f"expected stereo waveform [2, samples], got {tuple(waveform.shape)}")
    target_samples = int(need_frames / target_fps * AUDIO_SAMPLE_RATE)
    padded_samples = 0
    if waveform.shape[-1] < target_samples:
        deficit = target_samples - waveform.shape[-1]
        if deficit > AUDIO_PAD_TOLERANCE_S * AUDIO_SAMPLE_RATE:
            raise ValueError(f"audio too short: {waveform.shape[-1]} samples of {target_samples}")
        waveform = torch.nn.functional.pad(waveform, (0, deficit))
        padded_samples = deficit
    waveform = waveform[:, :target_samples].contiguous()
    return frames_array, waveform, {
        "decoded_source_fps": source_fps,
        "audio_source_sample_rate": int(source_sample_rate),
        "audio_samples_used": target_samples,
        "audio_padded_samples": padded_samples,
    }


class Encoders:
    def __init__(self, model_path: Path) -> None:
        from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.models.loader.component_loader import PipelineComponentLoader
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
        from fastvideo.utils import verify_model_config_and_directory

        model_index = verify_model_config_and_directory(str(model_path))
        self.fastvideo_args = FastVideoArgs(
            model_path=str(model_path),
            pipeline_config=MiniMaxH3PipelineConfig(),
            num_gpus=1,
            tp_size=1,
            sp_size=1,
            hsdp_shard_dim=1,
            use_fsdp_inference=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )

        def load(name: str):
            provider, _ = model_index[name][:2]
            return PipelineComponentLoader.load_module(
                module_name=name,
                component_model_path=str(model_path / name),
                transformers_or_diffusers=provider,
                fastvideo_args=self.fastvideo_args,
            )

        self.vae = load("vae")
        self.audio_vae = load("audio_vae")
        self.conditioning = MiniMaxH3ConditioningStage(
            conditioner=load("text_encoder"),
            tokenizer=load("tokenizer"),
            processor=load("processor"),
        )

    def encode_video(self, frames, seed: int):
        import torch

        pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
        pixels = pixels.to(device=torch.device("cuda:0"), dtype=torch.float32).div_(255.0)
        generator = torch.Generator("cpu").manual_seed(seed)
        posterior = self.vae.encode(self.vae.normalize_pixels(pixels)).latent_dist
        return self.vae.normalize_latents(posterior.sample(generator=generator)).squeeze(0).float().cpu().contiguous()

    def encode_audio(self, waveform):
        import torch

        waveform = waveform.to(device=torch.device("cuda:0"), dtype=torch.float32)
        posterior = self.audio_vae.encode(waveform[:, None]).latent_dist
        return self.audio_vae.normalize_latents(posterior.mode()).float().cpu().contiguous()

    def encode_text(self, prompt: str):
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY

        batch = ForwardBatch(data_type="video", prompt=prompt)
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
        batch = self.conditioning.forward(batch, self.fastvideo_args)
        if not batch.prompt_embeds:
            raise RuntimeError("conditioning returned no embedding")
        return batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous()


def expected_shapes(row: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    frames = int(row["num_frames"])
    height = int(row["height"])
    width = int(row["width"])
    if frames % 17 != 5:
        raise ValueError(f"num_frames must be 17*n+5, got {frames}")
    if height % 16 or width % 16:
        raise ValueError(f"source geometry {width}x{height} is not divisible by the H3 VAE spatial ratio 16")
    video_frames = (frames - 5) // 17 * 5 + 2
    return (24, video_frames, height // 16, width // 16), (
        2,
        32,
        packed_audio_latent_num_frames(frames),
    )


def build_record(row: dict[str, Any], video_latents, audio_latents, text_embedding) -> dict[str, Any]:
    record: dict[str, Any] = {"id": row["conditioning_id"]}
    for name, tensor in (
        ("vae_latent", video_latents),
        ("audio_latent", audio_latents),
        ("text_embedding", text_embedding),
    ):
        array = tensor.numpy()
        record[f"{name}_bytes"] = array.tobytes()
        record[f"{name}_shape"] = list(array.shape)
        record[f"{name}_dtype"] = "float32"
    record.update({
        "file_name": Path(row["raw_video_path"]).name,
        "caption": row["prompt"],
        "media_type": "video_with_audio",
        "width": int(row["width"]),
        "height": int(row["height"]),
        "num_frames": int(row["num_frames"]),
        "duration_sec": float(row["duration_sec"]),
        "fps": float(row["fps"]),
        "audio_sample_rate": int(row["audio_sample_rate"]),
    })
    return record


def encode_row(row: dict[str, Any], encoders: Encoders) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    frames, waveform, media_info = decode_native_media(row)
    prep_done = time.monotonic()
    seed = zlib.crc32(str(row["conditioning_id"]).encode()) & 0x7FFFFFFF
    video_latents = encoders.encode_video(frames, seed)
    video_done = time.monotonic()
    audio_latents = encoders.encode_audio(waveform)
    audio_latents = reconcile_audio_latent_length(audio_latents, int(row["num_frames"]))
    audio_done = time.monotonic()
    text_embedding = encoders.encode_text(row["prompt"])
    text_done = time.monotonic()
    expected_video, expected_audio = expected_shapes(row)
    if tuple(video_latents.shape) != expected_video:
        raise ValueError(f"video latent shape {tuple(video_latents.shape)} != {expected_video}")
    if tuple(audio_latents.shape) != expected_audio:
        raise ValueError(f"audio latent shape {tuple(audio_latents.shape)} != {expected_audio}")
    if text_embedding.ndim != 2 or text_embedding.shape[1] != 5120:
        raise ValueError(f"text embedding shape {tuple(text_embedding.shape)} is not [T, 5120]")
    manifest = {
        "conditioning_id": row["conditioning_id"],
        "source": row["source"],
        "bucket": bucket_id(row),
        "raw_video_path": row["raw_video_path"],
        "vae_latent_shape": list(video_latents.shape),
        "audio_latent_shape": list(audio_latents.shape),
        "text_embedding_shape": list(text_embedding.shape),
        "vae_sample_seed": seed,
        **media_info,
        "timing_sec": {
            "media": round(prep_done - started, 3),
            "video_vae": round(video_done - prep_done, 3),
            "audio_vae": round(audio_done - video_done, 3),
            "text": round(text_done - audio_done, 3),
            "total": round(text_done - started, 3),
        },
    }
    return build_record(row, video_latents, audio_latents, text_embedding), manifest


def write_parquet(records: list[dict[str, Any]], path: Path, worker_tag: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2va

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {field: [record[field] for record in records] for field in pyarrow_schema_t2va.names},
        schema=pyarrow_schema_t2va,
    )
    temporary = path.parent / f".{path.name}.{worker_tag}.tmp"
    pq.write_table(table, temporary, compression="zstd", row_group_size=1)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any], worker_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{worker_tag}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_chunk(
    chunk: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    set_root: Path,
    encoders: Encoders,
    args: argparse.Namespace,
) -> None:
    chunk_id = str(chunk["chunk_id"])
    done_path = set_root / "done" / f"{chunk_id}.json"
    if done_path.exists():
        return
    claim_path = set_root / "claims" / chunk_id
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_token = f"{args.worker_tag}:{time.time_ns()}"

    def write_claim_owner() -> None:
        (claim_path / "owner.json").write_text(json.dumps({"token": claim_token}) + "\n")

    def owns_claim() -> bool:
        try:
            return json.loads((claim_path / "owner.json").read_text()).get("token") == claim_token
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def refresh_claim() -> bool:
        if not owns_claim():
            return False
        try:
            os.utime(claim_path)
        except FileNotFoundError:
            return False
        return owns_claim()

    try:
        claim_path.mkdir()
        write_claim_owner()
    except FileExistsError:
        age_minutes = (time.time() - claim_path.stat().st_mtime) / 60.0
        if age_minutes < args.stale_minutes:
            return
        stale_path = set_root / "work" / "stale_claims" / f"{chunk_id}.{args.worker_tag}.{time.time_ns()}"
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(claim_path, stale_path)
        except FileNotFoundError:
            return
        try:
            claim_path.mkdir()
        except FileExistsError:
            return
        write_claim_owner()
        log(args.worker_tag, f"atomically replaced stale claim for {chunk_id} ({age_minutes:.0f} minutes)")

    encoded: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    import torch
    with torch.inference_mode():
        for record_id in chunk["conditioning_ids"]:
            try:
                record, manifest = encode_row(rows[record_id], encoders)
                encoded.append(record)
                manifests.append(manifest)
                if args.timing:
                    log(args.worker_tag, f"{record_id}: {manifest['timing_sec']}")
            except Exception as error:
                failures[record_id] = f"{type(error).__name__}: {error}"
                log(args.worker_tag, f"{record_id} FAILED: {failures[record_id]}")
                if args.timing:
                    traceback.print_exc()

    shape = chunk["shape"]
    bucket = f"{shape['width']}x{shape['height']}-{shape['num_frames']}f"
    if failures:
        # Never publish a partial parquet or a done marker: the recursive
        # loader would sweep that parquet, and the done marker would make the
        # transient failure permanent. Preserve evidence outside data/ and
        # move the claim aside so the whole deterministic chunk can retry.
        failure_path = set_root / "work" / "failures" / f"{chunk_id}.{args.worker_tag}.json"
        write_json(
            failure_path,
            {
                "schema_version": "minimax-h3-native-t2va-retry-v1",
                "chunk_id": chunk_id,
                "bucket": bucket,
                "encoded_before_failure": len(encoded),
                "failures": failures,
                "worker": args.worker_tag,
            },
            args.worker_tag,
        )
        if owns_claim():
            released = set_root / "work" / "failed_claims" / f"{chunk_id}.{args.worker_tag}.{time.time_ns()}"
            released.parent.mkdir(parents=True, exist_ok=True)
            os.rename(claim_path, released)
        log(args.worker_tag, f"{chunk_id}: retry required; {len(failures)} failures, no parquet/done published")
        return

    if not refresh_claim():
        race_path = set_root / "work" / "lost_claims" / f"{chunk_id}.{args.worker_tag}.{time.time_ns()}.json"
        write_json(
            race_path,
            {"chunk_id": chunk_id, "worker": args.worker_tag, "reason": "claim ownership changed before publish"},
            args.worker_tag,
        )
        log(args.worker_tag, f"{chunk_id}: claim ownership changed; discarded encoded rows")
        return
    parquet_path = set_root / "data" / f"bucket={bucket}" / f"{chunk_id}.parquet"
    write_parquet(encoded, parquet_path, args.worker_tag)
    write_json(
        done_path,
        {
            "schema_version": "minimax-h3-native-t2va-done-v1",
            "chunk_id": chunk_id,
            "bucket": bucket,
            "parquet": str(parquet_path),
            "rows": len(encoded),
            "failures": {},
            "rows_manifest": manifests,
            "worker": args.worker_tag,
        },
        args.worker_tag,
    )
    log(args.worker_tag, f"{chunk_id}: {len(encoded)} rows, bucket={bucket}")


def run_probe(worklist: dict[str, Any], rows: dict[str, dict[str, Any]], encoders: Encoders, args: argparse.Namespace) -> None:
    record_id = args.record_id
    if record_id is None:
        record_id = worklist["chunks"][0]["conditioning_ids"][0]
    if record_id not in rows:
        raise KeyError(f"probe id {record_id!r} is not in the training manifest")
    row = rows[record_id]
    import torch
    with torch.inference_mode():
        record, manifest = encode_row(row, encoders)
    set_root = Path(worklist["set_root"])
    path = set_root / "work" / "probe" / f"{record_id}.parquet"
    write_parquet([record], path, args.worker_tag)
    manifest["parquet"] = str(path)
    manifest["probe_only"] = True
    write_json(path.with_suffix(".json"), manifest, args.worker_tag)
    log(args.worker_tag, f"probe passed: {record_id} -> {path}; shapes video={manifest['vae_latent_shape']} "
        f"audio={manifest['audio_latent_shape']} text={manifest['text_embedding_shape']}")


def main() -> None:
    apply_env()
    args = parse_args()
    worklist, rows = load_work(args.worklist)
    if not worklist["chunks"]:
        raise ValueError("worklist is empty")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29600 + abs(zlib.crc32(args.worker_tag.encode())) % 400))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    log(args.worker_tag, "loading H3 video VAE, audio VAE, and Qwen3-VL conditioner")
    encoders = Encoders(args.model_path)
    if args.probe_only:
        run_probe(worklist, rows, encoders, args)
        return
    completed_here = 0
    set_root = Path(worklist["set_root"])
    linger_deadline: float | None = None
    while True:
        progress = 0
        for chunk in worklist["chunks"]:
            if args.limit_chunks is not None and completed_here >= args.limit_chunks:
                return
            before = (set_root / "done" / f"{chunk['chunk_id']}.json").exists()
            process_chunk(chunk, rows, set_root, encoders, args)
            after = (set_root / "done" / f"{chunk['chunk_id']}.json").exists()
            progress += int(after and not before)
            completed_here += int(after and not before)
        incomplete = [
            chunk["chunk_id"]
            for chunk in worklist["chunks"]
            if not (set_root / "done" / f"{chunk['chunk_id']}.json").exists()
        ]
        if not incomplete:
            log(args.worker_tag, "worklist complete")
            return
        if linger_deadline is None:
            linger_deadline = time.monotonic() + args.linger_minutes * 60.0
        if time.monotonic() >= linger_deadline:
            raise RuntimeError(f"linger expired with {len(incomplete)} incomplete chunks")
        if progress == 0:
            time.sleep(60.0)


if __name__ == "__main__":
    main()
