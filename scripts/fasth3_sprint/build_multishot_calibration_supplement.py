#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate an honest multi-shot latent supplement for H3 block scoring.

The inherited VGGSound corpus contains synchronized video/audio examples but no
reliable multi-shot labels. This job uses the pinned released Dense V1 model to
materialize eight explicitly multi-shot T2VA samples, including matching H3
text embeddings, on persistent experiment storage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from safetensors.torch import save_file
import torch

from fastvideo import VideoGenerator
from fastvideo.api import (
    CompileConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY


PROMPTS = (
    "A hard cut from a quiet library clock ticking to a city crosswalk where a bus brakes and pedestrians talk, then a hard cut to rain tapping on a window.",
    "Three distinct shots: ocean waves and gulls, a close-up of a kettle beginning to whistle, then a wide kitchen scene where someone says dinner is ready.",
    "A rapid montage cuts from a train arriving with wheel squeal, to a violinist playing one clear phrase, to an applauding audience in a theater.",
    "First shot: a dog barks in a sunny yard. Cut to a cyclist ringing a bell on a street. Final shot: wind moves trees beside a quiet lake.",
    "A cinematic sequence cuts from thunder over dark clouds, to shoes running through a puddle, to a person indoors saying we made it just in time.",
    "Three-scene video: a basketball bounces in a gym, a whistle triggers a hard cut to cheering fans, then the scene changes to a quiet locker room conversation.",
    "A hard-cut travel montage from an airplane taking off, to a subway door chime and closing doors, to a harbor where a boat horn sounds over waves.",
    "The video changes scenes twice: a drummer plays a short rhythm, cut to dancers clapping in time, then cut to a radio host speaking clearly into a microphone.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="h6-multishot-calibration-supplement")
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--max-prompts", type=int, default=0)
    return parser.parse_args()


def _init_single_process_distributed() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)


def _component(
    name: str,
    model_root: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> Any:
    provider, _architecture = model_index[name][:2]
    return PipelineComponentLoader.load_module(
        module_name=name,
        component_model_path=str(model_root / name),
        transformers_or_diffusers=provider,
        fastvideo_args=fastvideo_args,
    )


def _encode_prompts(model_root: Path, prompts: tuple[str, ...]) -> dict[str, torch.Tensor]:
    _init_single_process_distributed()
    # Dense V1 is a filtered T2VA snapshot: its modular manifest retains a
    # transformer_ref declaration while the unused Ref2VA directory is
    # intentionally absent. Read the pinned manifest directly because this
    # job loads only tokenizer/processor/text_encoder components.
    model_index = json.loads((model_root / "modular_model_index.json").read_text())
    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=str(model_root),
        num_gpus=1,
        inference_mode=True,
        training_mode=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )
    tokenizer = _component("tokenizer", model_root, model_index, fastvideo_args)
    processor = _component("processor", model_root, model_index, fastvideo_args)
    conditioner = _component("text_encoder", model_root, model_index, fastvideo_args)
    stage = MiniMaxH3ConditioningStage(conditioner, tokenizer, processor)
    embeddings: dict[str, torch.Tensor] = {}
    for index, prompt in enumerate(prompts):
        batch = ForwardBatch(data_type="video", prompt=prompt)
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
        batch = stage.forward(batch, fastvideo_args)
        if not batch.prompt_embeds:
            raise RuntimeError(f"No text embedding produced for multi-shot sample {index}")
        embeddings[f"multishot_synth_{index:02d}"] = batch.prompt_embeds[0].squeeze(0).half().cpu().contiguous()
    del stage, conditioner, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def _normalization(model_root: Path, component: str, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    config = json.loads((model_root / component / "config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
    std = torch.tensor(config["latents_std"], dtype=torch.float32)
    if mean.numel() != channels or std.numel() != channels or torch.any(std <= 0):
        raise ValueError(f"Invalid {component} latent normalization contract")
    return mean, std


def _generator(model_root: Path) -> VideoGenerator:
    return VideoGenerator.from_config(
        GeneratorConfig(
            model_path=str(model_root),
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
                compile=CompileConfig(enabled=False, vae_enabled=False),
            ),
            pipeline=PipelineSelection(
                experimental={
                    "attention_backend": "FLASH_ATTN",
                    "output_type": "latent",
                    "inference_torch_compile": False,
                    "vae_parallel_decode": False,
                },
            ),
        ),
    )


def _request(prompt: str, seed: int) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt,
        negative_prompt="",
        sampling=SamplingConfig(
            height=480,
            width=832,
            num_frames=124,
            fps=24,
            num_inference_steps=5,
            guidance_scale=1.0,
            batch_cfg=False,
            seed=seed,
        ),
        output=OutputConfig(save_video=False, return_frames=True),
    )


def main() -> None:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for calibration generation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[:args.max_prompts] if args.max_prompts > 0 else PROMPTS
    if not prompts:
        raise ValueError("The multi-shot calibration prompt set is empty")
    latent_dir = args.output_dir / "latents"
    text_dir = args.output_dir / "text"
    latent_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="calibration-generation",
        config={
            "source_revision": args.source_revision,
            "source_commit": args.source_commit,
            "model_root": str(args.model_root),
            "attention_backend": "FLASH_ATTN",
            "quantization": "none",
            "sample_count": len(prompts),
            "sigma_grid_points": 5,
            "transformer_calls": 4,
            "seed": args.seed,
            "persistent_output": str(args.output_dir),
        },
    )
    started = time.time()
    embeddings = _encode_prompts(args.model_root, prompts)
    generator = _generator(args.model_root)
    video_mean, video_std = _normalization(args.model_root, "vae", 24)
    audio_mean, audio_std = _normalization(args.model_root, "audio_vae", 32)
    records: list[dict[str, Any]] = []
    try:
        for index, prompt in enumerate(prompts):
            sample_id = f"multishot_synth_{index:02d}"
            result = generator.generate(_request(prompt, args.seed + index))
            if result.samples is None or result.audio is None:
                raise RuntimeError(f"Latent generation returned incomplete modalities for {sample_id}")
            video = result.samples.float().cpu()
            audio = result.audio.float().cpu()
            if video.ndim != 5 or video.shape[1] != 24:
                raise ValueError(f"Unexpected video latent shape for {sample_id}: {tuple(video.shape)}")
            if audio.ndim != 3 or tuple(audio.shape[:2]) != (2, 32):
                raise ValueError(f"Unexpected audio latent shape for {sample_id}: {tuple(audio.shape)}")
            video = ((video - video_mean.view(1, 24, 1, 1, 1)) /
                     video_std.view(1, 24, 1, 1, 1)).half().contiguous()
            audio = ((audio - audio_mean.view(1, 32, 1)) /
                     audio_std.view(1, 32, 1)).transpose(1, 2).reshape(-1, 32).half().contiguous()
            text_sha1 = hashlib.sha1(prompt.encode()).hexdigest()
            save_file({"video": video, "audio": audio}, str(latent_dir / f"{sample_id}.safetensors"))
            save_file({"embed": embeddings.pop(sample_id)}, str(text_dir / f"{text_sha1}.safetensors"))
            record = {
                "id": sample_id,
                "caption": prompt,
                "text_sha1": text_sha1,
                "num_latent_frames": int(video.shape[2]),
                "latent_height": int(video.shape[3]),
                "latent_width": int(video.shape[4]),
                "num_audio_latents": int(audio.shape[0] // 2),
                "category": "multiple_shots",
                "provenance": "released_dense_v1_synthetic_multishot",
                "seed": args.seed + index,
            }
            records.append(record)
            run.log({
                "calibration/completed_examples": index + 1,
                "calibration/generation_seconds": float(result.generation_time or 0.0),
                "calibration/video_latent_frames": record["num_latent_frames"],
                "calibration/audio_latents": record["num_audio_latents"],
            }, step=index)
    finally:
        generator.shutdown()

    manifest_path = args.output_dir / "manifest_rank00000.jsonl"
    manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    receipt = {
        "format_version": 1,
        "source_revision": args.source_revision,
        "source_commit": args.source_commit,
        "model_root": str(args.model_root),
        "attention_backend": "FLASH_ATTN",
        "quantization": "none",
        "sample_count": len(records),
        "sample_ids": [record["id"] for record in records],
        "wandb_run": run.url,
        "persistent_manifest": str(manifest_path),
        "elapsed_seconds": time.time() - started,
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    artifact = wandb.Artifact(f"{args.run_id}-receipt", type="calibration-supplement")
    artifact.add_file(str(manifest_path))
    artifact.add_file(str(receipt_path))
    run.log_artifact(artifact)
    run.summary.update(receipt)
    run.finish()


if __name__ == "__main__":
    main()
