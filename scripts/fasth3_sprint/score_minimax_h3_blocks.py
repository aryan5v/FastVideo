#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score MiniMax H3 blocks on a stratified, precomputed T2VA calibration set.

Each process owns one deterministic shard. Every example runs one full dense
forward plus one identity ablation per main block. Forward hooks also measure
the residual change on video/audio rows and the change in their pooled
cross-modal representation. The resulting partial is small and mergeable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any

import torch
from safetensors.torch import load_file

from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
)


CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "multiple_shots": re.compile(r"\b(scene|shot|transition|camera cut|hard cut)\b", re.I),
    "speech": re.compile(r"\b(speaking|speech|talking|conversation|singing|dialogue|voice)\b", re.I),
    "music": re.compile(r"\b(music|piano|violin|guitar|drum|orchestra|horn|cymbal)\b", re.I),
    "motion": re.compile(r"\b(running|driving|racing|moving|dancing|marching|riding|flying)\b", re.I),
    "sound_event": re.compile(r"\b(explosion|barking|cawing|engine|hammer|sawing|gunshot|impact|alarm)\b", re.I),
}
CATEGORY_QUOTAS = {
    "multiple_shots": 8,
    "speech": 62,
    "music": 62,
    "motion": 62,
    "sound_event": 62,
}
BASE_NOISE_BY_STRATUM = {
    "high": 0.9,
    "middle": 0.5,
    "low": 0.1,
}
METRIC_NAMES = (
    "ablation_video",
    "ablation_audio",
    "ablation_total",
    "residual_video",
    "residual_audio",
    "cross_modal_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--max-records-per-shard", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete-calibration",
        action="store_true",
        help="Smoke-test only: allow category shortages instead of claiming the locked 256-example set.",
    )
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    return parser.parse_args()


def _category(caption: str) -> str | None:
    for name, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(caption):
            return name
    return None


def _calibration_records(
    corpus_root: Path,
    seed: int,
    *,
    supplement_root: Path | None = None,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORY_QUOTAS}
    seen: set[str] = set()
    roots = [corpus_root]
    if supplement_root is not None:
        roots.append(supplement_root)
    for root in roots:
        for manifest in sorted(root.glob("manifest_rank*.jsonl")):
            for line in manifest.read_text().splitlines():
                record = json.loads(line)
                sample_id = str(record["id"])
                if sample_id in seen:
                    continue
                if record.get("provenance") == "released_dense_v1_synthetic_multishot":
                    category = "multiple_shots"
                else:
                    category = _category(str(record.get("caption", "")))
                    if category == "multiple_shots":
                        # VGGSound's label "shot football" is not evidence of
                        # scene cuts and must never populate the multi-shot gate.
                        category = None
                if category is None:
                    continue
                latent_path = root / "latents" / f"{sample_id}.safetensors"
                text_path = root / "text" / f"{record['text_sha1']}.safetensors"
                if not latent_path.is_file() or not text_path.is_file():
                    continue
                seen.add(sample_id)
                record["category"] = category
                record["_corpus_root"] = str(root)
                candidates[category].append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category, quota in CATEGORY_QUOTAS.items():
        records = sorted(candidates[category], key=lambda item: str(item["id"]))
        rng.shuffle(records)
        if len(records) < quota and not allow_incomplete:
            raise RuntimeError(f"Calibration category {category!r} has {len(records)} records; need {quota}")
        selected.extend(records[:quota])
    selected.sort(key=lambda item: str(item["id"]))
    expected_count = sum(CATEGORY_QUOTAS.values())
    if not allow_incomplete and (
        len(selected) != expected_count or len({item["id"] for item in selected}) != expected_count
    ):
        raise RuntimeError(
            f"The locked block-scoring calibration set must contain {expected_count} unique examples",
        )
    if len(selected) != len({item["id"] for item in selected}):
        raise RuntimeError("The block-scoring calibration set contains duplicate examples")
    return selected


def _loader_args(model_root: Path) -> Any:
    from fastvideo.fastvideo_args import FastVideoArgs

    return FastVideoArgs.from_kwargs(
        model_path=str(model_root),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        training_mode=False,
        inference_mode=True,
        trust_remote_code=False,
        revision="main",
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        attention_backend="TORCH_SDPA",
    )


def _load_transformer(model_root: Path, device: torch.device) -> torch.nn.Module:
    from fastvideo.models.loader.component_loader import TransformerLoader

    transformer = TransformerLoader().load(str(model_root / "transformer"), _loader_args(model_root))
    return transformer.to(device).eval()


def _shift(base_noise: float, shift: float) -> float:
    return shift * base_noise / (1.0 + (shift - 1.0) * base_noise)


def _inputs_for_record(
    transformer: torch.nn.Module,
    record: dict[str, Any],
    *,
    stratum: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Any]:
    corpus_root = Path(record["_corpus_root"])
    latents = load_file(str(corpus_root / "latents" / f"{record['id']}.safetensors"), device="cpu")
    text = load_file(str(corpus_root / "text" / f"{record['text_sha1']}.safetensors"), device="cpu")
    clean_video = latents["video"].to(device=device, dtype=torch.bfloat16)
    clean_audio = latents["audio"].to(device=device, dtype=torch.bfloat16)
    text_rows = text["embed"][None].to(device=device, dtype=torch.bfloat16)
    base_noise = BASE_NOISE_BY_STRATUM[stratum]
    video_sigma = _shift(base_noise, 12.0)
    audio_sigma = _shift(base_noise, 3.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    noisy_video = (1.0 - video_sigma) * clean_video + video_sigma * torch.randn(
        clean_video.shape, generator=generator, device=device, dtype=clean_video.dtype)
    noisy_audio = (1.0 - audio_sigma) * clean_audio + audio_sigma * torch.randn(
        clean_audio.shape, generator=generator, device=device, dtype=clean_audio.dtype)

    _, _, frames, height, width = clean_video.shape
    num_audio_latents = int(clean_audio.shape[0] // 2)
    text_tags = torch.full((text_rows.shape[1], ), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    layout = build_packed_sequence(
        text_tags,
        frames,
        height,
        width,
        num_audio_latents,
        transformer.patch_size,
    )
    unique, inverse = build_row_timesteps(
        layout,
        video_timestep=1.0 - video_sigma,
        audio_timestep=1.0 - audio_sigma,
        condition_video_timestep=1.0 - video_sigma,
        condition_audio_timestep=1.0 - audio_sigma,
    )
    inputs = {
        "hidden_states": patchify_video_latents(noisy_video, transformer.patch_size)[None],
        "audio_hidden_states": noisy_audio[None],
        "encoder_hidden_states": text_rows,
        "timestep": unique.to(device=device, dtype=torch.float32),
        "timestep_indices": inverse.to(device=device, dtype=torch.long),
        "token_tags": layout.token_tags.to(device=device, dtype=torch.long),
        "position_ids": layout.position_ids.to(device=device, dtype=torch.float32),
        "video_indices": layout.video_indices.to(device=device, dtype=torch.long),
        "audio_indices": layout.audio_indices.to(device=device, dtype=torch.long),
        "text_indices": layout.text_indices.to(device=device, dtype=torch.long),
    }
    return inputs, layout


def _normalized_mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (value.float() - reference.float()).square().mean()
    denominator = reference.float().square().mean().clamp_min(1e-8)
    return float((numerator / denominator).item())


def _representation_hooks(
    transformer: torch.nn.Module,
    layout: Any,
    sink: list[dict[str, float] | None],
    num_blocks: int,
) -> list[Any]:
    device = next(transformer.parameters()).device
    video_indices = layout.video_indices.to(device)
    audio_indices = layout.audio_indices.to(video_indices.device)

    def make_hook(index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            before = inputs[0]
            delta = output - before
            before_video = before.index_select(1, video_indices).float()
            before_audio = before.index_select(1, audio_indices).float()
            delta_video = delta.index_select(1, video_indices).float()
            delta_audio = delta.index_select(1, audio_indices).float()
            video_energy = before_video.square().mean().clamp_min(1e-8)
            audio_energy = before_audio.square().mean().clamp_min(1e-8)
            before_cross = torch.nn.functional.cosine_similarity(
                before_video.mean(dim=1), before_audio.mean(dim=1), dim=-1).mean()
            after_cross = torch.nn.functional.cosine_similarity(
                output.index_select(1, video_indices).float().mean(dim=1),
                output.index_select(1, audio_indices).float().mean(dim=1),
                dim=-1,
            ).mean()
            sink[index] = {
                "residual_video": float((delta_video.square().mean() / video_energy).item()),
                "residual_audio": float((delta_audio.square().mean() / audio_energy).item()),
                "cross_modal_change": float((after_cross - before_cross).abs().item()),
            }

        return hook

    return [
        block.register_forward_hook(make_hook(index))
        for index, block in enumerate(transformer.transformer_blocks[:num_blocks])
    ]


def _forward(transformer: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), set_forward_context(
            current_timestep=inputs["timestep"], attn_metadata=None):
        return transformer(**inputs)


def _score_record(
    transformer: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    layout: Any,
    num_blocks: int,
) -> list[dict[str, float]]:
    representation: list[dict[str, float] | None] = [None] * num_blocks
    hooks = _representation_hooks(transformer, layout, representation, num_blocks)
    reference_video, reference_audio = _forward(transformer, inputs)
    for handle in hooks:
        handle.remove()

    scores: list[dict[str, float]] = []
    for block_index, block in enumerate(transformer.transformer_blocks[:num_blocks]):
        original_forward = block.forward

        def identity(hidden_states: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            return hidden_states

        block.forward = identity  # type: ignore[method-assign]
        try:
            ablated_video, ablated_audio = _forward(transformer, inputs)
        finally:
            block.forward = original_forward  # type: ignore[method-assign]
        video_drift = _normalized_mse(ablated_video, reference_video)
        audio_drift = _normalized_mse(ablated_audio, reference_audio)
        representation_score = representation[block_index]
        if representation_score is None:
            raise RuntimeError(f"Block {block_index} representation hook did not run")
        scores.append({
            "ablation_video": video_drift,
            "ablation_audio": audio_drift,
            "ablation_total": video_drift + audio_drift,
            **representation_score,
        })
    return scores


def _empty_accumulator(num_blocks: int) -> list[dict[str, float]]:
    return [{metric: 0.0 for metric in METRIC_NAMES} for _ in range(num_blocks)]


def _add_scores(accumulator: list[dict[str, float]], scores: list[dict[str, float]]) -> None:
    for aggregate, score in zip(accumulator, scores, strict=True):
        for metric in METRIC_NAMES:
            aggregate[metric] += score[metric]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    shard_index = args.job_index * local_world_size + local_rank
    num_shards = args.num_jobs * local_world_size
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    # torchrun initializes process identity, but FastVideo attention also
    # requires its explicit TP/SP group objects even when both sizes are one.
    # The remaining world dimension is data parallel: every rank owns a
    # different scoring shard and a complete local transformer replica.
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    records = _calibration_records(
        args.corpus_root,
        args.seed,
        supplement_root=args.supplement_root,
        allow_incomplete=args.allow_incomplete_calibration,
    )
    shard = records[shard_index::num_shards]
    if args.max_records_per_shard > 0:
        shard = shard[:args.max_records_per_shard]
    if not shard:
        raise RuntimeError(f"Scoring shard {shard_index}/{num_shards} is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / f"partial-{shard_index:02d}-of-{num_shards:02d}.json"

    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for block scoring")
    run_id = f"h6-block-score-{shard_index:02d}-of-{num_shards:02d}"
    run = wandb.init(
        project=args.wandb_project,
        id=run_id,
        name=run_id,
        resume="allow",
        job_type="block-scoring",
        config={
            "source_revision": args.source_revision,
            "source_commit": args.source_commit,
            "model_root": str(args.model_root),
            "corpus_root": str(args.corpus_root),
            "supplement_root": str(args.supplement_root) if args.supplement_root else None,
            "category_quotas": CATEGORY_QUOTAS,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "sample_count": len(shard),
            "attention_backend": "TORCH_SDPA",
            "calibration_roots": [
                str(args.corpus_root),
                *([str(args.supplement_root)] if args.supplement_root else []),
            ],
            "seed": args.seed,
        },
    )
    started = time.time()
    transformer = _load_transformer(args.model_root, device)
    source_num_blocks = len(transformer.transformer_blocks)
    num_blocks = args.max_blocks if args.max_blocks > 0 else source_num_blocks
    if not 0 < num_blocks <= source_num_blocks:
        raise ValueError(f"max_blocks must be in [1, {source_num_blocks}], got {args.max_blocks}")
    overall = _empty_accumulator(num_blocks)
    by_stratum = {name: _empty_accumulator(num_blocks) for name in BASE_NOISE_BY_STRATUM}
    by_category = {name: _empty_accumulator(num_blocks) for name in CATEGORY_QUOTAS}
    stratum_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    processed_ids: list[str] = []
    for local_index, record in enumerate(shard):
        stratum = tuple(BASE_NOISE_BY_STRATUM)[(shard_index + local_index) % len(BASE_NOISE_BY_STRATUM)]
        inputs, layout = _inputs_for_record(
            transformer,
            record,
            stratum=stratum,
            seed=args.seed + shard_index * 10_000 + local_index,
            device=device,
        )
        sample_started = time.perf_counter()
        scores = _score_record(transformer, inputs, layout, num_blocks)
        elapsed = time.perf_counter() - sample_started
        _add_scores(overall, scores)
        _add_scores(by_stratum[stratum], scores)
        _add_scores(by_category[record["category"]], scores)
        stratum_counts[stratum] += 1
        category_counts[record["category"]] += 1
        processed_ids.append(str(record["id"]))
        run.log({
            "scoring/completed_examples": local_index + 1,
            "scoring/example_seconds": elapsed,
            "scoring/category": record["category"],
            "scoring/noise_stratum": stratum,
        }, step=local_index)
        partial = {
            "format_version": 1,
            "run_id": run_id,
            "wandb_run": run.url,
            "source_revision": args.source_revision,
            "source_commit": args.source_commit,
            "attention_backend": "TORCH_SDPA",
            "calibration_roots": [
                str(args.corpus_root),
                *([str(args.supplement_root)] if args.supplement_root else []),
            ],
            "category_quotas": CATEGORY_QUOTAS,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "num_blocks": num_blocks,
            "sample_count": len(processed_ids),
            "sample_ids": processed_ids,
            "stratum_counts": dict(stratum_counts),
            "category_counts": dict(category_counts),
            "overall": overall,
            "by_stratum": by_stratum,
            "by_category": by_category,
            "updated_at_unix": time.time(),
            "complete": local_index + 1 == len(shard),
        }
        partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
        torch.cuda.empty_cache()

    run.summary.update({
        "persistent_partial": str(partial_path),
        "sample_count": len(processed_ids),
        "elapsed_seconds": time.time() - started,
        "complete": True,
    })
    artifact = wandb.Artifact(f"{run_id}-partial", type="block-score-partial")
    artifact.add_file(str(partial_path))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
