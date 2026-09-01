#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate text embeddings referenced by H3 manifests but absent on disk."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from safetensors.torch import save_file
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    return parser.parse_args()


def _init_distributed() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29542")
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


def _missing_prompts(corpus_root: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    for manifest in sorted(corpus_root.glob("manifest_rank*.jsonl")):
        with manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text_sha1 = str(record["text_sha1"])
                caption = str(record["caption"])
                actual_sha1 = hashlib.sha1(caption.encode()).hexdigest()
                if actual_sha1 != text_sha1:
                    raise ValueError(
                        f"{manifest}:{line_number} caption hash {actual_sha1} does not match {text_sha1}")
                previous = captions.setdefault(text_sha1, caption)
                if previous != caption:
                    raise ValueError(f"Text hash collision for {text_sha1}")
    return {
        text_sha1: caption
        for text_sha1, caption in captions.items()
        if not (corpus_root / "text" / f"{text_sha1}.safetensors").is_file()
    }


def main() -> None:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required")
    missing = _missing_prompts(args.corpus_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="artifact-repair",
        config={
            "corpus_root": str(args.corpus_root),
            "model_root": str(args.model_root),
            "missing_text_artifacts": len(missing),
        },
    )
    repaired: list[str] = []
    if missing:
        _init_distributed()
        model_index = json.loads((args.model_root / "modular_model_index.json").read_text())
        fastvideo_args = FastVideoArgs.from_kwargs(
            model_path=str(args.model_root),
            num_gpus=1,
            inference_mode=True,
            training_mode=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
        )
        tokenizer = _component("tokenizer", args.model_root, model_index, fastvideo_args)
        processor = _component("processor", args.model_root, model_index, fastvideo_args)
        conditioner = _component("text_encoder", args.model_root, model_index, fastvideo_args)
        stage = MiniMaxH3ConditioningStage(conditioner, tokenizer, processor)
        for index, (text_sha1, prompt) in enumerate(sorted(missing.items()), start=1):
            batch = ForwardBatch(data_type="video", prompt=prompt)
            batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
            batch = stage.forward(batch, fastvideo_args)
            if not batch.prompt_embeds:
                raise RuntimeError(f"No text embedding produced for {text_sha1}")
            embedding = batch.prompt_embeds[0].squeeze(0).half().cpu().contiguous()
            if embedding.ndim != 2 or embedding.shape[-1] != 5120 or not bool(torch.isfinite(embedding).all()):
                raise ValueError(f"Invalid text embedding for {text_sha1}: {tuple(embedding.shape)}")
            destination = args.corpus_root / "text" / f"{text_sha1}.safetensors"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            save_file({"embed": embedding}, str(temporary))
            os.replace(temporary, destination)
            repaired.append(text_sha1)
            run.log({"repair/completed": index, "repair/token_rows": int(embedding.shape[0])}, step=index)
        del stage, conditioner, processor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    remaining = _missing_prompts(args.corpus_root)
    receipt = {
        "format_version": 1,
        "corpus_root": str(args.corpus_root),
        "model_root": str(args.model_root),
        "repaired_count": len(repaired),
        "repaired_sha1": repaired,
        "remaining_missing_count": len(remaining),
        "wandb_run": run.url,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    run.summary.update(receipt)
    run.finish()
    if remaining:
        raise RuntimeError(f"{len(remaining)} text artifacts remain missing after repair")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
