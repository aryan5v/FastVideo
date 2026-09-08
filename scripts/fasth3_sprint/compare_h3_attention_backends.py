#!/usr/bin/env python3
"""Compare SDPA and FlashAttention through the H3 training wrapper."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path

import torch

from fastvideo.distributed import get_sp_group, maybe_init_distributed_environment_and_model_parallel
from fastvideo.train.entrypoint.dcp_to_diffusers import _ensure_distributed, _run_config_from_raw
from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    _euler_update,
)
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount
from fastvideo.train.utils.instantiate import instantiate


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = candidate.double() - reference.double()
    reference_flat = reference.double().reshape(-1)
    candidate_flat = candidate.double().reshape(-1)
    rmse = float(difference.square().mean().sqrt())
    reference_rms = float(reference_flat.square().mean().sqrt())
    return {
        "max_abs": float(difference.abs().max()),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1.0e-12),
        "cosine": float(torch.nn.functional.cosine_similarity(reference_flat, candidate_flat, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026090801)
    args = parser.parse_args()
    if args.grid_points < 2:
        raise ValueError("At least two grid points are required")
    metadata = json.loads((Path(args.source_checkpoint).resolve() / "metadata.json").read_text())
    base_raw = copy.deepcopy(metadata["config"])
    base_raw["training"]["distributed"].update({
        "num_gpus": 1,
        "sp_size": 1,
        "tp_size": 1,
        "hsdp_replicate_dim": 1,
        "hsdp_shard_dim": 1,
    })
    base_raw["training"]["data"].update({
        "num_frames": 5,
        "num_latent_t": 2,
        "num_height": 64,
        "num_width": 64,
    })
    base_raw["training"]["dit_precision"] = "bf16"
    base_raw["training"]["model"]["enable_gradient_checkpointing_type"] = None
    base_raw["callbacks"] = {}
    _ensure_distributed()
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    backend_receipts: dict[str, dict[str, object]] = {}
    for backend in ("TORCH_SDPA", "FLASH_ATTN"):
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
        raw = copy.deepcopy(base_raw)
        teacher = copy.deepcopy(raw["models"]["teacher"])
        teacher.update({
            "init_from": str(Path(args.model_path).resolve()),
            "trainable": False,
            "disable_custom_init_weights": True,
            "enable_gradient_checkpointing_type": None,
            "attention_backend": backend,
        })
        raw["models"] = {"student": teacher}
        raw["method"] = {
            "_target_": "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod",
        }
        cfg = _run_config_from_raw(raw)
        model = instantiate(cfg.models["student"], training_config=cfg.training)
        if not isinstance(model, MiniMaxH3Model):
            raise TypeError("backend comparison requires MiniMaxH3Model")
        model.sp_group = get_sp_group()
        generator = torch.Generator(device=model.device).manual_seed(args.seed)
        raw_batch = {
            "text_embedding": torch.linspace(-0.5, 0.5, 8 * 5120).reshape(1, 8, 5120),
            "text_attention_mask": torch.ones(1, 8, dtype=torch.bool),
        }
        batch = model.prepare_batch(raw_batch, generator=generator, latents_source="zeros")
        if batch.noise is None or batch.audio_noise is None:
            raise RuntimeError("training wrapper did not create initial noise")
        video = batch.noise.permute(0, 2, 1, 3, 4)
        audio = batch.audio_noise
        video_sigmas = shift_noise_amount(torch.linspace(1, 0, args.grid_points, device=model.device), 12.0)
        audio_sigmas = shift_noise_amount(torch.linspace(1, 0, args.grid_points, device=model.device), 3.0)
        with torch.inference_mode():
            for interval in range(args.grid_points - 1):
                vt = (1.0 - video_sigmas[interval]).reshape(1)
                at = (1.0 - audio_sigmas[interval]).reshape(1)
                video_flow, audio_flow = model.predict_joint_noise(
                    video,
                    audio,
                    vt,
                    at,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                video = _euler_update(video, video_flow, video_sigmas[interval], video_sigmas[interval + 1])
                audio = _euler_update(audio, audio_flow, audio_sigmas[interval], audio_sigmas[interval + 1])
        outputs[backend] = (video.detach().float().cpu(), audio.detach().float().cpu())
        resolved = getattr(model.transformer.config, "_resolved_attention_backend", None)
        backend_receipts[backend] = {
            "requested": backend,
            "resolved": getattr(resolved, "name", str(resolved)),
        }
        del model, batch, video, audio
        gc.collect()
        torch.cuda.empty_cache()

    video_metrics = _metrics(outputs["TORCH_SDPA"][0], outputs["FLASH_ATTN"][0])
    audio_metrics = _metrics(outputs["TORCH_SDPA"][1], outputs["FLASH_ATTN"][1])
    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "seed": args.seed,
        "grid_points": args.grid_points,
        "transformer_calls": args.grid_points - 1,
        "scope": "64x64 five-frame synthetic-conditioning endpoint smoke test",
        "minimum_cosine": args.min_cosine,
        "backends": backend_receipts,
        "video": video_metrics,
        "audio": audio_metrics,
        "passed": min(video_metrics["cosine"], audio_metrics["cosine"]) >= args.min_cosine,
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("SDPA/FlashAttention cosine gate failed")


if __name__ == "__main__":
    main()
