#!/usr/bin/env python3
"""Run the released H3 teacher entirely through the training wrapper.

The output is terminal normalized video/audio latents.  Decoding happens in a
fresh one-GPU process so the sharded transformer cannot hide a VAE OOM.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.train.entrypoint.dcp_to_diffusers import _run_config_from_raw
from fastvideo.train.methods.knowledge_distillation.minimax_h3_recovery import (
    _deployment_sigmas,
    _euler_update,
)
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.instantiate import instantiate


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026090800)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    checkpoint = Path(args.source_checkpoint).resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    raw = copy.deepcopy(metadata["config"])
    teacher = copy.deepcopy(raw["models"]["teacher"])
    teacher.update({
        "init_from": str(Path(args.model_path).resolve()),
        "trainable": False,
        "disable_custom_init_weights": True,
        "enable_gradient_checkpointing_type": None,
        "attention_backend": "TORCH_SDPA",
    })
    raw["models"] = {"student": teacher}
    raw["training"]["data"]["data_path"] = str(Path(args.validation_data).resolve())
    raw["training"]["data"]["dataloader_num_workers"] = 0
    raw["training"]["dit_precision"] = "bf16"
    raw["training"]["model"]["enable_gradient_checkpointing_type"] = None
    raw["callbacks"] = {}
    raw["method"] = {
        "_target_": "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod",
    }
    cfg = _run_config_from_raw(raw)
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(
        tc.distributed.tp_size,
        tc.distributed.sp_size,
    )
    model = instantiate(cfg.models["student"], training_config=tc)
    if not isinstance(model, MiniMaxH3Model):
        raise TypeError("training-wrapper rollout requires MiniMaxH3Model")
    model.init_preprocessors(tc)
    if model.dataloader is None:
        raise RuntimeError("validation dataloader was not created")

    output_dir = Path(args.output_dir).resolve()
    if _rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()

    iterator = iter(model.dataloader)
    device = model.device
    video_sigmas = _deployment_sigmas(5, 12.0, device)
    audio_sigmas = _deployment_sigmas(5, 3.0, device)
    receipt: dict[str, object] = {
        "model_path": str(Path(args.model_path).resolve()),
        "source_checkpoint": str(checkpoint),
        "attention_backend": model.attention_backend_name,
        "samples": [],
    }
    for sample_index in range(args.samples):
        raw_batch = next(iterator)
        generator = torch.Generator(device=device).manual_seed(args.seed + sample_index)
        batch = model.prepare_batch(raw_batch, generator=generator, latents_source="data")
        if batch.noise is None or batch.audio_noise is None:
            raise RuntimeError("training wrapper did not create joint starting noise")
        video = batch.noise.permute(0, 2, 1, 3, 4)
        audio = batch.audio_noise
        with torch.inference_mode():
            for interval in range(4):
                video_time = (1.0 - video_sigmas[interval]).reshape(1)
                audio_time = (1.0 - audio_sigmas[interval]).reshape(1)
                video_flow, audio_flow = model.predict_joint_noise(
                    video,
                    audio,
                    video_time,
                    audio_time,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                video = _euler_update(
                    video,
                    video_flow,
                    video_sigmas[interval],
                    video_sigmas[interval + 1],
                )
                audio = _euler_update(
                    audio,
                    audio_flow,
                    audio_sigmas[interval],
                    audio_sigmas[interval + 1],
                )
        if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
            raise RuntimeError("training-wrapper rollout produced nonfinite terminal latents")
        info = (raw_batch.get("info_list") or [{}])[0]
        sample_name = f"sample-{sample_index:02d}"
        if _rank() == 0:
            save_file(
                {
                    "video": video.detach().float().cpu().contiguous(),
                    "audio": audio.detach().float().cpu().contiguous(),
                },
                str(output_dir / f"{sample_name}.safetensors"),
            )
            sample_receipt = {
                "name": sample_name,
                "seed": args.seed + sample_index,
                "id": info.get("id"),
                "prompt": info.get("prompt"),
                "video_shape": list(video.shape),
                "audio_shape": list(audio.shape),
                "video_rms": float(video.float().square().mean().sqrt()),
                "audio_rms": float(audio.float().square().mean().sqrt()),
            }
            cast_samples = receipt["samples"]
            assert isinstance(cast_samples, list)
            cast_samples.append(sample_receipt)

    if _rank() == 0:
        receipt["passed"] = True
        receipt["code_commit"] = os.environ.get("CODE_COMMIT")
        (output_dir / "rollout_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
