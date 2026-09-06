#!/usr/bin/env python3
"""Compare DCP and exported H3 predictions on one identical joint input.

Strict reload catches missing or renamed keys.  This gate is stronger: it
executes the DCP-loaded student and the exported student on exactly the same
small video/audio latent state and checks both outputs numerically.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from fastvideo.distributed import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _ensure_distributed,
    _role_model_checkpoint_state,
    _run_config_from_raw,
)
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.builder import build_from_config
from fastvideo.train.utils.checkpoint import CheckpointManager, _resolve_resume_checkpoint


def _predict(model: MiniMaxH3Model, batch: Any, attn_kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(batch.noisy_model_input, torch.Tensor)
    assert isinstance(batch.audio_noisy_model_input, torch.Tensor)
    assert isinstance(batch.timesteps, torch.Tensor)
    assert isinstance(batch.audio_timesteps, torch.Tensor)
    model.transformer.eval()
    with torch.inference_mode():
        video, audio = model.predict_joint_noise(
            batch.noisy_model_input.permute(0, 2, 1, 3, 4),
            batch.audio_noisy_model_input,
            batch.timesteps,
            batch.audio_timesteps,
            batch,
            conditional=True,
            attn_kind=attn_kind,  # type: ignore[arg-type]
        )
    return video.detach().float().cpu(), audio.detach().float().cpu()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"prediction shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    difference = candidate - reference
    ref_rms = float(reference.square().mean().sqrt())
    rmse = float(difference.square().mean().sqrt())
    denominator = max(ref_rms, 1.0e-12)
    flat_ref = reference.reshape(-1).double()
    flat_candidate = candidate.reshape(-1).double()
    cosine = float(torch.nn.functional.cosine_similarity(flat_ref, flat_candidate, dim=0))
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rmse": rmse,
        "reference_rms": ref_rms,
        "relative_rmse": rmse / denominator,
        "cosine": cosine,
    }


def _canonical_parameter_name(name: str) -> str:
    """Remove wrappers that are present only on the trainable DCP graph."""
    return name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")


def _parameter_samples(model: MiniMaxH3Model, *, samples_per_tensor: int = 16) -> dict[str, dict[str, Any]]:
    """Collect small deterministic samples without copying the full 14B model."""
    result: dict[str, dict[str, Any]] = {}
    for raw_name, parameter in model.transformer.named_parameters():
        name = _canonical_parameter_name(raw_name)
        if name in result:
            raise ValueError(f"duplicate canonical parameter name: {name}")
        detached = parameter.detach()
        # FSDP exposes DTensor parameters even when this gate reshards onto a
        # one-rank process. Its local shard is the complete tensor here and can
        # be sampled with ordinary Tensor indexing.
        local_parameter = detached.to_local() if hasattr(detached, "to_local") else detached
        flat = local_parameter.reshape(-1)
        sample_count = min(samples_per_tensor, flat.numel())
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            sample_count,
            device=flat.device,
            dtype=torch.float64,
        ).round().to(torch.long)
        result[name] = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": detached.numel(),
            "indices": indices.cpu().tolist(),
            "values": flat[indices].float().cpu().tolist(),
        }
    return result


def _compare_parameter_samples(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    mismatches: list[dict[str, Any]] = []
    compared_values = 0
    different_values = 0
    max_abs = 0.0
    for name in sorted(set(reference) & set(candidate)):
        ref = reference[name]
        cand = candidate[name]
        metadata_equal = all(ref[key] == cand[key] for key in ("shape", "dtype", "numel", "indices"))
        ref_values = torch.tensor(ref["values"], dtype=torch.float32)
        cand_values = torch.tensor(cand["values"], dtype=torch.float32)
        values_equal = ref_values.shape == cand_values.shape and torch.equal(ref_values, cand_values)
        if ref_values.shape == cand_values.shape:
            difference = (ref_values - cand_values).abs()
            compared_values += difference.numel()
            different_values += int(torch.count_nonzero(difference))
            if difference.numel():
                max_abs = max(max_abs, float(difference.max()))
        if not metadata_equal or not values_equal:
            mismatches.append({
                "name": name,
                "metadata_equal": metadata_equal,
                "values_equal": values_equal,
                "reference": ref,
                "candidate": cand,
            })
    return {
        "passed": not missing and not unexpected and not mismatches,
        "reference_parameter_count": len(reference),
        "candidate_parameter_count": len(candidate),
        "sampled_values_compared": compared_values,
        "sampled_values_different": different_values,
        "max_abs": max_abs,
        "missing_parameters": missing[:20],
        "unexpected_parameters": unexpected[:20],
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-relative-rmse", type=float, default=1.0e-4)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args()

    checkpoint = _resolve_resume_checkpoint(args.checkpoint, output_dir=args.checkpoint)
    export = Path(args.export).resolve()
    if not (checkpoint / "dcp").is_dir():
        raise FileNotFoundError(f"missing DCP directory under {checkpoint}")
    if not (export / "transformer").is_dir():
        raise FileNotFoundError(f"missing exported transformer under {export}")

    metadata = CheckpointManager.load_metadata(checkpoint)
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint metadata has no resolved config")
    cfg = _run_config_from_raw(raw_config)
    tc = cfg.training
    _ensure_distributed()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1

    _, method, _, _ = build_from_config(cfg)
    model = method._role_models["student"]
    if not isinstance(model, MiniMaxH3Model):
        raise TypeError("prediction parity is specific to MiniMaxH3Model")
    states = _role_model_checkpoint_state(method, "student")
    dcp.load(states, checkpoint_id=str(checkpoint / "dcp"))
    dcp_parameter_samples = _parameter_samples(model)

    # Use the smallest valid H3 geometry so this tests model arithmetic rather
    # than exhausting memory. Five frames map to two video and eight audio
    # latents, preserving the real modality timing relation.
    tc.data.num_frames = 5
    tc.data.num_latent_t = 2
    tc.data.num_height = 64
    tc.data.num_width = 64
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    raw_batch = {
        "text_embedding": torch.linspace(-0.5, 0.5, 8 * 5120, dtype=torch.float32).reshape(1, 8, 5120),
        "text_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32),
    }
    batch = model.prepare_batch(raw_batch, generator=generator, latents_source="zeros")
    attn_kind = "vsa" if model.attention_backend_name in {"VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"} else "dense"
    reference_video, reference_audio = _predict(model, batch, attn_kind)
    reference_repeat_video, reference_repeat_audio = _predict(model, batch, attn_kind)
    dcp_repeat = {
        "video": _metrics(reference_video, reference_repeat_video),
        "audio": _metrics(reference_audio, reference_repeat_audio),
    }

    # Release the original role graph, including the 33B teacher, before
    # loading the exported model into the same one-GPU process.
    attention_backend = model.attention_backend
    del states, method, model
    gc.collect()
    torch.cuda.empty_cache()

    exported_model = MiniMaxH3Model(
        init_from=str(export),
        training_config=tc,
        trainable=False,
        disable_custom_init_weights=True,
        attention_backend=attention_backend,
    )
    exported_model.sp_group = get_sp_group()
    exported_parameter_samples = _parameter_samples(exported_model)
    parameter_parity = _compare_parameter_samples(dcp_parameter_samples, exported_parameter_samples)
    candidate_video, candidate_audio = _predict(exported_model, batch, attn_kind)
    candidate_repeat_video, candidate_repeat_audio = _predict(exported_model, batch, attn_kind)
    export_repeat = {
        "video": _metrics(candidate_video, candidate_repeat_video),
        "audio": _metrics(candidate_audio, candidate_repeat_audio),
    }
    video_metrics = _metrics(reference_video, candidate_video)
    audio_metrics = _metrics(reference_audio, candidate_audio)
    passed = all(
        math.isfinite(value)
        for values in (video_metrics, audio_metrics)
        for value in values.values()
    ) and parameter_parity["passed"] and all(
        values["relative_rmse"] <= args.max_relative_rmse and values["cosine"] >= args.min_cosine
        for values in (video_metrics, audio_metrics)
    )
    receipt = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "export": str(export),
        "seed": args.seed,
        "attention_kind": attn_kind,
        "input_geometry": {
            "num_frames": 5,
            "video_latents": 2,
            "audio_latents": 8,
            "height": 64,
            "width": 64,
        },
        "thresholds": {
            "max_relative_rmse": args.max_relative_rmse,
            "min_cosine": args.min_cosine,
        },
        "video": video_metrics,
        "audio": audio_metrics,
        "parameter_samples": parameter_parity,
        "same_model_repeat": {
            "dcp": dcp_repeat,
            "export": export_repeat,
        },
        "passed": passed,
    }
    receipt_path = Path(args.receipt).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))

    if args.wandb_project and args.wandb_run_id:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow", config=receipt)
        run.log({
            "parity/video_relative_rmse": video_metrics["relative_rmse"],
            "parity/video_cosine": video_metrics["cosine"],
            "parity/audio_relative_rmse": audio_metrics["relative_rmse"],
            "parity/audio_cosine": audio_metrics["cosine"],
            "parity/parameter_samples_passed": int(parameter_parity["passed"]),
            "parity/dcp_repeat_video_relative_rmse": dcp_repeat["video"]["relative_rmse"],
            "parity/dcp_repeat_audio_relative_rmse": dcp_repeat["audio"]["relative_rmse"],
            "parity/export_repeat_video_relative_rmse": export_repeat["video"]["relative_rmse"],
            "parity/export_repeat_audio_relative_rmse": export_repeat["audio"]["relative_rmse"],
            "parity/passed": int(passed),
        })
        run.finish()
    if not passed:
        raise SystemExit("DCP/export joint prediction parity FAILED")


if __name__ == "__main__":
    main()
