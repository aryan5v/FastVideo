#!/usr/bin/env python3
"""Generate the NVFP4 QAD yaml from checkpoint-1400 metadata."""
import json
import pathlib
import sys

M = pathlib.Path("/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1")
SPRINT = pathlib.Path("/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829")
RUN = SPRINT / "runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3"

qc = M / "fastvideo/layers/quantization/nvfp4_qat_config.py"
t = qc.read_text(); orig = t
if '"ff.fc_in"' in t:
    print("layer list: already fixed")
else:
    anchor = 'DEFAULT_FP4_LAYERS = (\n'
    assert t.count(anchor) == 1, f"anchor={t.count(anchor)}"
    t = t.replace(anchor, anchor + '    # MiniMax-H3 names its FFN "ff.", not "ffn." -- without these the\n'
                                   '    # FFN is silently left dense and QAD only covers attention.\n'
                                   '    "ff.fc_in",\n    "ff.fc_out",\n', 1)
    assert t != orig
    qc.with_suffix(".py.pre-qad-bak").write_text(orig)
    qc.write_text(t)
    print("layer list: added ff.fc_in / ff.fc_out for H3")

meta = json.loads((RUN / "checkpoint-1400/metadata.json").read_text())
c = meta["config"]

def y(path, default=None):
    cur = c
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

qad = {
    "models": {
        "student": {
            "_target_": y("models.student._target_"),
            "init_from": str(RUN / "checkpoint-1400"),
            "trainable": True,
            "enable_gradient_checkpointing_type": "full",
            "attention_backend": y("models.student.attention_backend", "TORCH_SDPA"),
            "quant_config": "nvfp4_qat_train",
        },
        "teacher": {
            "_target_": y("models.teacher._target_"),
            "init_from": y("models.teacher.init_from"),
            "trainable": False,
            "disable_custom_init_weights": True,
            "attention_backend": y("models.teacher.attention_backend", "TORCH_SDPA"),
        },
    },
    "method": {
        "_target_": y("method._target_"),
        "rollout_mode": y("method.rollout_mode"),
        "rollout_carry": y("method.rollout_carry"),
        "rollout_carry_slots": y("method.rollout_carry_slots"),
        "rollout_sample_type": y("method.rollout_sample_type"),
        "generator_update_interval": y("method.generator_update_interval"),
        "real_score_guidance_scale": y("method.real_score_guidance_scale"),
        "dmd_denoising_steps": y("method.dmd_denoising_steps"),
        "min_timestep_ratio": y("method.min_timestep_ratio"),
        "max_timestep_ratio": y("method.max_timestep_ratio"),
        "score_timestep_shift": y("method.score_timestep_shift"),
        "score_timestep_warp_max": y("method.score_timestep_warp_max"),
        "score_timestep_continuous": y("method.score_timestep_continuous"),
        "fake_score_loss_space": y("method.fake_score_loss_space"),
        "modality_loss_weights": y("method.modality_loss_weights"),
        "dmd_denom_floor_ratio": y("method.dmd_denom_floor_ratio"),
        "dmd_grad_cap": y("method.dmd_grad_cap"),
        "cfg_uncond": y("method.cfg_uncond"),
        "fake_score_learning_rate": y("method.fake_score_learning_rate"),
        "fake_score_betas": y("method.fake_score_betas"),
        "fake_score_lr_scheduler": y("method.fake_score_lr_scheduler"),
    },
    "training": {
        "distributed": y("training.distributed"),
        "data": y("training.data"),
        "optimizer": y("training.optimizer"),
        "loop": {"max_train_steps": 200, "gradient_accumulation_steps": y("training.loop.gradient_accumulation_steps", 8)},
        "checkpoint": {
            "output_dir": str(SPRINT / "runs/release20b-dmd2-v12-qad-nvfp4-4call-v1"),
            "resume_from_checkpoint": str(RUN / "checkpoint-1400"),
            "training_state_checkpointing_steps": 25,
            "require_complete_training_checkpoint": True,
            "checkpointing_start_step": 1400,
            "checkpoints_total_limit": 12,
        },
        "tracker": {"trackers": ["wandb"], "project_name": "fasth3-14b-2step-qad-sprint",
                    "run_name": "release20b-dmd2-qad-nvfp4-4call-v1"},
    },
    "callbacks": {
        "grad_clip": {"_target_": "fastvideo.train.callbacks.grad_clip.GradNormClipCallback", "max_grad_norm": 1.0},
        "validation": y("callbacks.validation"),
    },
    "model": {"precondition_outputs": False, "enable_gradient_checkpointing_type": "full", "enable_torch_compile": False},
    "dit_precision": "fp32",
    "vsa": y("vsa"),
}

out = M / "examples/train/configs/distribution_matching/minimax_h3/qad_nvfp4_4call.yaml"
out.parent.mkdir(parents=True, exist_ok=True)

import yaml
header = """# NVFP4 QAD for the 42-block four-call DMD2 student (checkpoint-1400).
"""
out.write_text(header + yaml.safe_dump(qad, sort_keys=False))
print("wrote", out)
print("student init_from:", qad["models"]["student"]["init_from"])
print("quant_config      :", qad["models"]["student"]["quant_config"])
print("denoise steps     :", qad["method"]["dmd_denoising_steps"])
print("output_dir        :", qad["training"]["checkpoint"]["output_dir"])
