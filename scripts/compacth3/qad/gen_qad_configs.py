#!/usr/bin/env python3
"""Emit launch-ready NVFP4 QAD configs for BOTH the rank-768 (20B) and rank-16 (~17B) students.

QAD = the release path: FP4 forward with a full-precision backward (STE), so FSDP
sharding and checkpointing stay dense-identical. No weight conversion needed.
"""
import json, pathlib, sys, yaml

S = pathlib.Path("/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829")
M = pathlib.Path("/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1")
RUN = S / "runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3"
TEACHER = S / "release-candidates/base-h3-teacher-complete-v1"

meta = json.loads((RUN / "checkpoint-1400/metadata.json").read_text())
c = meta["config"]
def y(p, d=None):
    cur = c
    for k in p.split("."):
        if not isinstance(cur, dict) or k not in cur: return d
        cur = cur[k]
    return cur

VARIANTS = {
    "r768": {
        "adaln_rank": 768,
        "init_from": str(RUN / "inference/checkpoint-1400"),
        "note": "20B baseline lineage; identical architecture to the shipped rank-768 model.",
    },
    "r16": {
        "adaln_rank": 16,
        "init_from": str(RUN / "inference/checkpoint-1400-reparam-r16"),
        "note": ("~17B. Student is the post-hoc centered-affine rank-16 reparameterization of "
                 "checkpoint-1400, materialized as a rank-16 checkpoint. Requires that "
                 "materialization to exist -- see the rank-compression run."),
    },
}

HEADER = """# NVFP4 QAD -- {tag} (adaln_rank={rank})
"""

def build(tag, v):
    q = {
        "models": {
            "student": {
                "_target_": y("models.student._target_"),
                "init_from": v["init_from"],
                "trainable": True,
                "enable_gradient_checkpointing_type": "full",
                "attention_backend": y("models.student.attention_backend", "TORCH_SDPA"),
                "quant_config": "nvfp4_qat_train",
                "adaln_rank": v["adaln_rank"],
            },
            "teacher": {
                "_target_": y("models.teacher._target_"),
                "init_from": y("models.teacher.init_from", str(TEACHER)),
                "trainable": False,
                "disable_custom_init_weights": True,
                "attention_backend": y("models.teacher.attention_backend", "TORCH_SDPA"),
            },
        },
        "method": {k: y(f"method.{k}") for k in (
            "_target_", "rollout_mode", "rollout_carry", "rollout_carry_slots",
            "rollout_sample_type", "generator_update_interval", "real_score_guidance_scale",
            "dmd_denoising_steps", "min_timestep_ratio", "max_timestep_ratio",
            "score_timestep_shift", "score_timestep_warp_max", "score_timestep_continuous",
            "fake_score_loss_space", "modality_loss_weights", "dmd_denom_floor_ratio",
            "dmd_grad_cap", "cfg_uncond", "fake_score_learning_rate", "fake_score_betas",
            "fake_score_lr_scheduler")},
        "training": {
            "distributed": y("training.distributed"),
            "data": y("training.data"),
            "optimizer": y("training.optimizer"),
            "loop": {"max_train_steps": 200,
                     "gradient_accumulation_steps": y("training.loop.gradient_accumulation_steps", 8)},
            "checkpoint": {
                "output_dir": str(S / f"runs/release20b-qad-nvfp4-4call-{tag}-v1"),
                "resume_from_checkpoint": v["init_from"],
                "training_state_checkpointing_steps": 25,
                "require_complete_training_checkpoint": True,
                "checkpoints_total_limit": 12,
            },
            "tracker": {"trackers": ["wandb"], "project_name": "fasth3-14b-2step-qad-sprint",
                        "run_name": f"release20b-qad-nvfp4-4call-{tag}-v1"},
        },
        "callbacks": {
            "grad_clip": {"_target_": "fastvideo.train.callbacks.grad_clip.GradNormClipCallback",
                          "max_grad_norm": 1.0},
            "validation": y("callbacks.validation"),
        },
        "model": {"precondition_outputs": False, "enable_gradient_checkpointing_type": "full",
                  "enable_torch_compile": False},
        "dit_precision": "fp32",
        "vsa": y("vsa"),
    }
    out = M / f"examples/train/configs/distribution_matching/minimax_h3/qad_nvfp4_4call_{tag}.yaml"
    out.write_text(HEADER.format(tag=tag, rank=v["adaln_rank"], init=v["init_from"], note=v["note"])
                   + yaml.safe_dump(q, sort_keys=False))
    return out, q

for tag, v in VARIANTS.items():
    out, q = build(tag, v)
    print(f"{tag}: {out.name}")
    print(f"   adaln_rank={q['models']['student']['adaln_rank']}  init={q['models']['student']['init_from'][-40:]}")
    print(f"   quant={q['models']['student']['quant_config']}  steps={q['training']['loop']['max_train_steps']}")
