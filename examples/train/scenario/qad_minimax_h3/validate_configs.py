# SPDX-License-Identifier: Apache-2.0
"""Parse-validation for the two-stage H3 QAD configs.

Runs `load_run_config` on both YAMLs and then checks that every key actually
resolves:

  * `_target_` resolves to a real class for every model role, the method, and
    every callback;
  * every non-`_target_` key is a real parameter of that class's `__init__`
    (a typo'd key would otherwise be silently swallowed by `instantiate()`);
  * the per-role quantization scope survived config resolution;
  * the resolved distributed / loop / method values match what the launcher
    passes.

Usage (inside the training container):

    python validate_configs.py stage1_qat_finetune.yaml stage2_qad_distill.yaml
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from fastvideo.train.utils.config import load_run_config

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: Any = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  -> {detail!r}" if detail != "" else ""))
    if not condition:
        FAILURES.append(label)


def resolve_target(dotted: str) -> Any:
    module_name, _, attr = dotted.rpartition(".")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def check_mapping_keys(kind: str, name: str, mapping: dict[str, Any], cls: Any) -> None:
    """Every key in `mapping` other than `_target_` must be a ctor parameter."""
    params = set(inspect.signature(cls.__init__).parameters)
    for key, value in mapping.items():
        if key == "_target_":
            continue
        check(key in params, f"{kind}.{name}.{key} is a ctor parameter", sorted(params - {key})[:6])
        if key not in params:
            continue
        if isinstance(value, dict) and key not in ("data_path",):
            # Nested dicts (cfg_uncond, modality_loss_weights, ...) are
            # method-defined payloads, not ctor kwargs.
            continue


def validate(path: str, *, stage: int) -> None:
    print(f"\n=== stage {stage}: {path} ===")
    cfg = load_run_config(path)

    # --- every _target_ resolves, and every key is a real parameter ---
    for role, model_cfg in cfg.models.items():
        target = resolve_target(model_cfg["_target_"])
        check(isinstance(target, type), f"models.{role}._target_ is a class", model_cfg["_target_"])
        check_mapping_keys("models", role, model_cfg, target)

    method_cfg = cfg.method
    method_target = resolve_target(method_cfg["_target_"])
    check(isinstance(method_target, type), "method._target_ is a class", method_cfg["_target_"])
    params = set(inspect.signature(method_target.__init__).parameters)
    # Method ctor is (*, cfg, role_models); all other keys are read off the cfg
    # dict by the method itself, so there is no ctor contract to check here.
    check("cfg" in params, "method ctor takes cfg", sorted(params))

    for name, cb_cfg in cfg.callbacks.items():
        cb_target = resolve_target(cb_cfg["_target_"])
        check(isinstance(cb_target, type), f"callbacks.{name}._target_ is a class", cb_cfg["_target_"])

    t = cfg.training
    d = t.distributed

    if stage == 1:
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model

        check(method_target is FineTuneMethod, "method is FineTuneMethod", method_target.__name__)
        check(len(cfg.models) == 1 and "student" in cfg.models, "only a student role", sorted(cfg.models))
        check(resolve_target(cfg.models["student"]["_target_"]) is MiniMaxH3Model,
              "student is MiniMaxH3Model", cfg.models["student"]["_target_"])
        check(cfg.models["student"].get("trainable") is True, "student is trainable")
        # Supervised stage: needs real paired latents, not the text_only schema.
        check(t.data.preprocessed_data_type == "t2va",
              "stage 1 reads the t2va (latent-bearing) schema", t.data.preprocessed_data_type)
        check(bool(t.data.native_shape_bucketing), "native_shape_bucketing on", t.data.native_shape_bucketing)
        check(t.data.train_batch_size == 1, "train_batch_size == 1", t.data.train_batch_size)
        check(t.data.training_cfg_rate == 0.0, "training_cfg_rate == 0.0", t.data.training_cfg_rate)
        check(t.optimizer.learning_rate == 1.0e-6, "learning_rate 1.0e-6", t.optimizer.learning_rate)
        check(t.loop.max_train_steps == 4000, "max_train_steps 4000", t.loop.max_train_steps)
        check(str(t.dit_precision) == "fp32", "dit_precision fp32", t.dit_precision)
    else:
        from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
        from fastvideo.train.models.minimax_h3.minimax_h3_dmd import MiniMaxH3DMDModel

        check(method_target is DMD2Method, "method is DMD2Method", method_target.__name__)
        check(sorted(cfg.models) == ["critic", "student", "teacher"], "student/teacher/critic present", sorted(cfg.models))
        for role in ("student", "teacher", "critic"):
            check(resolve_target(cfg.models[role]["_target_"]) is MiniMaxH3DMDModel,
                  f"{role} is MiniMaxH3DMDModel", cfg.models[role]["_target_"])
        check(cfg.models["student"].get("trainable") is True, "student trainable")
        check(cfg.models["teacher"].get("trainable") is False, "teacher frozen")
        check(cfg.models["critic"].get("trainable") is True, "critic trainable")
        check("transformer_override_safetensor" in cfg.models["student"],
              "student carries a stage-1 weight override",
              cfg.models["student"].get("transformer_override_safetensor"))
        # Proven H3 DMD2 settings that must not regress.
        m = cfg.method
        check(m["generator_update_interval"] == 5, "generator_update_interval 5", m["generator_update_interval"])
        check(m["dmd_denom_floor_ratio"] == 0.05, "dmd_denom_floor_ratio 0.05", m["dmd_denom_floor_ratio"])
        check(m["dmd_grad_cap"] == 100.0, "dmd_grad_cap 100.0", m["dmd_grad_cap"])
        check(m["rollout_carry_slots"] == 8, "rollout_carry_slots 8", m["rollout_carry_slots"])
        check(m["fake_score_loss_space"] == "x0", "fake_score_loss_space x0", m["fake_score_loss_space"])
        check(m["modality_loss_weights"] == {"video": 1.0, "audio": 1.0},
              "video/audio loss parity", m["modality_loss_weights"])
        check(m["rollout_mode"] == "simulate", "rollout_mode simulate", m["rollout_mode"])
        check(m["cfg_uncond"] == {"text": "zero"}, "cfg_uncond text=zero", m["cfg_uncond"])
        check(t.data.preprocessed_data_type == "text_only",
              "stage 2 is data-free over the text_only schema", t.data.preprocessed_data_type)
        check(t.loop.max_train_steps == 2000, "max_train_steps 2000", t.loop.max_train_steps)
        check(t.optimizer.learning_rate == 2.0e-6, "learning_rate 2.0e-6", t.optimizer.learning_rate)
        # init_from must be resolvable TODAY; the adapted weights arrive only
        # through the override after stage 1 exports.
        check(os.path.isfile(os.path.join(cfg.models["student"]["init_from"], "model_index.json")),
              "student init_from is a resolvable pipeline dir",
              cfg.models["student"]["init_from"])
        check(cfg.models["student"]["init_from"] != cfg.models["teacher"]["init_from"],
              "student base differs from the teacher/critic reference")

    # --- quantization scope is per role and survived resolution ---
    scope = {role: cfg.models[role].get("construction_quant_config") for role in cfg.models}
    if stage == 1:
        check(scope.get("student") == "nvfp4_qat_train", "student scope nvfp4_qat_train", scope.get("student"))
    else:
        check(scope.get("student") == "nvfp4_qat_train", "student scope nvfp4_qat_train", scope.get("student"))
        check(scope.get("teacher") is None, "teacher stays dense", scope.get("teacher"))
        check(scope.get("critic") is None, "critic stays dense", scope.get("critic"))

    # --- attention backends must be legal for H3 ---
    from fastvideo.platforms import AttentionBackendEnum
    from fastvideo.train.models.minimax_h3.minimax_h3 import _ALLOWED_ATTENTION_BACKENDS

    for role, model_cfg in cfg.models.items():
        backend = model_cfg.get("attention_backend")
        enum = AttentionBackendEnum[backend]
        check(enum in _ALLOWED_ATTENTION_BACKENDS, f"models.{role} backend legal for H3", backend)

    # --- geometry ---
    check(d.num_gpus == 32, "num_gpus 32", d.num_gpus)
    check(d.sp_size == 4, "sp_size 4", d.sp_size)
    check(d.hsdp_shard_dim == 32, "hsdp_shard_dim 32", d.hsdp_shard_dim)
    check(t.loop.gradient_accumulation_steps == (1 if stage == 1 else 8),
          "gradient_accumulation_steps", t.loop.gradient_accumulation_steps)

    print(f"  resolved: gpus={d.num_gpus} sp={d.sp_size} shard={d.hsdp_shard_dim} "
          f"steps={t.loop.max_train_steps} accum={t.loop.gradient_accumulation_steps} "
          f"lr={t.optimizer.learning_rate} dtype={t.dit_precision} data={t.data.preprocessed_data_type}")


def warn_paths(paths: list[tuple[str, str]], *, required: bool) -> None:
    for label, path in paths:
        exists = os.path.exists(path)
        tag = "PASS" if (exists or not required) else "FAIL"
        print(f"  [{tag}] {label} exists -> {path}")
        if required and not exists:
            FAILURES.append(label)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    stage1 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "stage1_qat_finetune.yaml")
    stage2 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "stage2_qad_distill.yaml")

    validate(stage1, stage=1)
    validate(stage2, stage=2)

    print("\n=== input paths ===")
    s1 = load_run_config(stage1)
    s2 = load_run_config(stage2)
    # Inputs that must resolve today (load_run_config reads model_index.json
    # from the student init to pick the pipeline config class).
    warn_paths([
        ("stage1 student bf16 init", s1.models["student"]["init_from"]),
        ("stage2 student base", s2.models["student"]["init_from"]),
        ("stage2 teacher/critic base", s2.models["teacher"]["init_from"]),
    ], required=True)
    # Produced by stage 1; absent until then, and run_stage2.sh hard-fails on it.
    warn_paths([("stage2 stage-1 weight override (exists only after stage 1)",
                 s2.models["student"]["transformer_override_safetensor"])], required=False)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
