# Two-stage QAD for MiniMax-H3 (NVFP4 QAT -> DMD2)

H3 port of `examples/train/scenario/qad_wan2_1_mixkit/`. The published recipe is
FastWan-QAD (<https://haoailab.com/blogs/fastwan-qad/>); the shipped Wan variant
lives next door and is the source of truth for anything not H3-specific.

**The one thing that does not port:** `attention_backend: ATTN_QAT_TRAIN`.
`MiniMaxH3Model._ALLOWED_ATTENTION_BACKENDS` is
`(TORCH_SDPA, FLASH_ATTN, VIDEO_SPARSE_ATTN_H3)` and the model raises at config
time on anything else. Attention therefore stays bf16 in both stages, and NVFP4
applies to the **linear layers only**, through the per-role
`construction_quant_config: nvfp4_qat_train`. Attn-QAT is deferred until
attention itself is quantized, which it is not in this recipe.

```
examples/train/scenario/qad_minimax_h3/
  stage1_qat_finetune.yaml   supervised NVFP4-QAT finetune of the bf16 student
  stage2_qad_distill.yaml    DMD2 distillation on the stage-1 student
  run_stage1.sh              launcher (geometry + dataset overrides)
  export_stage1.sh           DCP checkpoint -> diffusers dir
  run_stage2.sh              launcher (stage-1 init override + geometry)
  validate_configs.py        load_run_config parse + key-resolution checks
  validate_configs.sbatch    runs the above in the training container
```

## Why two stages

A previous H3 run did stage 2 only: DMD2 straight onto an unadapted quantized
student, asking one run to both adapt to NVFP4 *and* match the teacher
distribution. It drifted (invented people, wrong-shot timing, anatomy errors)
and scored below the plain NVFP4 baseline on two independent graders. Skipping
stage 1 is the leading hypothesis for that failure.

## Running it

```bash
cd "$TREE"                       # code/release20b-dmd2-v12-corrected-v17
SCEN=examples/train/scenario/qad_minimax_h3

# 0. parse-validate both YAMLs (one node, one GPU, ~2 min)
sbatch "$SCEN/validate_configs.sbatch"

# 1. stage 1: supervised NVFP4-QAT finetune, 4000 steps
NUM_GPUS=32 bash "$SCEN/run_stage1.sh"

# 2. export the stage-1 DCP checkpoint to a diffusers dir
bash "$SCEN/export_stage1.sh" \
    "$SPRINT/runs/release20b-qad2s-nvfp4-qat-stage1-v1/checkpoint-4000" \
    "$SPRINT/runs/release20b-qad2s-nvfp4-qat-stage1-v1/diffusers"

# 3. stage 2: NVFP4-QAT DMD2 distillation from the exported student
NUM_GPUS=32 PRODUCTION_TARGET=200 bash "$SCEN/run_stage2.sh"
```

## Handoff: stage 1 -> stage 2

`export_stage1.sh` runs `python -m fastvideo.train.entrypoint.dcp_to_diffusers
--role student --weights-only --link-base --verify`, which produces

```
<stage1_output>/diffusers/
  model_index.json, vae/, audio_vae/, text_encoder/, tokenizer/,
  processor/, scheduler/, audio_scheduler/     <- symlinked from the base
  transformer/config.json                      <- copied from the base
  transformer/model.safetensors                <- the adapted weights, ONE file
```

H3 needs **no different export path**: `dcp_to_diffusers` reshards on a single
GPU and collapses the base's 8-shard transformer into a single
`model.safetensors`, which is exactly what
`models.student.transformer_override_safetensor` expects. This is the same
`DCP checkpoint -> diffusers dir -> student override` shape the Wan scenario
uses.

Two H3-specific notes:

* `models.student.init_from` in `stage2_qad_distill.yaml` stays on the **bf16**
  student, not on the stage-1 export. `load_run_config` resolves the pipeline
  config class via `PipelineConfig.from_kwargs ->
  get_pipeline_config_cls_from_name`, which reads `<init_from>/model_index.json`;
  pointing `init_from` at a directory stage 1 has not produced yet makes the
  stage-2 YAML fail to parse at all. This is precisely why the Wan template
  also keeps `init_from` on the always-resolvable base and delivers the stage-1
  result through `transformer_override_safetensor`.
* `run_stage2.sh` hard-fails when the override file is missing, because
  `component_loader` silently ignores an override path that does not exist and
  would then train the unadapted student.

## Smoke-tested, and one dead end

`smoke_stage1.sbatch` runs the real stage-1 config for 2 steps on one 4-GPU
node. It passes, and the log shows `NVFP4 QAT: attached 264 linears (51 skipped
by prefix filter)`, so the QAT scope is real for H3 rather than a dense run
wearing a QAD name.

`smoke_stage2.sbatch` proves model construction and the handoff: the student
loads its weights from `models.student.transformer_override_safetensor` (the log
names the override path, not `init_from`), and the teacher/critic stay dense in
bf16/fp32. It does **not** reach an optimizer step on a 4-GPU node: DMD2's
finite-update guard raises `No finite FP32 Adam update for student`.

That is a property of the one-node smoke geometry, not of these configs. The
sprint's own `qad_smoke_1node.yaml`, run unmodified on the same node, fails
identically at the same point -- and its output dir did not exist, so that smoke
path had never actually passed. The guard does pass at production geometry: the
existing `release20b-qad-nvfp4-r768-32gpu-v1` run wrote its
`dmd2_update_*_rank*.json` receipts. Treat the one-node stage-2 smoke as
unavailable until someone fixes it; stage 2 is only proven at launch geometry.

## Validation

`validate_configs.py` runs `load_run_config` on both YAMLs and then asserts
that every key resolves: each `_target_` is a real class, every non-`_target_`
key is a real `__init__` parameter of that class (`instantiate()` swallows
typos otherwise), the per-role quantization scope survived resolution, both
attention backends are legal for H3, and the resolved values match what the
launchers pass. It also re-checks the proven DMD2 settings
(`generator_update_interval` 5, `dmd_denom_floor_ratio` 0.05, `dmd_grad_cap`
100, `rollout_carry_slots` 8, `fake_score_loss_space` x0, video/audio loss
parity) so a later edit cannot quietly drop one.
