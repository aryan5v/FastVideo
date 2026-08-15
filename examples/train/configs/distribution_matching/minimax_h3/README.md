# MiniMax-H3 DMD2 distillation configs

Data-free DMD2 distillation (teacher 50-step → student 3-step) of the 33B
dual-modality (video+audio) MiniMax-H3 DiT with the modular trainer
(`fastvideo/train/`). SFT configs live in
[`../../fine_tuning/minimax_h3/`](../../fine_tuning/minimax_h3/); the Slurm
launch is
[`../../../slurm/dmd2_32xgb200.sbatch`](../../../slurm/dmd2_32xgb200.sbatch).

## Configs

| Config | What it is |
|---|---|
| `dmd2_sp1_fsdp40_vidprom_v6.yaml` | **Current.** v5 topology/data with the recipe aligned to fastgen's DMD2 choices: uniform base-t score sampling over [0.001, 0.999], Adam betas (0.9, 0.999), one LR (2e-6) for student and critic, x0-space critic loss, grad clip 10. Deltas and rationale in the config header. |
| `dmd2_sp1_fsdp40_vidprom.yaml` | v5 (kept for reproducibility): fp32-masters + shift-aware schedule fixes, but uniform-in-sigma score sampling (`score_timestep_shift: 12.0`), beta1=0, asymmetric LRs, clip 1.0. |
| `validation_wan64_h3.json` | 64 held-out prompts (Wan2.1 DMD2 validation_64 in H3's three-field format); source of the cluster-side validation set. |

Both configs: `rollout_mode: simulate` (student rolls out from pure noise
through `dmd_denoising_steps`; only text conditioning is consumed — fastgen
has no simulate mode, its multi-step student sees forward-noised real data),
guidance-distilled teacher at `real_score_guidance_scale: 1.0`, base-t ladder
`[1000, 667, 333]` with `warp_denoising_step` OFF (the H3 adapter shifts per
modality internally: video 12 / audio 3). Known remaining gap vs fastgen: no
GAN/discriminator branch (fastgen runs `gan_loss_weight_gen: 0.03`).

## Launch

```bash
# Slurm, any node count (sp=1, full-shard FSDP derived from the allocation)
sbatch --nodes=10 examples/train/slurm/dmd2_32xgb200.sbatch
# or explicitly:
CONFIG=examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml \
  sbatch --nodes=10 examples/train/slurm/dmd2_32xgb200.sbatch

# single node (4x GB200), topology overridden on the CLI
bash examples/distill/MiniMax-H3/distill_dmd.sh
```

`FASTVIDEO_FA4=1` selects the FA4 path inside the FLASH_ATTN backend and is
required for the dense roles (the sbatch and distill script export it).
Per-role backends come from `models.<role>.attention_backend` — do not set
`FASTVIDEO_ATTENTION_BACKEND` globally.

## Topology and batch semantics

- **The H3 packed pipeline is batch-1 per forward by design** (documents with
  different caption lengths cannot stack). Batching composes through data
  parallelism and `training.loop.gradient_accumulation_steps`.
- DMD2 TTUR: critic every step, generator every
  `generator_update_interval`-th. Simulate rollout costs ~4 dense forwards
  per critic step and ~7 forwards + 2 backwards per generator step.
- Memory: v6's betas (0.9, 0.999) keep Adam's `exp_avg` (fp32, sharded —
  ~3.3 GiB/rank per trainable role at shard=40). v5's `betas: [0.0, 0.999]`
  auto-selected the buffer-free AdamW
  (`fastvideo/train/utils/optimizer.py`), which saved that state.

## Weight shard cache

`FASTVIDEO_WEIGHT_SHARD_CACHE=<tmpfs dir>` caches each rank's post-shard
DTensor chunks after the first full load
(`fastvideo/models/loader/shard_cache.py`); relaunches rebuild each 33B role
in ~2-4 s instead of ~10 min of cold-NFS reads (DMD2's three roles share one
entry). On multi-node runs with node-local cache dirs set
`FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE=1` (the sbatch does). Any validation
failure degrades to the normal full load.

## Data and validation

Training data is text-only conditioning rows
(`fastvideo/pipelines/preprocess/preprocess_minimax_h3_text_only.py`);
simulate mode never reads VAE latents. t2va rows for `data_latent`
experiments come from
`fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py` (one mp4
with soundtrack + prompt → one row; ≥124 frames at 24 fps required). The
validation callback samples every prompt in
`callbacks.validation.dataset_file` at the method's exact training sigmas
(`dmd_denoising_steps` is injected into the validation pipeline).
