# MiniMax-H3 DMD2 distillation configs

Agent handoff: [`HANDOFF-h3-dmd2-vsa.md`](../../../../../HANDOFF-h3-dmd2-vsa.md).

Few-step DMD2 distillation of the joint video/audio MiniMax-H3 transformer:
a carried backward-simulation walk over the 4-step grid, optionally mixed
per batch with data-forced training on real t2va latents (v9). The current
recipe, comparison scope, open issues, and launch preflight are tracked in
[`h3_dmd.md`](h3_dmd.md).

SFT configs live in
[`../../fine_tuning/minimax_h3/`](../../fine_tuning/minimax_h3/). The production
Slurm launcher is
[`../../../slurm/dmd2_32xgb200.sbatch`](../../../slurm/dmd2_32xgb200.sbatch).

## Files

| File | Purpose |
|---|---|
| `dmd2_sp1_fsdp40_nuva_v9_dataforce_vsa64.yaml` | **Current.** v8 + per-batch data forcing over mixed prompt/latent data (NuVA t2va + text-only roots), batch 128 (accum 4), regional compile pending its gate. |
| `dmd2_sp1_fsdp40_vidprom_v8_bwdsim_vsa64.yaml` | Carry-only FastGen-parity recipe (data-free backward simulation, VSA-64 student). |
| `dmd2_sp1_fsdp40_vidprom_v7_vsa90.yaml` | Pre-parity 256-tile VSA recipe. |
| `dmd2_sp1_fsdp40_vidprom_v6.yaml` | Dense-student recipe: SP=1/full-shard, text-only simulate rollout, exclusive 4:1 cadence, FP32 compute boundaries. |
| `dmd2_sp1_fsdp40_vidprom.yaml` | Earlier hyperparameters under the current implementation. |
| `validation_wan64_h3.json` | Held-out prompts in H3's three-field format. |

## Launch

```bash
# Topology is derived from the allocation.
sbatch --nodes=10 examples/train/slurm/dmd2_32xgb200.sbatch

# Select the config explicitly when a submit helper sets CONFIG.
CONFIG=examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml \
  sbatch --nodes=10 examples/train/slurm/dmd2_32xgb200.sbatch
```

`FASTVIDEO_FA4=1` selects FA4 inside roles configured with `FLASH_ATTN`; the
launcher exports it. Do not set `FASTVIDEO_ATTENTION_BACKEND` globally because
attention backends are configured per role.
