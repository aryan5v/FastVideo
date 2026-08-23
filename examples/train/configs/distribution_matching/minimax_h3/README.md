# MiniMax-H3 DMD2 distillation configs

Agent handoff: [`HANDOFF-h3-dmd2-vsa.md`](../../../../../HANDOFF-h3-dmd2-vsa.md).

Few-step DMD2 distillation of the joint video/audio MiniMax-H3 transformer:
the v10 launch candidate is data-only over native-shape T2VA latents; earlier
recipes use a carried backward-simulation walk over the 4-step grid, optionally
mixed per batch with data-forced training (v9). The recipe history, comparison
scope, open issues, and launch preflight are tracked in
[`h3_dmd.md`](h3_dmd.md).

SFT configs live in
[`../../fine_tuning/minimax_h3/`](../../fine_tuning/minimax_h3/). The production
Slurm launcher is
[`../../../slurm/dmd2_32xgb200.sbatch`](../../../slurm/dmd2_32xgb200.sbatch).

## Files

| File | Purpose |
|---|---|
| `dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml` | **Launch candidate.** Data-only FastGen regime over five native-shape T2VA sources after filtering resolutions represented by fewer than 10 frozen videos, global batch 64 on 64 GPUs with full-world FSDP, regional compile for dense roles, and an eager VSA student. |
| `dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml` | Two-step, 64-GPU critic/student capacity gate over the isolated `1760x768-362f` bucket; not a training lineage. |
| `dmd2_sp1_fsdp32_v10_dataonly_mixed_vsa64.yaml` | Historical 32-GPU job-2960 recipe; failed on the `1760x768-362f` critic backward and must not reuse the fsdp64 output namespace. |
| `dmd2_sp1_fsdp40_nuva_v9_dataforce_vsa64.yaml` | Previous v8 + per-batch data-forcing experiment over mixed prompt/latent data, batch 128 (accum 4). |
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
