# MiniMax-H3 DMD2 distillation configs

Few-step DMD2 distillation of the joint video/audio MiniMax-H3 transformer.
SFT configs live in
[`../../fine_tuning/minimax_h3/`](../../fine_tuning/minimax_h3/).

## Files

| File | Purpose |
|---|---|
| `release20b_dmd2_v12_dense.yaml` | Corrected four-call DMD2 recipe for the 42-block student. |
| `qad_nvfp4_4call.yaml` | NVFP4 QAD on the selected checkpoint-1400 student. |
| `release20b_validation_five.json` | Five-prompt validation panel for release grading. |

## Launch

```bash
# Topology is derived from the allocation.
sbatch --nodes=8 scripts/submit_release20b_dmd2_v12_corrected_16gpu.sbatch

# Or use the checked-in shell launcher directly:
bash scripts/run_release20b_dmd2_v12_16gpu.sh
```

`FASTVIDEO_FA4=1` selects FA4 inside roles configured with `FLASH_ATTN`. Do not
set `FASTVIDEO_ATTENTION_BACKEND` globally; attention backends are configured
per role.
