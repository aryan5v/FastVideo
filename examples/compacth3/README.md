# CompactH3

Runnable code for the CompactH3 42-block and 34-block model lines. Checkpoints,
generated media, W&B data, logs, and internal research notes are intentionally
excluded.

## Recipes

1. **Block selection:** score MiniMax-H3 blocks and select the activation-based map.
2. **42-block recovery:** staged recovery (500 updates, 200-update continuation,
   selection at update 750).
3. **42-block DMD2:** initialize from the recovered update-750 parent with base H3
   as teacher and fake-score critic. Selected four-call checkpoint: update 1400.
4. **34-block recovery:** derive the 34-block student from the recovered 42-block
   line and continue dense teacher-state recovery. No 34-block DMD2 checkpoint is
   promoted yet.
5. **Quantization:** export BF16, INT8-affine, NVFP4, and W4A16 variants. The NVFP4
   QAD recipe keeps audio projections out of FP4 and validates audio and video
   separately.

## Layout

| Area | Path |
|---|---|
| Block scoring, folding, recovery checks, AV gates | `scripts/fasth3_sprint/` |
| Recovery configs | `examples/train/configs/fasth3_*.yaml` |
| 34-block recovery config | `examples/train/configs/compacth3/release14b_recovery_wandb.yaml` |
| DMD2 and QAD configs | `examples/train/configs/distribution_matching/minimax_h3/` |
| DMD2 launch and export | `scripts/run_release20b_dmd2_v12_16gpu.sh`, `scripts/submit_release20b_dmd2_v12_corrected_*.sbatch`, `scripts/checkpoint_conversion/export_h3_dmd2_student.py` |
| Recovery eval, QAD, quantized export | `scripts/compacth3/` |
| AdaLN rank and checkpoint-sweep analysis | `scripts/compacth3/analysis/` |

Cluster launchers keep the paths used by the original runs. Override
`SPRINT_ROOT`, `CODE_ROOT`, and checkpoint variables when deploying elsewhere.
