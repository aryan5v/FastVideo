# CompactH3 code index

This branch consolidates the runnable code for the CompactH3 42-block and
34-block model lines. It intentionally excludes checkpoints, generated media,
W&B data, logs, paper drafts, and internal handoff reports.

## Carried-forward recipes

1. **Block selection:** score MiniMax-H3 blocks and select the activation-based
   map. The uniform pruning experiments remain reproducible through the shared
   scoring/selection tools, but are not release recipes.
2. **42-block recovery:** recover the activation-selected student in stages
   (500 updates, a 200-update continuation, then selection at update 750).
3. **42-block DMD2:** initialize from the recovered update-750 parent and use
   base H3 for both the frozen teacher and the trainable fake-score critic. The
   selected corrected four-call checkpoint is update 1400.
4. **34-block recovery:** derive the 34-block student from the recovered
   42-block line and continue dense teacher-state recovery. No 34-block DMD2
   checkpoint is promoted yet.
5. **Quantization:** export BF16, INT8-affine, NVFP4, and W4A16 variants. The
   NVFP4 QAD recipe keeps the audio projections out of FP4 and validates audio
   and video separately.

## Where things live

- Block scoring, map selection, folding, recovery checks, and AV gates:
  `scripts/fasth3_sprint/`
- Recovery configs:
  `examples/train/configs/fasth3_*.yaml`
- Current 34-block recovery config:
  `examples/train/configs/compacth3/release14b_recovery_wandb.yaml`
- Corrected four-call DMD2 and QAD configs:
  `examples/train/configs/distribution_matching/minimax_h3/`
- Corrected DMD2 launch/export scripts:
  `scripts/run_release20b_dmd2_v12_16gpu.sh`,
  `scripts/submit_release20b_dmd2_v12_corrected_*.sbatch`, and
  `scripts/checkpoint_conversion/export_h3_dmd2_student.py`
- Hardened recovery, checkpoint evaluation, QAD, and quantized export tools:
  `scripts/compacth3/`
- AdaLN rank and checkpoint-sweep analysis:
  `scripts/compacth3/analysis/`

The checked-in SLURM launchers retain the cluster paths used by the runs so
their behavior is auditable. Override the root/checkpoint variables when
deploying elsewhere; no credentials are stored in this repository.

## Code provenance

The consolidation is applied as a squash on current upstream `main`. Recovery
utilities come from the verified recovery snapshot (`2e7fa15f`), corrected
DMD2 from the audited release snapshot (`c63ae5b`), and quantization/QAD support
from the quant-support snapshot (`236cd131`). Newer upstream H3 inference,
FastH3 V2 scheduling, LoRA, TAEH3, MXFP8, and multi-device loading changes are
retained.
