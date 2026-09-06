# FastH3 day 1–2 mask recovery pilot

> This remains the protocol for the already launched bounded pilots. The
> [current consumer model decision](fasth3_consumer_model_decision.md) supersedes
> the fixed 14B target and specifies the proposed hard-recovery continuation.

The deliverable remains a static, substantially pruned joint video/audio model,
with roughly 14B / 20 blocks as the primary target and 24 blocks as a fallback.
PDD, QAT and QAD follow successful recovery. This experiment tests the removal
policy before spending a full recovery budget or searching many masks.

## First controlled comparison

Both dense arms start from the same immutable released 50-block V1 checkpoint,
use the same final uniform 24-block set, FP32 parameters/Adam moments, BF16
forward, training order, noise seeds, four-call solver and joint teacher loss.

- `hard`: execute only the final retained blocks from the first update.
- `annealed_skip`: always execute retained blocks; randomly skip other blocks
  with probability 0.5 increasing to 1 over 100 updates. Extract the final
  static model after adaptation.

Both arms initially load the complete parent. The hard arm is mathematically
block removal, but its loaded memory is not a compact deployment benchmark.
This fixed-mask pilot is not learned mask search or a paper reproduction.
A successful method can then be compared across constrained 20/24-block maps.

## Numerical and quality gates

CPU tests compare the actual packed H3 forward with a physically shortened
block list. They require exact joint-output and gradient agreement, including
checkpointed backward, and no gradients for removed blocks. Independent mask
RNG must not shift training noise RNG. Real attention and FSDP require a GPU
check in addition to these small-component tests.

The first GPU jobs stop after two updates. The annealed arm changes from a
partial mask to the final mask at its second update to exercise dynamic FSDP
branches. Both require finite gradients, a nonzero FP32 optimizer update,
checkpoint creation and fixed-seed held-out evaluation. Their short schedule
is a numerical check, not the recovery schedule or evidence of quality.

Subsequent fresh 25/200-update pilots require a recorded successful numerical
gate. Twenty-five updates diagnose early behavior; they are not expected to
finish recovery. Initial equal-update curves are not equal-compute evidence.
Record step time, total job time, evaluation/checkpoint overhead and peak
memory. Use matched GPU-hour checkpoints (and additional control updates if
needed) before selecting a method by compute efficiency.

Validation uses a separate RNG, the final fixed mask and held-out captions.
It records both teacher-state velocity error and closed-loop endpoint error
for video and audio separately, locally and in W&B. Two samples are a numerical
canary; broader held-out evaluation plus decoded speech, motion, synchronization
and appearance are required for selection. An improving training loss alone
cannot advance a checkpoint to PDD.

## Data and run provenance

`prepare_mask_pilot_split.py` builds artifact views from the corrected 576-pair
cache without recaching. Connected groups share normalized captions, source
IDs or text hashes; no group crosses splits. The prepared split contains 526
training records and 50 validation records across 32 validation groups. This
is exact/normalized deduplication, not semantic deduplication or a large release
training corpus. Expand the corpus after the mechanism shows promise.

`slurm_h3_mask_pilot.sbatch` requires an immutable code snapshot, records its
commit and split receipt, and refuses to overwrite an existing experiment.
All GPU operations run on SLURM compute nodes. Weights and run artifacts stay
outside Git. Before shipping, physically extract retained weights, verify
masked-to-static prediction parity and inspect the decoded compact model.
