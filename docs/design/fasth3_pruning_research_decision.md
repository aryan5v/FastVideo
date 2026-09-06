# FastH3 pruning: research review and experiment decision

## Decision

Approximately 14B, structurally pruned, joint video/audio, and two transformer
calls remain the target. Keep PDD, QAT and QAD. Do not commit to thirteen or
fifteen separate two-block recovery rounds. Small cuts are a control and a
fallback, not an established best method for reaching the target.

The research supports learning which blocks can be removed and teaching the
remaining blocks to compensate before final extraction. The proposed next
method experiment is fixed-budget block selection plus mask-aware distillation,
compared with ordinary hard-prune recovery under the same compute budget.
This is an H3 adaptation to test, not a reproduced result from another model.

## Evidence from primary sources

| Work | Verified result or procedure | Consequence for H3 |
|---|---|---|
| [TinyFusion, CVPR 2025](https://arxiv.org/html/2412.01199v1) | Learns removal decisions with weight adaptation; low immediate calibration loss does not reliably predict recovered quality. Uses masked representation distillation. Main DiT experiments evaluate 100K/500K recovery steps. | Choose masks by post-adaptation behavior. Do not infer that 25 steps are a recovery budget or that poor step-zero media proves irrecoverability. Image-generation evidence does not establish H3 speech recovery. |
| [FastLightGen, CVPR 2026](https://arxiv.org/html/2603.01685v3) | Trains selected block skipping before final size/step distribution matching. Reports 30% pruning as its preferred tradeoff, with deeper cuts degrading quality. Appendix: 4K adaptation iterations and 1K distribution-matching iterations on 16 H100s, totaling 80 GPU-days. | A training phase that prepares the model for block removal is a stronger candidate than repeatedly extracting nearly-full models. Its audio-conditioned video generation is not joint audio generation. |
| [PPCL, CVPR 2026](https://arxiv.org/html/2511.16156v1) | Distills contiguous teacher intervals into student replacements. Qwen-Image recipe: 6K depth, 2K width, 1K full-finetuning steps on eight H20s, using 100K image pairs. | Test local teacher-interval supervision when simple retained-block matching cannot compensate for deleted computation. Width surgery would expand the current scope and is not the first experiment. |
| [NeoDragon, ICLR 2026](https://arxiv.org/html/2511.06055v1) | Reduces a 24-block, 2.028B video denoiser to 18 blocks, 1.518B. Stage 1 uses 300 iterations, reported as 1–2 hours on four H100s; a separate teacher-alignment stage restores further quality. | Fast initial recovery is possible in some settings, but 1–2 hours excludes the complete pipeline. Its roughly 25% cut and smaller video-only denoiser do not establish a 60% H3 cut budget. |
| [PARE, 2026 preprint](https://arxiv.org/html/2605.27336v1) | Combines width reduction and dynamic block routing. Reports 7K steps each for width and routing, then 1.2K step-distillation steps. Tables label active parameters. | Routing fewer blocks does not by itself remove their stored weights. Reuse modality/time-sensitive evaluation ideas, not an active-parameter count as proof of a small static model. |
| [MobileWan, 2026 preprint](https://arxiv.org/abs/2607.06173) | Learns binary attention-head gates with a noise-biased objective, alongside recurrent reformulation and step distillation. | Additional evidence for learning removals and evaluating across noise levels. Recurrence/attention redesign is a larger change than the current depth-pruning repair. |

These works report different models, datasets, modalities, batch sizes and
hardware. Their success is evidence for an experimental mechanism, not a
transferable H3 accuracy or wall-time guarantee. None of the reviewed work
establishes a 50-to-20-block, two-call joint H3 release in three days.

## Why small cuts can help, and why they are insufficient

Deleting a residual block changes the input distribution of later blocks.
Deleting several interacting blocks can compound those changes. Adapting after
a modest change may give optimization a more useful starting point. This is a
mechanistic hypothesis, not a theorem that a sequence of small cuts preserves
final quality. Selection mistakes, accumulated drift and insufficient remaining
capacity can still defeat the entire ladder.

A fixed two-block ladder from 50 to 24 requires 13 recoveries; reaching 20
requires 15. It adds evaluation/export overhead and can spend most compute on
models much larger than the deliverable. It also never tests whether a better
jointly selected small model could recover with the same total compute.

## Bounded comparison before scaling

1. Reuse the released branch-matched V1 teachers, corrected data, attention
   parity fix and FP32-master implementation. Keep endpoints and all audio/video
   interfaces. Check mask synchronization across sequence-parallel ranks and
   checkpoint recomputation before any distributed training.
2. Prepare a small family of structured masks at the intended 20/24-block
   budgets. Use activation scores only as proposals. Constrain groups so the
   search is tractable; keep a uniform control. Learn or rank removals using
   short weight adaptation and held-out modality-normalized teacher error.
3. Compare hard-prune recovery against preparing the student with selected
   block skipping and teacher distillation before static extraction. Use the
   same parent, prompts, tokens, optimizer and GPU-hour budget. Begin with one
   bounded method pilot; do not launch a full architecture search by default.
4. Evaluate the actual extracted static model. Confirm deleted blocks and
   their weights are absent; retained VSA gates, block maps, source hashes and
   exact four-call solver are preserved. Score decoded speech, motion and
   synchronization as well as held-out losses. A learned training mask is not
   yet a smaller shipped checkpoint.
5. If token-level feature transfer is needed, compare correctly aligned local
   teacher-interval targets with the existing mean/RMS summaries. The latter
   omit token correspondence. Normalize modalities and control extreme
   activations; do not add several untested loss changes simultaneously.
6. Use small-cut recovery if the bounded comparison favors it. A gate based
   only on untrained exact-speech WER must not categorically exclude a
   research-backed adaptation pilot. This exception permits testing a new
   recovery mechanism; it does not authorize blindly extending the old
   collapsed 20/24-block runs. Advancement to PDD/QAT/QAD still requires a
   healthy pruned joint-media checkpoint.

## Remaining training stages

Use the latest handoff sequence: recovered pruned four-call model, two-call
PDD/consistency bootstrap, target QAT, then QAD/DMD2 refinement. This explicitly
resolves the older manifesto's QAT-before-bootstrap variant in favor of the
later handoff. Do not rerun all these stages after each pruning increment.

PDD must be implemented and validated for H3's actual two-call joint solver;
a generic KD loss is not evidence that PDD is complete. Preserve a frozen
high-precision two-call anchor while specializing MLX and CUDA branches. QAD
must use the actual quantized generator and joint video/audio score contract.
Optional DFD/reward repair remains conditional as in the handoff.

## Time estimate and decision budget

A reliable total ETA is not yet available: corrected H3 recovery throughput,
quality-versus-updates and joint PDD/QAD execution remain unmeasured. The old
BF16 20-block run's approximately 6.8 seconds per step on 12 GPUs cannot be
used as a benchmark for the new FP32-master rollout method on four GPUs.

Use these as provisional engineering planning allowances, excluding queue
waits and assuming sustained GPU access and a successful method pilot:

- 0.5–1 day: implement and compare the pruning/adaptation mechanism, including
  numerical checks and an initial quality curve.
- 1–3 days: selected pruned-model recovery and held-out joint-quality gates.
- 1–2 days: two-call PDD implementation/validation and training.
- 1–3 days: target QAT/QAD branches, runtime integration and release validation;
  independent work may overlap, but each shipped branch needs actual tests.

This yields roughly 4–9 days in a favorable engineering scenario, not a
statistical forecast or a promise that 14B is recoverable. Budget 1–2 weeks
operationally, with longer possible if aggressive pruning, data sufficiency or
joint QAD fails. Three days remains a stretch for a promising pruned candidate;
it is unsupported as a commitment for the complete 14B/two-call multi-format
release. After a stable pilot, replace allowances with measured steps/sec,
examples/sec, checkpoint/evaluation cost and observed recovery slope.

For scale only: 5,000 updates at 10/30/60 seconds per update take approximately
14/42/83 hours before evaluation, retries or other training stages. These are
sensitivity examples, not measured H3 throughput or a chosen step count.

## Current evidence

Job 6529 completed both uniform 48-block controls and speech checks in 30m19s
on hpc-rack-3-7. Both speech clips have WER 0; visual review is still required.
No corrected recovery training has run. Job 6524 ended with an NFS stale file
handle after the final VSA32 evaluation log; preserve and inventory its media
rather than rerunning the entire matrix. Do not change the checkout beneath a
running shell job; use immutable job script/code snapshots for future runs.
