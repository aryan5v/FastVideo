# FastH3 sprint repair addendum

> Latest scope: follow the [consumer model decision](fasth3_consumer_model_decision.md).
> The latest follow-up restores structural pruning as the main path and asks
> whether the improving hard run can recover with a longer training budget.

> Research review, September 6: before committing to repeated small cuts,
> follow the [pruning experiment decision](fasth3_pruning_research_decision.md).
> It requires a bounded comparison of adaptation-aware pruning and hard-cut
> recovery, superseding the presumed small-cut ladder below. The 48-block run
> remains a diagnostic control. The approximately 14B pruned/two-call objective
> is unchanged; the review also resolves training-stage order and time estimates.
> The user now permits a week or longer when results are promising, replacing
> the earlier three-day constraint described below.

## Scope

The user's latest direction requires a smaller, structurally pruned FastH3.
Restore the [original sprint plan](fasth3_14b_2step_qad_sprint_agent_prompt.md)
as the main plan. Approximately 14B and two transformer calls remain the
primary targets. Model size has some flexibility for a demonstrated quality
fallback; it is not permission to substitute the intact released model.

Keep the original distillation, QAT and QAD program, joint audiovisual release
gates, MLX INT8/INT6/INT4 and trained-VSA CUDA artifacts, checkpoint retention,
license requirements, and required branch. Quantization complements pruning.
The intact V1 checkpoints remain teachers and quality references.

The roughly three-day release-candidate target is a schedule objective, not
proof that aggressive pruning will recover in that time. Report a missed gate
honestly rather than changing the deliverable to meet the date. Defer hardware
access setup while model recovery is the active priority; retain final consumer
validation and the requested team dogfood instructions.

## Necessary corrections

1. **Replace a large direct cut with progressive pruning and recovery.**
   Direct 20/24-block initializers collapsed. Both uniform 40-block branches
   retain recognizable scenes but fail exact speech. Start a small cut from
   healthy V1, initially 50 to 48 blocks, then choose further reductions based
   on joint quality after recovery. This first depth is an experiment, not an
   established working initializer or a proposed release size.
2. **Select removals jointly and preserve recovered weights.** Activation
   scores propose candidates; they do not prove that simultaneous removals
   work. Compare joint video/audio behavior and bounded recoverability, keep
   endpoints, and reject destructive cuts. Later stages inherit the last
   accepted recovered checkpoint, with parent hashes and composed original
   block maps. Keep original branch-matched V1 supervision as a stable anchor.
3. **Fix numerical learning and data geometry.** Keep FP32 master parameters
   and Adam moments with BF16 computation, and record actual nonzero updates.
   Reuse the corrected 512 real plus 64 synthetic samples at 124 output frames,
   37 video latents and 207 audio latents. Do not rebuild valid data or discard
   existing references and audit receipts.
4. **Use targets appropriate to the evaluated state.** Match teacher and
   student on the same four-call rollout states, normalize video/audio losses
   independently, and keep feature matching subordinate. The paired data-flow
   target needs its own straight-interpolation forward. It is disabled on
   rollout predictions in the first corrected recovery trial.
5. **Make 25 steps a diagnostic checkpoint.** Check gradients and updates from
   the first optimizer step and inspect decoded results at step 25. Lack of
   visible improvement that early does not establish irrecoverability. Use
   later bounded checkpoints to assess stable recovery before accepting the
   next cut. Never advance automatically after audio/video collapse, and never
   present a recognizable initializer as release-ready.
6. **Keep verified loading and export arithmetic.** Explicit attention backend
   preservation fixed dense/VSA DCP-export parity. Preserve all VSA gates and
   rerun matching-backend joint prediction parity for new artifacts. The old
   discrepancy did not establish export weight corruption.

## Immediate sequence

- Preserve the completed 40-block artifacts and failed speech receipts.
  Job 6527 reports WER 1.0 for both exact-speech clips; this establishes a
  failed speech-prompt gate, not a complete diagnosis of all audio behavior.
  The guarded recovery job 6528 stopped before training, as intended.
- Finish the already running 36/32 diagnostic comparisons as controls. They
  are not a presumed solution, and their existence does not authorize training.
- Prepare and evaluate a small first pruning increment from the healthy parent.
  Require recognizable video, motion and noncollapsed audio before committing
  recovery compute under the existing handoff gate. If it fails, reduce or
  reselect the cut rather than spending the sprint on another collapsed start.
- Recover the accepted cut using the corrected implementation. Preserve the
  four-call trajectory while pursuing progressively smaller students toward
  the approximately 14B tier. A larger intermediate is progress only.
- Continue the original few-call distillation and target-precision training
  program after a healthy pruned recovery checkpoint exists. Preserve the
  high-precision anchor and evaluate joint media at every transition.

## Consumer constraints retained

The final model must demonstrate quality, end-to-end speed and memory fit on
consumer machines. RTX 4090/5090 and the user's 24 GB M5 and 36 GB M4 Max are
validation targets, with team dogfooding for additional machines. GB200 timing
alone does not establish consumer performance. RTX 30/40 require a supported
format and execution path for their architecture; native NVFP4 is a Blackwell
route. No new hardware setup is on the critical path to the next model gate.
