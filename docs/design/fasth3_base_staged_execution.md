# Base H3 staged execution

The current sequence is Base50 controls, two Base42 step-zero candidates, recovered42, then recovered34. Preserve all previous experiments. Never initialize34 freshly from Base50 after recovering42.

## Active work

- Base50 control 6822 completed; sampled frames0/60/120 across four prompts are coherent. Audio and full-motion quality are not yet reviewed. Local contact sheet: review-artifacts/base-h3-start/base-control-media/contact.jpg.
- Extraction6839 completed after removing a GPU-runtime import from the CPU converter. Candidate: checkpoints/base42-uniform-v1 under the sprint root.
- Prompt audit6836 completed:58669 requested/matched/validated embeddings, no missing/conflicts. Encoder provenance and native-geometry GPU forward/backward remain. Prompt-only zero latents are not real paired targets.
- Recovery6846 cancelled during initial inference loading, before training, following the new controls-first instruction. Do not restart this old summary-feature recipe.
- Gate6868: Base50 SDPA and FLASH_ATTN four-prompt panels, ASR receipts and backend endpoint smoke. Code: code/base-gates-d3d425c9. Log: diagnostics/base-gate-6868.log.
- Step-zero6869: uniform42 SDPA four-prompt decode. Same code; log: diagnostics/base42-step0-6869.log.
- Activation scoring6870: base42-score job, code/base-score-1db49208. Four GPU shards,12 examples each, all50 block ablations, four-hour cap. This48-example screen allows incomplete categories and is not the old locked256-example calibration. Check actual category/noise coverage before trusting its map. Aggregator retains first/last blocks. Extract and decode the resulting42 map; compare both candidates before training.

All remote relative paths above are under /mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829. Use existing SSH credentials recorded in the task. Every new job must use an immutable code snapshot and --export=NIL. Preserve checkpoints and outputs.

## Interpretation and remaining work

The existing runner's --steps50 means50 grid points and49 transformer calls; report that exactly. It uses guidance1.0. Current backend parity is only64x64/five-frame synthetic-conditioning endpoint parity, not full-panel parity. FLASH_ATTN is now permitted for frozen wrapper evaluation, while trainable recovery remains SDPA. Earlier parity6817 failed because FLASH_ATTN was rejected, not because numerical parity failed. The new result is still pending.

ASR --max-wer100 deliberately collects diagnostics without aborting the other backend. It does NOT constitute an audio pass. Read WER values and review audio; exact speech should be near0. Review motion over the full clip, not only contact sheets. Cosine>0.999 and endpoint≤1.5 are provisional numerical gates;2.0 is an uncorrelated equal-energy reference under specific normalization, not a universal noise floor. Recognizable content on half the panel is an experimental progression floor, not release acceptance.

The requested new recovery loss is NOT implemented yet. Existing Base42 config remains sampled real-data timestep flow/KD with mean/RMS features0.01 and must not be launched under the new plan. Implement teacher-prefix velocity KD and tokenwise seam features with per-modality normalization, explicit outlier treatment and requested weight1.0; audit gradient magnitudes. Real paired flow loss needs its own noised forward. Teacher-prefix trajectory samples must come from Base, with no student-prefix or closed-loop rollout training. Rollouts remain validation only. Verify packed-row/SP alignment and checkpoint-recomputation hooks; naive global row indexing into SP-local states is wrong.

After controls, candidate comparison and focused tests pass, run two updates with actual FP32-master/Adam changes recorded, then a200-update/four-hour pilot. Save, export, verify predictions and decode. Continue500/1000/2000 only while broader held-out quality improves. No promise of recovery from a fixed number of steps.

Staged converter now composes original block identities and supports single-file DCP exports; a synthetic test changes42 weights before the34 extraction and verifies those changes survive. Promotion script requires explicit checkpoint-bound quality review. prepare_h3_stage34.py generates a bounded next-stage launcher, but currently inherits the old recovery template: update it to the new recipe before use. Never treat generated configuration as proof the method was implemented or validated.

No BVD, annealed arm, PDD, QAT/QAD or hardware work at this stage. Six-hour follow-up automation is active and should advance actionable work, remain quiet when unchanged, and pause when the comparison is complete or user input is required.

## Step-zero review and backend diagnosis

Uniform42 job6869 completed in8m51s. Local media are in review-artifacts/base42-step0-6869. At frames0/60/120: presenter recognizable but severely discolored/contrasty; motorcycle recognizable initially but absent at frame120; mechanical press replaced by unrelated texture; train transition recognizable with severe saturation. This is not a quality pass. Full motion and audio remain unreviewed; WER pending.

Baseline6868 failed after producing the SDPA panel because ffmpeg was absent from PATH. Retry6871 uses imageio_ffmpeg's resolved executable, generates the missing FLASH_ATTN panel, transcribes existing SDPA and uniform42 speech, and runs backend parity. Immutable code/backend-fix-43c35c37; diagnostics/base-gate-retry-6871.log.

Scoring6870 bypassed PipelineComponentLoader's explicit backend scope by calling TransformerLoader directly: FastVideoArgs.attention_backend alone was not applied and logs showed automatic FlashAttention despite hardcoded SDPA metadata. Preserve this as exploratory evidence, not matched-SDPA ranking. Retry6872 explicitly scopes TORCH_SDPA and asserts the resolved backend; same immutable code, diagnostics/base42-score-sdpa-6872.log. Do not silently combine its partials with6870.

Recovery duration: first2-update numerical proof, then200-update pilot with evaluations at50/100/200 if feasible. Continue to500 only if held-out audio/video improves; review1000/2000 subsequently. Compare elapsed GPU hours, not only updates. A severe regression pauses immediately; two successive flat/degrading held-out evaluations trigger diagnosis rather than automatic extension. Preserve best checkpoints. These are proposed operational gates pending implementation of the new recipe and held-out evaluator, not an already-working automatic training chain.

## Recovery now staged (supersedes missing-recipe notes above)

User explicitly requested uniform42 recovery while activation selection proceeds. New implementation in minimax_h3_base_recovery.py uses a uniformly sampled interval from the50-grid-point Base schedule; integrates only frozen teacher prefixes without autograd; predicts joint velocity at that state; matches full tokenwise states at the eight uniform-pruning seams with weight1.0; and uses a separate correctly noised real-data forward at the same timestep for the paired flow loss. Extreme teacher values above10 times modality RMS are masked, not student-clamped; excluded values are also removed from the energy denominator. This is an outlier-masked variant, not literal clipping. No student-prefix path or annealing. Prompt-only inputs deliberately rejected for this paired pilot. Full58669 prompt integration is not active here.

Six focused CPU tests pass for actual-update guards, modality balance, text/padding exclusion, and outlier masking/normalization; pre-commit including mypy passes. GPU forward/backward and distributed gradient-scale validation remain experimental. Two-update gate precedes automatic continuation to200, capped4hours total. The launcher requires Base backend parity and exact-speech WER0 for both backends. Numerical success does not authorize the34 cut. Export and panel generation occur at200; a wall-time stop may require exporting the latest complete intermediate checkpoint on follow-up. Check first gradients/loss scales and actual update receipts before accepting prolonged recovery.

Job6873 was cancelled while pending to include the corrected outlier denominator. Its replacement6875 uses immutable revision d2520f79. Activation extraction/decode job6874 is queued afterok6872 and uses immutable code/recovery-d1f2b3c2. It extracts the activation-selected42 and automatically decodes four samples; the48-example scoring screen is still limited evidence.

Uniform42 speech from6869 now has WER0.0: transcript exactly matches the eight-word expected sentence. Receipt: diagnostics/base-h3-controls/6871-TORCH_SDPA/6869-TORCH_SDPA-wer.json. This is single-clip intelligibility evidence, not audio naturalness or sync certification. Base SDPA speech also WER0.0. Read actual values rather than the diagnostic passed flag whose threshold was100.

## New numerical gate failure

6871 finished panels/ASR but failed its49-call cross-backend endpoint gate: video cosine0.9972312860, relative RMSE0.07606369, versus required0.999. Consequently6875 was cancelled by SLURM before it ran. Recovery is staged but NOT currently training. Do not report a live200-update recovery job.

A new bounded same-state diagnostic6876, code/same-state-78d4d7f0, compares SDPA and FlashAttention velocities on identical SDPA teacher-prefix states at every interval. This separates local backend mismatch from accumulated trajectory divergence. Its output must be reported separately and must not erase/relabel the failed endpoint gate. If same-state parity passes, review both decoded Base panels and explicitly document whether SDPA-only recovery is interpretable before resubmitting; do not silently weaken the previous gate. User wants actual200-update recovery soon but not false quality claims. Activation extraction/decode6874 remains dependent on corrected scorer6872.

Scorer6872 emits W&B monotonic-step warnings because the initial run-ID used the leaf directory name partials, shared with6870. Keep remote partial JSON files as authoritative, isolated evidence; do not use their shared W&B series as a clean comparison. Future run IDs include the job-specific parent directory. Running code remains unchanged.

## Latest user authorization: launch both500-update experiments

User explicitly directs both recovery branches to start without waiting for external verification. This supersedes the external baseline/review holds for these experiments, not the requirement for truthful reporting. Jobs6877 (uniform42) and6878 (activation42) are submitted from immutable code/recovery500-dedc0d66. Each targets500 updates with an8-hour wall limit, four GPUs, checkpoints every50 updates and automatic export/decode after500. These bounds replace the previous200-update/four-hour pilot bounds for these two newly authorized jobs only.500 is a first recovery budget, not an estimated optimum or guaranteed quality. No external baseline gate or two-update pause; first actual-update guards and finite-loss checks remain in the training method.

Activation6878 depends afterany6874, then verifies actual extracted checkpoint files exist. It may proceed if step-zero decoding fails after successful extraction; it must not train from incomplete or wrong weights. The config builder derives tokenwise seam locations from each candidate's actual block map. Both use Base teacher and SDPA. No34-stage promotion from mere job completion. All58669 prompt embeddings are still not used in this paired-data pilot. If8hours expires before500, report actual completed steps and export the latest complete checkpoint; do not claim500 ran.

## Quality tolerance clarified by user

The user accepts modest quality degradation from compression when memory/speed benefits are worthwhile. Exact perceptual parity with Base H3 is not required. Review detail, texture and realism loss as tradeoffs rather than automatic failures. Severe artifacts, missing subjects, prompt failure, motion collapse and unintelligible speech remain unacceptable. Numerical implementation/export checks have a different purpose and are not relaxed by this perceptual tolerance. Compare recovered samples against both Base and step-zero candidates; quantify deployment benefits when measured, without claiming consumer readiness from cluster timings. Existing500-update jobs and resource caps remain unchanged.

## Completed500 and uniform checkpoint failure

6878 activation42 completed500 updates in4h47m55s, including export and four decoded videos (~19.19 GPU-hours on4GPUs). checkpoint-500 metadata states step500; DCP.metadata plus four~83.88GB shard files are present, as are all50-step milestones. All four first-update receipts show nonzero FP32 changes at LR3e-5. This establishes completed numerical training, not quality improvement. Local outputs: review-artifacts/activation42-500-6878 and review-artifacts/activation42-step0-6874. At frames0/60/120, activation42 before training is already coherent and substantially healthier than uniform42; step500 remains recognizable but changes content and shows motorcycle ghosting. No demonstrated recovery-quality gain or34 promotion yet. Speech comparison pending.

6877 uniform failed at the first checkpoint after50 updates: DCP.save returned and wrote.metadata/four shards, then a default NCCL barrier failed allocating72bytes due GPU OOM. No evidence of a numerical-loss failure. Checkpoint50 is preserved; post-save RNG snapshot was not reached, so resume may fall back to DCP RNG state and is not guaranteed bitwise equivalent. Fixed CheckpointManager coordination barriers to use the existing CPU group when use_cpu_process_group=true, matching DCP planning; all39 checkpoint tests pass. Immutable code/checkpoint-fix-6ed25a78. Retry6924 resumes uniform from checkpoint50 toward500 total, eight-hour cap; do not restart from scratch.6925 transcribes activation42 step0 versus500 with actual WER receipts. One bounded retry for this barrier failure is now used.

6876 same-state backend diagnostic failed too: minimum video flow cosine0.88150996 on a small synthetic-conditioning input. Cross-backend interchangeability is not established; keep SDPA matched for recovery/evaluation and preserve these diagnostics. Per user instruction this does not block the authorized42 recovery runs.

Paired-data audit reconfirmed526 training records and341 unique captions; the58669 prompt-only embeddings are NOT in these two jobs. Do not imply otherwise. A passing42 candidate can initialize34 from its recovered full-rank weights; approximate34+AdaLNrank16 count remains~13.9B pending actual extraction/conversion and quality validation. Broader prompt supervision, held-out low-precision AdaLN verification and broader evaluation are useful next work; do not mutate live jobs to add them.

## Three-branch expansion authorized

The user approves the presenter perceptually and requests three experiments, preserving activation42 step500. Step500 checkpoint and export are now additionally hard-linked into preserved/activation42-step500-job6878/{checkpoint-500,export-500}; original preservation marker remains. These are separate directory entries sharing storage, not a separate physical backup. Do not overwrite either source or preserved weight files. All new outputs have distinct job directories.

Audio6925 completed: activation42 step0 WER0; step500 WER2/7=0.285714, transcript "Fast video and clear audio are rivaled together." User reports it sounds great. Treat ASR as fallible but report the measured regression; no assertion of across-the-board recovery improvement. More training is an experiment, not a demonstrated remedy.

Three current branches:

-6924 uniform42 resumes step50 toward500 total with fixed CPU checkpoint barriers; eight-hour cap.
-6926 activation42 resumes the preserved step500 model and optimizer toward1000 total (500 additional updates); eight-hour cap. Immutable code/three-branches-b2607fe2.
-6927 activation34 pipeline uses that same preserved42 step500 export, not the later1000 endpoint: score all42 blocks on48 calibration examples, retain selected34 including first/last, compose original50-block identities, extract, decode step0, train500 with Base50 teacher and seam locations derived from the actual map, export/decode. Eight-hour cap including selection/conversion/evaluation;192GB CPU memory request. Same immutable code. Output root runs/activation34-recovery/job-6927. Failure in any stage stops dependent work. This is an authorized exploratory34 branch, not a statement that42 passed all former promotion thresholds.

Selected34 training remains full-rank, FP32 masters/BF16 compute, same paired-data objective. AdaLN compression has NOT happened. Preserve stage42-step500 regardless of later outcomes, compare500 versus1000 and34 under the user's modest-degradation tolerance, and choose the best useful candidate instead of assuming later/smaller is better.

## September8 live check and bounded34 resource retry

6924 resumed actual FP32 updates at51 on all four ranks and reached checkpoint100 save; completeness was not yet established during that write.6926 restored500 plus RNG and logged updates beyond529; all four rank501 receipts show nonzero FP32 changes. Both are numerically progressing, not newly quality-certified.

6927 FAILED after5m54s: its SLURM container step was OUT_OF_MEMORY, peak host RSS201248576KiB against192GiB requested. Failure occurred while loading four complete scoring replicas, before34 extraction or recovery. One bounded resource retry6928 uses immutable code/a34-serialscore-c6d73a34 and scripts/fasth3_sprint/slurm_h3_activation34_serialscore_recovery500.sbatch. Scoring uses one process and48 records; aggregation expects one shard. Recovery still uses four GPUs,500 updates, and an8-hour whole-job cap. The same first48 record IDs are selected, but changing sharding changes per-record seed/timestep-stratum assignment; do not call this identical numerical replay. Log diagnostics/activation34-serial-to500-6928.log; outputs runs/activation34-recovery/job-6928. Preserve failed6927 and the42-step500 source. Shell syntax and applicable pre-commit checks passed. This consumes the bounded retry for the replicated-scoring host-memory failure.

Audio finding is narrowly defined: expected seven-word sentence Fast video and clear audio arrive together; ASR reported are rivaled together at500, two word errors versus zero at step0. This is not established waveform corruption, general speech failure, or a diagnosed cause. User listening was positive. Check more held-out speech/seed pairs and listening before changing loss weights; if reproducible, test speech-balanced teacher supervision and per-modality gradient balance. Current526 paired examples/341 captions remain a coverage limitation;58669 prompt embeddings are prepared but not consumed by these live jobs.

Few-call work is a future phase, not launched: current Base schedule is49 transformer calls. Official FastH3 release documents an H3-specific DMD2 four-call recipe with Base teacher, learned critic, and prompt-only backward simulation (https://haoailab.com/blogs/fasth3-preview/). Use that as the established four-call baseline on the recovered smaller architecture; compare a bounded PDD alternative on equal compute and held-out motion/speech, rather than assuming either wins. PDD means Parallel Decoding Distillation and reports4–8 NFE including LTX-2.3 video/audio (https://arxiv.org/abs/2607.26004); that is not evidence of successful two-call H3. Explicit two-call training/evaluation follows a good four-call checkpoint; do not simply reduce the sampling argument. No distillation launch is authorized by this planning note.
