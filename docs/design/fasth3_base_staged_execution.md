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
