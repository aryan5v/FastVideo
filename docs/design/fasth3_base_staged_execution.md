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
