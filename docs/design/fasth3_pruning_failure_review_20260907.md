# Independent review: FastH3 consumer-pruning failure analysis and recovery plan

Reviewer evidence snapshot: September 7, 2026. Based on direct inspection of the
working tree at `d582572e` + uncommitted changes, the pilot config, the training
methods (`minimax_h3_recovery.py`, `minimax_h3_mask_recovery.py`), the model
wrapper (`fastvideo/train/models/minimax_h3/minimax_h3.py`), the DiT forward
(`fastvideo/models/dits/minimax_h3.py`), the export/parity/decode scripts, the
review bundle (`review-artifacts/fasth3-review-bundle.zip`), and the design
docs/journal (`docs/design/*.md`).

## 1. Candid assessment

**This is not primarily an implementation defect. It is insufficient recovery —
an optimization budget that is a functional no-op — applied to a pruning
severity far beyond the parent's zero-shot tolerance, with a supervision signal
too weak to heal the cut.** The decoded noise at step 200 is exactly what the
numbers already said: the *untrained* hard-masked 24-block model sits at
statistical noise level, and 200 updates at LR 1e-6 moved the endpoint error by
2% (video) / 12% (audio) — i.e., the step-200 model is *still* noise-level. The
decode is consistent with the metrics; there is no hidden mystery mismatch.

Three quantitative anchors:

1. **Noise signature.** The validation metric is target-energy-normalized MSE.
   An output that is *uncorrelated* with the teacher at matched energy scores
   ≈ 2.0. The initial video endpoint error is 2.0808 — the untrained 24-block
   closed loop is statistically indistinguishable from noise. Audio at 4.92 is
   worse: ~3.9× energy blowup, not just decorrelation. Step 200: 2.039 / 4.336.
   Still noise. "Video improved more 100→200 than 0→100" is a 1–2% move inside
   the noise regime — it cannot be read as a recovery trend.
2. **Optimization is a no-op at this LR.** AdamW at LR 1e-6, constant, batch 1
   (one joint document/update), 200 updates. Per-element per-step motion is at
   most ~LR ≈ 1e-6 (typically less while m/√v < 1); with weight elements of
   order 1e-2, that is ≲1e-4 *relative* per step. The forward runs under
   `torch.autocast(bfloat16)` with FP32 masters (`predict_joint_noise`), so a
   weight change only alters the *function* once it crosses a BF16 rounding
   boundary (~2e-3–4e-3 relative). Most elements never cross in 200 steps; the
   realistic cumulative motion over the *entire authorized 4,000-step
   continuation* at 1e-6 is ~0.5–1% relative — versus the tens-of-percent
   functional change needed to replace 26 deleted residual blocks. **The
   currently prepared continuation (`slurm_h3_hard_continuation.sbatch`) resumes
   the same config at LR 1e-6; running it is very likely wasted compute.**
3. **Severity is beyond the demonstrated zero-shot cliff.** From the journal:
   untrained uniform **48-block** decodes coherent, speech WER 0 (job 6529);
   **40-block** retains recognizable scenes but speech WER 1.0 (job 6527);
   **24-block** is pure noise. The zero-shot tolerance of this parent is ~4–8
   blocks (8–16%); the 20B target (~30 blocks ≈ 20.4B, 40% cut) is far beyond
   the cliff, so it *requires* genuine recovery training — which has never
   actually been tested at a working optimization scale (see §5, overclaim 1).

Secondary contributor: the supervision is weak for catastrophic damage —
one-interval trajectory KD against an already-distilled four-call teacher,
hidden-state supervision compressed to mean/RMS summaries at weight 0.01, and
half of all updates conditioned on the broken student's own (garbage) rollout
states (`student_state_probability: 0.5`).

## 2. Established vs. suspected (evidence ledger)

| Claim | Status | Evidence |
| --- | --- | --- |
| Mask mechanics skip whole residual blocks; endpoints + `norm_out`/`proj_out` preserved | Proven (code + CPU tests) | `fastvideo/models/dits/minimax_h3.py:925` (`continue` before block), mask-pilot CPU tests require exact agreement with physically shortened block list |
| Step-0 hard-24 model is noise-level | Proven | Initial endpoint errors 2.0808/4.9215 vs ≈2.0 uncorrelated baseline; earlier 24-block step-0 decodes were noise |
| 200 updates @1e-6 did not functionally heal | Proven | Endpoint 2.039/4.336 still noise-level; BF16+FP32 decode noise; FP32 decode noise |
| Old BF16-master runs were optimizer no-ops | Proven (sampled) | 99.2%/94.0% sampled params unchanged |
| Decode of media ran under FLASH_ATTN while training/parity used SDPA | Proven | `evidence/run_manifest.json`: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN` despite sbatch exporting TORCH_SDPA |
| Parity receipt covers only the training wrapper (both sides) | Proven (code) | `verify_recovery_export_prediction_parity.py` calls `MiniMaxH3Model.predict_joint_noise` on both sides; inference stack never in the comparison |
| Recovery mechanism never tested at effective LR | Proven | All recovery configs use 1e-6; prior "search" was 1e-6/2e-6 only |
| FP32-master updates mostly invisible to the BF16 forward at LR 1e-6 | Strong inference (not directly measured) | autocast bf16 forward + rounding-quantum math above; first-step probe verified master deltas, not functional change |
| Weak supervision limits recovery ceiling | Suspected | TinyFusion/FastLightGen/PPCL all use token-level alignment; our summary is mean/RMS @0.01 |
| 30-block zero-shot is noise | Suspected (extrapolated) | 40-block already speech-broken; 36/32 step-0 diagnostics were launched but results not in the handoff — re-verify in E1 |
| Teacher decodes coherently through the *training wrapper* | Untested | Nobody has decoded the teacher rollout (E0b) |
| Parent decodes coherently through the *6627 decode script* | Untested | 48-block control used `run_baseline_matrix.py` in a different job/script |
| DCP optimizer restore preserves moments/step | Untested | `seed_optimizer_state_for_resume` matches keys and zero-fills; GPU preflight planned, not run |
| Cached embeddings are unpadded/encoder-equivalent | Untested | Ones-mask loader assumes no padding; value audit incomplete; 32 missing |

## 3. Ranked hypotheses

1. **H1 — Optimization budget is the binding constraint (proven at the current
   setting).** LR 1e-6 × 200 × batch-1 is ~2–4 orders of magnitude below every
   comparable recipe: wlsaidhi H3 PDD runs use AdamW 1e-5 at global batch 64;
   TinyFusion-style pruning recovery uses ~1e-4 over 40–60k steps; VDN-H3 uses
   200/500/2000-step staged adaptation. Nothing here has been tried above 2e-6.
2. **H2 — Pruning severity beyond recoverable-without-real-training range
   (strong evidence).** 48 OK / 40 speech-broken / 24 noise. FastLightGen's
   preferred cut is ~30%; ours is 52% (24 blocks) and 40% (30 blocks).
3. **H3 — Supervision too weak for catastrophic damage (suspected, decisive
   test cheap).** Mean/RMS summaries can be matched by garbage token
   distributions; trajectory KD on student-rollout states trains on
   off-manifold garbage (GIGO) half the time; the four-call teacher is itself a
   distilled trajectory anchor, not a score oracle.
4. **H4 — Residual implementation risks (each cheap to close, none currently
   proven to cause the noise):**
   - decode/training attention-backend mismatch (proven mismatch, but 48-block
     control decoded coherently under the same resolution → unlikely to be the
     noise cause; still a train/deploy consistency defect to fix);
   - wrapper↔inference equivalence never demonstrated end-to-end (parity is
     wrapper-vs-wrapper; the 48-block control validates the inference stack
     only);
   - `backward()` recompute context passes scalar `video_time` while the
     original forward used per-row `unique_timesteps` (benign for dense SDPA —
     nothing in that path reads `current_timestep` and metadata is None on both
     sides — but assert it);
   - optimizer-state restore on resume (unproven);
   - cached-embedding padding/encoder provenance (unproven; ones-mask would
     feed pad tokens as real text if the cache is padded).
5. **H5 — Data coverage (weak).** 341 distinct captions cannot explain
   noise-level output; it limits *generalization after* coherence is recovered.

## 4. What actually failed vs. what was never tested

- The earlier 20/24-block collapses (dense and VSA) occurred under the **broken
  BF16-master optimizer** — they are evidence about zero-shot severity plus a
  broken optimizer, *not* about recoverability under a working one.
- The only FP32-master recovery attempt is hard/annealed 200 @1e-6 — a no-op
  budget. **The recovery mechanism has never been tested at an effective
  learning rate.** That is the single most important reframing of this
  investigation.
- The annealed-skip arm is slightly worse and slower; at a no-op LR the
  hard-vs-annealed comparison is uninformative except on runtime. Drop
  annealed.

## 5. Overclaims in the handoff

1. "FP32 master repair … repaired an optimization defect" — it made updates
   *possible*, not *effective*. At 1e-6 with a BF16-autocast forward, master
   motion below ~0.2–0.4% relative per element does not change the deployed
   function. The probe verifies master deltas, not functional change.
2. "Video improved more from 100→200 than 0→100; no numerical plateau" —
   arithmetic true, but the metric sits at the ≈2.0 uncorrelated-noise level;
   1–2% moves there are compatible with "output energy shrinking toward
   teacher scale while remaining noise." Not evidence of an improving trend.
3. Hard-vs-annealed endpoint comparison — both arms noise-level; only the
   runtime conclusion stands.
4. "Compact export matches masked training checkpoint" — both sides through the
   training wrapper on a 64×64×5-frame input; the inference stack (different
   script, FLASH_ATTN resolution) was never in the comparison.
5. The authorized 500/1,000/2,000/4,000-step continuation "if improving" — at
   LR 1e-6 this is dead on arrival (§1, anchor 2). The continuation launcher
   does not override LR.
6. "48-block control coherent" — from a different job/script (6529 via
   `run_baseline_matrix.py`) than the hard200 media (6627); re-run it through
   the current script as the positive control (E0).
7. "Latency ~4s" — GB200, irrelevant to the consumer brief (already flagged in
   the handoff; keep flagged).

## 6. Decisive experiment matrix (one night, then 48 h)

All GPU work via SLURM, immutable code snapshots, W&B under the existing
project. Costs from measured throughput: 24-block hard ≈ 27 s/update
all-inclusive (job 6533: 5,398 s / 200 updates incl. validation, checkpoints,
startup; pure step ≈ 15 s). 30-block ≈ ~32–35 s/update and 36-block ≈ ~38–40
s/update estimated by executed-block scaling — measure in E1.

### Tonight (max information per GPU-hour)

| ID | Experiment | Spec | Cost | Decides |
| --- | --- | --- | --- | --- |
| **E0** | Decode-path validity control | Run the **unpruned parent V1** and the **48-block untrained control** through the *exact* 6627 static-extract → `run_baseline_matrix.py` decode path (4 prompts, same seed/geometry). Fix the backend resolution first (or record both SDPA and FLASH_ATTN decodes). | ~30 min | Whether the evaluation pipeline itself is sound. **Gate: parent coherent, speech WER 0. Fail → stop all training; all decoded evidence to date is quarantined.** |
| **E0b** | Teacher-through-wrapper control | Run the teacher's closed-loop rollout (the `on_validation_begin` loop with full mask) on 2 canaries, VAE-decode, save frames. | ~15 min | Whether the training wrapper (packing, timesteps, flow sign, text) produces coherent media. **Gate: coherent. Fail → wrapper defect; fix before any training.** |
| **E1** | Severity ladder, no training | For uniform 36/32/30 maps (reuse 48/40 artifacts): masked validation endpoint errors (2 canaries) + step-0 4-prompt decodes + WER. | ~2 h | The zero-shot error-vs-depth curve and the "distance to heal" for the 20B target; also measures 30/36-block step cost. |
| **E2** | LR probe (the decisive training test) | Resume `dense24-hard-pilot200-v1/checkpoint-200` → 500 total updates, unchanged data, **LR 1e-4 (arm A)** and **LR 3e-5 (arm B)**, two nodes in parallel. Log pre-clip grad norm every step; per-layer update/weight ratio + `update_probe_l2` every 50; teacher/student update cosine per modality; auto-decode 4 prompts + WER at 500 (wire this in — currently missing). | ~2.5–3 h per arm | Whether the budget was the binding constraint. **Gate at 500: video endpoint ≤ 1.5 AND ≥1/4 decode prompts with recognizable prompted content.** |
| **E3** (optional 3rd arm) | Supervision probe | Arm A + **tokenwise hidden-state MSE** (outlier-clamped) at blocks 11/23/36/49, weight 1.0 (keep mean/RMS as logged control), + `student_state_probability: 0.0` (teacher-prefix only). | ~3 h | Whether supervision is the binding constraint at fixed budget. |

Preconditions before E2: the planned 2-update resume preflight (200→202) with
moment/step-counter checksums — this also closes H4-d.

### Decision tree after tonight

- **E0/E0b fail** → fix evaluation/wrapper first; nothing else matters.
- **E2 passes** → budget was binding. Next 48 h: port the winning recipe to a
  **30-block (~20.4B) map** (E4: fresh 500 updates, ~4.5–5 h), then continue
  1,000/2,000 (authorized) with decode gates at every preserve step; add the
  expanded-prompt arm only *after* the unchanged-data control (see §7.8).
- **E2 fails, E3 passes** → supervision was binding; run E4 with tokenwise KD
  at both 1e-4 and 3e-5.
- **Both fail** → uniform direct-cut KD at this severity is falsified at this
  budget. Pivot to **staged layerwise adaptation** (below) or **DMD2 with
  Base-H3 score teacher + learned critic** (the original FastH3 mechanism).
  Do **not** continue to 4,000 steps of the current recipe.

### Staged layerwise fallback (Path B), costed

Drop 2 blocks at a time 50→30 (10 stages), each stage ~200 updates with
tokenwise alignment at the new seam, decodes every stage: ~2,000 updates at
~31 s avg ≈ 17 h + ~2.5 h decodes ≈ **~20 h wall** on one 4×GB200 node. This
fits the authorized week and mirrors VDN-H3's staged recipe (200 per-layer
alignment updates). Inherit recovered weights between stages; compose block
maps; keep the parent as frozen teacher throughout.

## 7. Minimal necessary changes (necessary, not speculative)

1. **LR**: recovery arms at 1e-4 / 3e-5 (keep 1e-6 only for post-recovery
   finetuning regimes). Optional 100-step warmup as cheap insurance. The
   continuation launcher must accept an LR override — today it silently
   resumes at 1e-6.
2. **Decode-backend consistency**: make the static-audit decode honor
   `FASTVIDEO_ATTENTION_BACKEND` (the manifest proves it did not), or train
   under FLASH_ATTN too; add an SDPA↔FLASH_ATTN parity check on the compact
   model (same input, both backends, cosine > 0.999) before trusting any
   recovery decode.
3. **Tokenwise hidden loss** behind a config flag (E3), outlier-clamped
   (TinyFusion-style), per-modality normalized; keep mean/RMS as control.
4. **`student_state_probability` curriculum**: 0.0 until video endpoint < 1.0,
   then 0.5. Training on a noise-level student's own states is GIGO half the
   time.
5. **Observability**: per-step pre-clip grad norm; per-layer update/weight
   ratio and `update_probe_l2` at milestones; teacher/student update cosine;
   and a **decode probe at every preserve step** (VAE-decode the validation
   rollout, save a frame grid + WER on speech canaries). This ends the
   metric-vs-perceptual ambiguity permanently.
6. **Assertions**: `student_features[i].requires_grad` after the first forward
   (guards the hook-under-checkpointing path); wrapper/inference latent parity
   assert in E0b.
7. **Drop the annealed arm** (worse, slower, uninformative at no-op LR).
8. **Ordering discipline**: the unchanged-data 200→500 control must run before
   any full-prompt switch; fix the 32 missing embeddings and encoder-provenance
   verification first; fix the prompt-index row-group read amplification
   (whole row group per sample over NFS) by pre-sharding or NVMe-caching
   embeddings before any 58k-prompt run.

## 8. Falsification criteria and explicit fallback ladder

- **Falsifier (current approach)**: at LR ≥ 3e-5 with tokenwise KD and
  teacher-prefix states, a 24- or 30-block model at 1,000–2,000 updates still
  has video endpoint > 1.5 and decodes without recognizable content → uniform
  depth pruning from this four-call parent under KD-family objectives is
  falsified at this budget. Stop; do not spend the 4,000-step authorization.
- **Path B** (above): staged layerwise 50→30 with tokenwise alignment (~20 h).
- **Path C**: DMD2 port — frozen **Base H3** score teacher + learned critic
  (init from student copy), distribution-matching loss on student rollout
  states, per-modality normalization; audit joint audio/video score
  conventions first. ~1–2 dev-days + a 500-update pilot (~4 h). This is the
  mechanism the released FastH3 itself used; it is the principled answer to
  "student is catastrophically broken" because the critic supplies a
  distribution-level signal instead of matching teacher outputs on garbage
  states.
- **Path D (deliverable fallback)**: 36–40 blocks (~24.5–27B) four-call +
  INT4 (~13–15 GB weights) — quality-strong, consumer-feasible, explicitly
  documented as missing the 20B ask; revisit width pruning (PPCL-style) later.
- **Hard stop**: if Paths B and C both fail their 2,000-update gates, depth
  pruning beyond ~20% is not viable for this parent; report with evidence and
  pivot the consumer brief to quantization + call-reduction of the full
  backbone (with the user's explicit sign-off, since that leaves the pruning
  requirement unmet).

## 9. Answers to the ten priority questions

1. **Teacher through the wrapper**: untested; E0b (~15 min) is the cheapest
   decisive test. Indirect evidence (48-block zero-shot coherent through the
   inference stack; parity wrapper-vs-wrapper bit-exact) suggests yes.
2. **Equivalence audit**: timestep conventions match (per-modality `1−σ`,
   shifts 12/3, shared `packing.py` layout/rotary); flow sign consistent
   (`predict_joint_noise` negates the DiT output; Euler
   `state − Δσ·(noise−clean)` is correct flow-matching); audio packing shared;
   text masking is a ones-mask on cached embeddings (safe only if unpadded —
   verify in the value audit); VAE normalization is moot at denoising weight 0;
   **backend mismatch is proven** (§2) — fix or parity-test; checkpoint
   recompute selects mask branches outside the checkpointed block (code
   comment + CPU tests) and hooks capture grad-connected outputs — add the
   assertion anyway.
3. **One-interval matching on noise prefixes** is weak: at interval 0 it is
   well-posed, but on student-rollout prefixes it is GIGO. Teacher-prefix-only
   (curriculum), tokenwise hidden alignment, or correctly forward-noised
   paired supervision each give a cleaner gradient. Endpoint supervision is
   currently only *measured*, never trained.
4. **Update magnitudes**: bounded by LR; mostly below BF16-forward granularity
   at 1e-6 (§1). Add the §7.5 logging; run the E2 LR probe. No LR ablation has
   ever been done above 2e-6.
5. **DCP optimizer restore**: `seed_optimizer_state_for_resume` matches keys
   and zero-fills; the 2-step resume preflight with moment/step checksums is
   the right (unrun) proof — do it before E2.
6. **Yes, 26/50 is too aggressive for zero-shot; whether it is recoverable is
   exactly what E2/E3/E4 decide.** Do not commit 4,000 steps before the LR
   probe; run the 30-block arm as the ~20B candidate rather than extending the
   24-block run.
7. **Broader prompts cannot fix noise-level collapse** — coherence is
   capacity/optimization-limited. Prompts help generalization *after*
   coherence. Fix the 32 missing embeddings + provenance regardless (§7.8).
8. **Smallest KD-vs-DMD2 comparison**: same 24-block student, same data/steps
   (500), two arms — (A) current trajectory KD, teacher-prefix;
   (B) DMD2 distribution loss, Base-H3 score teacher, critic init from student
   copy. Decode gates. Implement only if E2/E3 fail or plateau; audit joint
   audio/video score conventions first.
9. **Tonight's matrix**: E0, E0b, E1, E2(A/B), optional E3 — specs, costs,
   gates in §6.
10. **Falsification + fallback**: §8. Necessary changes vs speculative: LR
    scale, backend consistency, decoded gates, control-before-confound
    ordering are *necessary*; full-prompt corpus, annealed skip, BVD,
    PDD/QAT/QAD ordering are *speculative/downstream*.

## 10. Compute estimates (from measurements)

| Item | Cost |
| --- | --- |
| E0 + E0b (validity controls) | ~45 min |
| E1 ladder (36/32/30: validation + step-0 decodes) | ~2 h |
| E2 arm (300 resumed updates + decode) | ~2.5–3 h |
| E4 30-block fresh 500 updates | ~4.5–5 h |
| 1,000-step continuation (24-block / 30-block) | ~7.5 h / ~9 h |
| 2,000-step continuation (24-block / 30-block) | ~15 h / ~18 h |
| Staged layerwise 50→30 (Path B) | ~20 h |
| DMD2 port (Path C) | 1–2 dev-days + ~4 h pilot |

All well inside the authorized budget and the week horizon — **provided the LR
is fixed first**. The most expensive mistake available right now is running the
prepared 200→500+ continuation at 1e-6.

---

*No expensive jobs were launched and no cluster state was modified during this
review. All GPU specs above are proposals pending approval.*
