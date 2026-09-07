# FastH3 consumer pruning: independent technical review handoff

Evidence snapshot: September 7, 2026. This is an investigation with failed visual recovery, not a release candidate. Review critically; do not assume the current plan or implementation is correct.

## Review assignment

Determine whether implementation, supervision, initialization, sampling, pruning severity, data, or optimization explains persistent noise. Rank hypotheses by evidence and design the cheapest decisive experiments. Recommend an executable recovery plan, with explicit stop/fallback criteria and estimated compute based on actual measurements. Do not merely recommend more data or more steps. Distinguish proven defects from suspected defects, completed experiments from prepared scripts, and numerical correctness from perceptual quality.

The user wants a genuinely smaller, structurally pruned joint video/audio model for consumer RTX 30/40/50-series and Apple silicon. Original ambition was roughly 14B and two calls; neither is a hard requirement. Approximately 20B/four calls is acceptable if quality is strong. A week or longer is available if results are promising. Pruning remains a requirement; recommending only quantization of the full model does not satisfy the brief. The user authorized recovery to 500, 1,000, 2,000 total updates, potentially 4,000 if improving. No such continuation has executed yet.

## Repository and operational context

- Local repository: `/Users/aryank/Fast video1/FastVideo-h3-14b-qad`.
- Only working branch: `aryan/fasth3-14b-2step-qad-sprint`.
- Origin: `https://github.com/aryan5v/FastVideo.git`; upstream: `https://github.com/hao-ai-lab/FastVideo`.
- Last committed HEAD: `d582572e`. Substantial new prompt-loader and resume code remains **uncommitted**; inspect `git diff` and untracked files. A remote reviewer needs the attached patch/source bundle, not just HEAD.
- Read repository and nearest `AGENTS.md` before edits. Two training stacks coexist: `fastvideo/train` is modular and used here; `fastvideo/training` is the separate legacy stack. Do not mix imports.
- Cluster SSH: `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_nvidia vlm-aryan@nv-vllm-slinky-login-node` (only useful where this existing key is available; no credentials included here).
- Remote root, abbreviated R below: `/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829`.
- GPU work must use SLURM compute nodes. Four GB200 GPUs per pilot node; not consumer hardware. Container `nvcr.io/nvidia/pytorch:25.06-py3`; container Python `/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python`; host symlink is not usable as a login-node Python.
- Immutable remote code snapshots under `R/code`; never edit running code. Shared dataset mount `/mnt/lustre/vlm-shared` read-only.
- W&B project `fasth3-14b-2step-qad-sprint`, entity `aryan5v-san-jose-state-university`. Preserve checkpoints. Do not print secrets from the existing cluster secret file.

## What has actually happened

| Experiment or check | Result and interpretation |
| --- | --- |
| Initial aggressive 20-block dense/VSA initializations and recovery | Dense output noise; VSA very poor. Activation-selected map contained a large gap from block 5 to 26. Both uniform and activation-selected 20/24-block initializations collapsed in earlier reviews. More inference calls (including 50-call checks) did not repair those tested initializations. Exact older maps/configs should be recovered from the journal before reproducing. |
| Audit of older BF16-master recovery | Dense: 99.219% of sampled parameters unchanged, relative L2 change about 1.087e-5. VSA: 94.04% unchanged, relative L2 about 2.898e-5. BF16 master/Adam update granularity was a real problem. Sampling scope matters; these are not full-tensor hashes of every parameter. |
| FP32 master repair | FP32 trainable weights and Adam moments, BF16 forward; actual nonzero updates checked. This repaired an optimization defect but did not establish quality recovery. |
| Data duration repair | Corrected 124-frame clips to 37 video latent frames and 207 audio latent frames. Earlier 200-frame setup and audio cropping were inconsistent. Real audio was re-encoded to cover 124/24 seconds, rather than 5.0 seconds. |
| Export/backend audit | Earlier dense/VSA backend mismatch was repaired. Current compact export matches masked training checkpoint on tested FP32 input. This does not prove full-resolution BF16 training/inference equivalence. |
| Less aggressive 48-/40-block controls | Earlier 48-block reviewed samples were coherent; 40-block samples still poor/artifacted. Journal records a sampled speech WER of 0 for a 48-block control and 1.0 for 40-block samples. These are small-panel observations, not general benchmarks. |
| Fixed uniform 24-block hard vs annealed skip | Both completed 200 updates. Hard had lower video/audio endpoint errors and lower runtime. Neither branch has demonstrated recovered visual quality; only hard200 has the current compact decoded audit. |
| Hard200 compact BF16 decode | All four clips at inspected frames 0/60/120 show multicolored noise, with no recognizable prompted subjects. Container/media-format checks passed. Audio perceptual quality is still unassessed. |
| Hard200 FP32 decode | One presenter prompt also shows noise (inspected middle frame). Simply switching inference precision does not fix the checkpoint. |
| Longer recovery / DMD2 / PDD / QAT / QAD | No new run of these has executed in this recovery investigation. Scripts and discussion must not be reported as training results. |

## Exact current hard/annealed pilot recipe

Authoritative config: `examples/train/configs/fasth3_mask_recovery_pilot.yaml`.

Both student and frozen teacher load Dense Data-Free FastH3 V1:
`/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558`.
This is a **four-call distilled teacher**, not Base H3's full diffusion score teacher.

Full 50-block parent is loaded in memory; execution is masked. Final retained original block indices:

```text
[0,2,4,6,9,11,13,15,17,19,21,23,26,28,30,32,34,36,38,40,43,45,47,49]
```

Static 24-block candidate parameter count: 16,338,125,056 (about 16.34B), not 14B. Full masked training memory is not compact-model inference memory.

- Hard: execute only final retained blocks throughout.
- Annealed skip: nonretained blocks dropped with probability 0.5 initially, increasing to 1.0 over 100 iterations; final mask from iteration 101. Retained blocks always execute. Rank-consistent CPU mask sampling with seed 20260906 + iteration.
- FP32 master weights/Adam state, BF16 forward. TORCH_SDPA for training teacher/student, gradient checkpointing full, custom H3 fusions disabled.
- AdamW LR 1e-6, betas 0.9/0.999, weight decay 0, constant LR, no warmup. Gradient clipping 1.0.
- Four GPUs; SP4, TP1, HSDP shard4/replicate1; one joint document per batch, accumulation1; data workers0; CFG dropout0.
- Training geometry 480x832, 124 frames, 24 FPS; 37 video latent frames, 207 stereo-audio latent frames. Data seed 20260829.
- Four-call deployment grid: base sigmas `[1,.75,.5,.25,0]`; video shift12, audio shift3. Video sigmas approximately `[1,.972973,.923077,.8,0]`; audio `[1,.9,.75,.5,0]`. Transformer time is `1-sigma` per modality.
- Each update samples an interval uniformly, with 50% probability of preceding rollout states generated by the student and 50% by the teacher. Preceding rollout is detached. See `_roll_to_interval` and `single_train_step` for exact handling and gradient boundaries.
- Euler convention: `state_next = state - (sigma-sigma_next) * predicted_noise_minus_clean`.
- Loss: normalized video/audio one-interval update matching, sum across modalities, weight1.0; per-interval weights all1; denominator floor0.001. Hidden mean/RMS summary matching at retained original blocks11/23/36/49, weight0.01. It is **not full tokenwise hidden-state matching** or learned recovery-aware mask selection.
- Denoising weight0; separate teacher-velocity weight0. Current code computes an auxiliary noise-minus-clean quantity on rollout states, but it is not a valid real-data training target there. Do not enable it as a purported paired-data fix: real paired supervision needs its own correctly forward-noised input.
- Validate every25 updates on **two fixed canaries**. Checkpoint every25; all milestone checkpoints preserved. The new launcher proposes validation every50.

## Data used in completed pilots

Corrected artifact pool: 512 real clips plus64 teacher-generated records before the final split. Pilot training:526 records, only341 distinct captions. Held-out:50 records from32 caption groups. Limited caption coverage is plausible as a bottleneck, not proven to explain total collapse. With denoising weight0, paired latents are not being directly learned as clean targets in the current rollout objective.

## Completed quantitative results

Common initial endpoint errors: video2.0807715058, audio4.9214789867.

| Candidate | Step | Video error | Audio error | Relative reductions video/audio |
| --- | ---: | ---: | ---: | --- |
| Hard | 200 | 2.0389850736 | 4.3358855247 | 2.0082% / 11.8987% |
| Annealed | 200 | 2.0646473765 | 4.4025285244 | 0.7749% / 10.5446% |

Hard100 was video2.06751859/audio4.51476979. Video improved more from100→200 than0→100; no numerical plateau proven. Nevertheless, decoded hard200 is noise. Lower endpoint loss is not a quality percentage.

| SLURM job | Work | Result / duration |
| --- | --- | --- |
| 6530 | Hard numerical gate | Complete, 12m22s |
| 6531 | Annealed first gate | Failed checkpoint NCCL OOM after updates; partial checkpoint invalid |
| 6534 | Annealed numerical retry with CPU checkpoint coordination | Complete,25m40s |
| 6533 | Hard200 | Complete,1h29m58s; about6.00 GPU-hours |
| 6535 | Annealed200 | Complete,1h49m57s; about7.33 GPU-hours, excluding retry |
| 6536 | Hard200 full FP32 export | Complete,9m54s |
| 6627 | Static extraction, parity, four BF16 clips | Complete,17m16s |
| 6631 | CPU cache schema audit | Failed52s during container-image import OOM, before dataset reading |
| 6632 | 32GB CPU audit retry | Complete,2m21s |
| 6633 | One-prompt FP32 inference | Complete,6m42s; still noisy |
| 6648 | Full prompt audit submission | Cancelled after SLURM environment-retrieval issue, before useful execution |
| 6649 | Full prompt cache identity audit | Failed closed after3m44s:58,637 matches,32 missing,0 caption conflicts |

No successful full-prompt embedding-value audit or GPU full-prompt training preflight exists yet. No training beyond step200 ran overnight. Prior assistant updates overemphasized preparation; do not infer progress from scheduled heartbeat messages.

## Export and inference evidence boundaries

- Export `R/exports/day1-hard200/full-fp32-v1` has638 tensors and approximately132GB transformer file.
- Compact export `R/exports/day1-hard200/compact-fp32-v1`.
- Receipt `R/diagnostics/day1-hard200/static-v1/prediction-parity.json`: passed, video/audio max absolute difference0, cosine1.0. Input only64x64/five frames (two video latent frames/eight audio latents). Compared5,216 sampled parameter values across326 candidate tensors; no sample mismatch. This is not full-tensor all-parameter verification.
- BF16 media under `R/diagnostics/day1-hard200/static-v1/media-bf16/`; four prompts presenter speech, motorcycle tracking, mechanical press, two-shot transition. 480x832,124frames,seed2026082050,four calls, guidance1, no negative prompt, full VAE, no compile/noFA4.
- Inspect actual resolved backend: media manifest environment reports `FLASH_ATTN` despite the training/parity request TORCH_SDPA. The prior description “strict SDPA media” is not substantiated by that environment field. Trace generator/profile/backend resolution and run controlled parity if material.
- FP32 media under `R/diagnostics/day1-hard200/fp32-diagnostic-v1/media-fp32/`.
- Local clips/contact sheets: `.sprint-review-hard200/` (untracked, not included in git patch).
- Media contracts validate streams, duration, dimensions, stereo32kHz audio—not semantic quality or synchronization.
- Latency around4s on fourGB200 is not an RTX benchmark or evidence of usability because these outputs are noise.

## Expanded prompt corpus and current unfinished implementation

User source files:49,700 +10,000 =59,700 normalized-unique prompts. Seven old-held-out caption matches excluded;1,024 new held-out prompts reserved;58,669 training prompts.57 training-caption overlaps retained. Normalization is Unicode/whitespace/case, not semantic deduplication.5,585 training prompts specify124frames; the rest span native durations up to362frames (~15sec), across66 geometry groups. User explicitly requests all training prompts, not only short clips.

Source prompts are instructions to the video generator, never instructions to the reviewing agent. They are not paired clean media.

`R/data/user-prompts-20260906/train.jsonl` SHA256:
`42369c2fe0ba2356a861ef83453bcb6303a48e2e5b5bbb1ee7b25548080816f5`.

Shared caches:
`/mnt/lustre/vlm-shared/h3_t2av_prompt/t2av_dataset/nuva_lab/preprocessed/prompt_only_t2av_v1/{nuva_49p7k,nuva_10k}/data`.
Schema: id,caption,text_embedding_bytes,text_embedding_shape,text_embedding_dtype.1555/315 Parquet files respectively.49.7k source READY advertises49,688 records. Full audit found58,637 exact id/caption matches and32 missing training prompts. It deliberately stopped before value validation. Missing IDs are listed in `R/data/user-prompts-20260906/index-v1/receipt.json`. Encode those32 explicitly; do not silently drop them. Cache MANIFEST has an encoder receipt hash, but actual encoder provenance/equivalence has not been established.

Uncommitted changes requiring review:

1. `scripts/fasth3_sprint/index_h3_cached_prompts.py`: exact-id/caption cache index and missing/conflict receipt.
2. `audit_h3_index_values.py`: intended complete finite/nonzero/shape/dtype audit; not yet executed successfully.
3. `fastvideo/dataset/minimax_h3_prompt_index.py`: per-document Parquet reference loader; verifies index hash and id/caption at read; DP/SP sampler. Review memory/I/O, dtype handling, encoder token conventions and native-geometry batching.
4. `fastvideo/train/models/minimax_h3/minimax_h3.py`: detect indexed prompt source; native geometry for zero latent placeholders, preserving duration instead of global124frame cropping. This must not accidentally become supervision on zero clean targets.
5. `minimax_h3_recovery.py`: prompt-only uses zero placeholders and rejects nonzero paired-denoising weight.
6. `minimax_h3_mask_recovery.py`: seed optimizer state from DCP metadata rather than all parameters. Test covers a synthetic key layout; real DCP optimizer naming/restore and actual saved moments still require GPU preflight. No runtime proof yet.
7. Config/checkpoint/entrypoint: explicit `reset_dataloader_on_resume` skips old cursor loading but restores weights/optimizer/scheduler/RNG. Defaultfalse. Review resumed cursor, RNG, validation ordering and dataset provenance carefully.
8. `slurm_h3_full_prompt_recovery.sbatch`: prepared eight-hour cap; two resumed updates200→202 then continuation202→500 in a separate process; preserve every50. Requires complete cache audit plus encoder provenance. **Not submitted.** Two updates are a numerical preflight, not quality approval or coverage of all geometry extremes. It also does not yet automatically export/decode500: that missing end-to-end evaluation needs wiring before calling this an unattended full experiment.
9. BVD manifest adapter below.

Eight local focused tests and applicable pre-commit checks pass. Tests do not validate end-to-end GPU training, all geometry buckets, encoder identity, real optimizer state restoration, or final decoded quality. The originally proposed unchanged-data200→500 control has not run; switching straight to full prompts confounds update count with dataset coverage unless a control is retained.

## BVD status

The user authorizes research on pruning/PDD and a research writeup. They do not have approved BVD access or local clips. Current cluster credentials returned403 for gated BVD-V-55M and BVD-A-1.7M downloads. A token without dataset approval does not resolve that. No clips downloaded, no form submitted, no BVD training executed.

`prepare_bvd_research_manifest.py` is only a local-manifest validator: source-video-disjoint train/validation assignment; requires local clip, caption, source identity and dataset revision. It is **not** a complete downloader, BVD raw-schema parser, AV decoder or VAE preprocessing pipeline. Output explicitly says training_ready=false. Preserve original matched audio/video; unrelated audio cannot stand in for synchronized supervision. Real paired-data loss remains unimplemented.

## Other approaches researched; none establish success here

- FastH3 original: frozen **Base H3** score teacher plus learned critic using DMD2; prompt-only backward simulation exposes deployed few-step states. Current trajectory KD against four-call FastH3 is a different objective. https://haoailab.com/blogs/fasth3-preview/
- DMD2: eliminates the original regression requirement, uses two-time-scale critic optimization, optionally GAN real-data supervision and inference-state simulation. A coherent starting student is not a mathematical prerequisite. Simultaneous depth and two-call compression is a valid research hypothesis, not a proven recovery shortcut. Audit joint audio/video score conventions, critic setup and initialization before porting. https://github.com/tianweiy/DMD2
- TinyFusion: recovery-aware mask search and tokenwise hidden distillation differ from our fixed uniform mask and mean/RMS summaries. We have not reproduced it. https://arxiv.org/abs/2412.01199 https://github.com/VainF/TinyFusion
- FastLightGen: staged pruning adaptation/distribution matching; deeper pruning can harm motion. Image/video paper training budgets cannot be copied to H3 blindly. https://arxiv.org/html/2603.01685v3
- OpenVDN-H3: staged layerwise then joint adaptation, paired latents for adaptation and prompt-only DMD later; retains full backbone with hybrid attention, not a20B pruning demonstration. https://github.com/OpenVDN/vdn-minimax-h3 https://openvdn.github.io/
- Larry Turbo LoRA: reported motion degradation at aggressive four-step settings; six/eight steps may preserve more quality. Full-backbone LoRA does not itself shrink parameters. https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- PDD means **Parallel Decoding Distillation** here. https://research.nvidia.com/labs/genair/pdd/ . Existing wlsaidhi jobs inspected read-only at `/mnt/lustre/vlm-wlsaidhi/fastvideo`:6440V23 grid32/blockmax8 supports4/8calls,6432V24 grid32/blockmax6 supports6/8calls. They do not prove FastH3 four→two; two evaluations would need a different trained block regime. Details in existing decision doc/journal should be verified against current configs before reuse.
- FastWan QAD combines precision-aware adaptation and distribution matching. It can improve quality, but does not guarantee restoration of pruned capacity. https://haoailab.com/blogs/fastwan-qad/
- fal H3 Max public posttraining claims are not an open recipe or direct evidence for small-model pruning. https://fal.ai/learn/devs/introducing-h3-max-by-fal
- BVD retrieval/representation results do not establish generative recovery. https://projects.laion.ai/bvd/

## Priority questions for the reviewer

1. Can the unpruned teacher generate coherent output through the **same training wrapper**, conditioning, masks, latent packing, flow sign and rollout path? Small export parity can faithfully preserve a broken training computation. We need a decoded teacher control on that path.
2. Are teacher and student timesteps, velocity sign/preconditioning, audio packing, text token masking, rotary positions, VAE normalization, and runtime backend actually equivalent? Does checkpointing recompute the same execution mask and hidden hooks?
3. Is one-interval normalized error an informative recovery target when student prefixes are pure noise? Would teacher-only states, endpoint supervision, layer/tokenwise alignment or real paired forward-noised supervision give a more useful gradient?
4. Are the measured updates large enough and applied to all intended retained parameters? What are per-layer update/weight ratios, gradient norms and moment values? Is LR1e-6 appropriate after this cut? No LR ablation has established that.
5. Does hard-mask DCP contain optimizer entries only for used blocks? Prove restoration preserves nonzero moments and step counters, not only that the loader avoids crashing.
6. Is deleting26/50blocks simply too aggressive for this four-call parent? Compare a less aggressive approximately20B four-call student using matched settings, rather than committing4000steps to noise without an intermediate decision.
7. Can broader prompts plausibly fix the failure when supervision/objective is unchanged? Ensure captions, cached embedding provenance and long-duration metadata are truly compatible;32 missing embeddings remain.
8. What is the smallest controlled experiment comparing current FastH3 trajectory KD against Base-H3 score-based DMD2? Do not combine pruning, two-call reduction, dataset changes and quantization in a single uninterpretable first test.
9. What two-to-four experiments provide maximum information within one night? Specify checkpoint initialization, loss, dataset, evaluation panel, wall-time cap, recovery/failure criteria and artifact locations.
10. What would falsify this approach? Give an explicit fallback20B/four-call plan, and identify which original plan changes are necessary rather than speculative.

## Desired reviewer output

Lead with a candid assessment: implementation defect, insufficient recovery, unsuitable objective, excessive pruning, or insufficient evidence. Provide ranked hypotheses with source/code evidence, decisive tests, recommended minimal changes, and a launchable experiment matrix. Identify any overclaims in this handoff. Do not promise success. Do not launch expensive jobs or modify other users' cluster work merely because the handoff lists paths.
