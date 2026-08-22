# MiniMax-H3 / FastH3 performance numbers (measured, GB200)

This is a dated results ledger. Sections 1-4 and historical section 8 preserve the measurements
through 2026-08-21 at their row-local SHAs; section 5 is the job-2666 merged-preview baseline,
section 6 is the current 5-second matched-serving refresh, and section 7 is the matched 5/10/15-
second T2VA and Ref2VA duration grid. Unless a row says otherwise, the
FastVideo runs use the lustre venv (torch 2.12.0+cu130), driver 580.82.07, synth64 prompt 0, seed
1000, warmup excluded, and 2-3 timed repeats. "FastH3" = the 4-forward DMD2 student (v8 data-free
step-1400 preview export) with VSA @0.9; it does not mean the live v10 training run. Raw artifacts:
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/`
`{sm100a_e2e,sp8,ref2va_grid,ref2va_duration_grid_20260822,vsa64bench,`
`preview_merged_20260822,h3_vllm_match_20260822,h3_vllm_match_varlen_99cd_20260822,`
`h3_duration_grid_20260822}` and `vllm_omni_bench/`; master index
`~/h3-results-index.md`.

## 1. T2VA @ 5s (768x1344, 124 frames, S=38,224)

### End-to-end (seconds) — jobs 2518 (1x), 2519/2555/2556 (SP-4), 2564 (SP-8)

| config | 1x GPU | SP-4 (one tray) | SP-8 (2 trays, external launcher) |
|---|---:|---:|---:|
| Base, dense FA4, 50 steps | 185.2-188.2 | 62.8-63.7 | **43.2** |
| FastH3 4-fwd, VSA-64 Triton | 22.3-22.8 | 13.6-15.0 | 16.1 (worse — fixed-bound) |
| FastH3 4-fwd, VSA-64 **sm100a** | **20.7** | 15.1 | 16.3 |
| FastH3 4-fwd, VSA-256 CuTe | 20.7 | — | — |

- Speedups (dense-50 → FastH3): **8.3-8.9x at 1 GPU, 4.2-4.6x at SP-4**.
- Denoise-only: dense 175-179 s (1x) / 52.9-54.2 (SP-4) / 30.4 (SP-8); FastH3 10.7 (1x sm100a) /
  3.9 (SP-4 sm100a) / 2.05 (SP-8 sm100a). SP-4→SP-8 denoise scaling 1.78x.
- Per-forward: dense FA4 3.58 s (1x) / 1.08 (SP-4); VSA-64 Triton 3.10 / 1.09; sm100a 2.68 / 0.98.
- Fixed (non-denoise) cost ~9.2-9.3 s (SP-invariant): text encode ~2, video VAE decode 6.4-7.5,
  audio+save ~1. FastH3 at SP-4 is ~68% fixed-cost-bound; SP-8 makes the student WORSE
  (VAE decode replicated per rank). **SP-8 is the teacher's shape; SP-4 the student's.**
- SP-8 raw per-request (job 2564): dense e2e 42.51/42.17/44.88, denoise 28.99/29.81/32.33;
  FastH3-triton e2e 16.63/16.10/25.76*, sm100a 16.27/17.07/30.62* (*identical output hashes —
  stall outliers, suspected fabric/lustre interference from the co-resident 8-tray training job).
  Known anomaly: external-launcher mode adds ~3.2 s in `video_decoding_stage` vs the stock
  executor at equal world size (7.26→10.44 s); denoise unaffected; prime suspect torchrun's
  `OMP_NUM_THREADS=1` default.
- Sharding note: multi-node inference requires the external-launcher path
  (`FASTVIDEO_EXTERNAL_LAUNCHER=1`, torchrun; byte-identical gate vs stock executor; unmerged
  worktree commits `c65733463`+`ac24e7e78`).

### Cross-stack: vllm-omni (same workload, FA4 verified, their commit 73b623f2)

| config | e2e | FastVideo equivalent | ratio |
|---|---:|---:|---:|
| 1 GPU dense FA4 50-step (no offload) | **135.8 s** | 188.2 s | 0.72x |
| 1 GPU + CPU offload | 140.7 s (82.6 GB peak) | — | — |
| 4 GPU (USP-4 + text-encoder TP-4 + VAE patch-parallel + regional compile) | **40.7 s** | 62.8 s | 0.65x |

Per-step: ~2.69 s (1 GPU) / ~0.79 s (4 GPU). Boot→healthy 110 s (1x) / 310 s (4 GPU incl.
regional-compile warmup 63.5 s); 1x warmup 157.5 s. Their 4-GPU margin comes from regional
compile + parallel VAE decode + encoder TP — our #1718 compile port (unenabled) and the
#1703-seam parallel decode (unbuilt) are the corresponding levers. (vllm 0.26.0 wheels,
vllm-omni @73b623f2, isolated venv `vllm_omni_bench/venv`; jobs 2562/2563.)

## 2. Ref2VA @ 15s (345 frames, 768x1344 ref + target, S=220,628)

Historical roofline/prototype study using base `transformer_ref` weights; full grid in
`vsa_gate/ref2va_grid/ROOFLINE.md`. Section 7 is the current matched duration grid and identity
contract.

| leg (SP-4 unless noted) | fwds | DiT s/fwd | e2e |
|---|---:|---:|---:|
| dense FA4 50-step, 1x GPU (fits: 99.6 GiB) | 49 | 81.5 | 4046 s |
| dense FA4 50-step | 49 | 21.0 | 1084 s |
| VSA@0.9 **Triton-256, current policy — DO NOT USE at this scale** | 4 | 37.6 (1.79x SLOWER than dense) | 203 s |
| VSA@0.9 tile-64 sm100a, current policy (P1) | 4 | 17.3 (1.22x) | 121 s |
| **P2 ref-sparsified keep-0.10, CuTe-256 (50-step)** | 49 | **9.2 (2.34x denoise, 2.17x e2e)** | **513 s** |
| P2b (+semantic-ref trim to ~300 text), attention-layer measured | — | 18.6 s attn-fwd @1x = **4.19x vs dense** | projected |

- Component split at 221k dense: **attention 91.5% of DiT wall** (89.1% FLOPs), MLP 5.4%, QKV/O 3.1%.
  At 38k: attention 62.9%. Dense FA4 effective throughput at 221k: 0.96-0.98 PF/s.
- Fixed block **51-53 s** (GPU-invariant): ref VAE encode 26.6-27.2, VAE decode 19-21, text ~2
  (incl. ~12.6k Qwen3-VL semantic-ref tokens), save ~1.5.
- P2 sits ON the kernel-bounded ceiling (FLOP 4.95x; sparse-keep kernel eff 1052→661 TF).
  P2b (semantic-ref trim, probe says ~3% mass) reaches 4.19x per-forward.
- Probe: target queries put 0.156 mass on ref video, no ref-specialist heads; ref keep 0.10 OK
  for trained students, 0.25 for zero-finetune. 4-fwd student projection at P2: ~87 s e2e,
  ~60% fixed-bound → **VAE encode+decode (46.5 s) is the dominant remaining lever at 15s**.

## 3. Kernel level (block-sparse attention, 25% density, bf16, B=1 H=8 D=128)

sm_100a CUDA (merged #1719 + per-tile-count fix) vs Triton-64, blk64:

| S | 4k | 8k | 16k | 32k | 65k | 131k | 163k | 200k |
|---|---|---|---|---|---|---|---|---|
| speedup | 3.14x | 1.97x | 2.10x | 2.10x | 2.04x | 1.99x | 1.91x | 1.85x |
| sm100a TFLOPS | 498 | 780 | 1026 | 1076 | 1093 | 1072 | 1020 | 993 |
| blk128 TFLOPS | 645 | 930 | 1131 | 1293 | 1271 | 1274 | 1270 | 1268 |

Triton-64 saturates ~536 TF from 65k (job 2558; parity vs Triton max|diff| 0.002; sm100a
holds ~1000+ TF to 200k with ~7% taper past 131k). H3 layer path (56 heads, prod 5s shape, sparsity 0.9):
19.2 ms (Triton-64) → 11.4 ms (sm100a, 1.68x). Dense-50-vs-3-step legacy headline (VSA-256,
124f): 4.66x e2e / 14.1x denoise (job 2432 era).

## 4. Training throughput (v8, for context)

Job 2486: 32 GPUs, global batch 64, ~67-69 s/step (carried backward-simulation walk + VSA-64
Triton student; v7 was ~81 s/step, v6 dense ~same MFU). MFU tables and accounting:
`h3_dmd.md` MFU section + `scripts/train/mfu_calc_minimax_h3.py`.

## 5. Preview checkpoint on the fully merged H3 stack (job-2666 baseline)

This was the merged-preview serving baseline measured by Slurm job 2666 (`COMPLETED 0:0`) on
`hpc-rack-2-0` (4x GB200). Runtime code was `internal/h3-dmd` at exact HEAD
`a5384533011f4641830dfde3ab105f79a600d8a0`; its merged-main parent is `2f3d40740`. The internal
branch later advanced to `c26fec419` for the H3 validation-decode ownership bugfix; that later tip
is not the code measured in this section.
The checkpoint was the local `FastVideo-Minimax-FastH3-Preview-v0.1` export: the v8
data-free step-1400 preview, **not** a v9 checkpoint. Provenance hashes are
`74c11bff...` (`transformer/config.json`), `63a5c56b...` (`modular_model_index.json`),
and `ab3a8677...` (transformer safetensor index).

### Merged-PR coverage

All nine H3 PR merge commits are ancestors of the measured HEAD:

| PR / merge SHA | contribution | disposition in job 2666 |
|---|---|---|
| #1362 `fca45bc8e` | uint8 conversion before post-decode D2H | active in every leg |
| #1703 `e0a3db565` | streaming / lower-peak H3 VAE | active in every leg |
| #1711 `0462e1b0e` | stop Qwen3-VL construction after the last consumed layer | active in every leg |
| #1719 `907f2100e` | Blackwell sm100a block-sparse forward | active through the asserted kernel prefix |
| #1730 `56d4a6074` | corrected Triton sparse backward scaling | present, but inference-inert |
| #1731 `6d6a10be7` | preview example plus tile-64 / sm100a H3 route | active in every leg |
| #1732 `bcffa4026` | slim conditioner plus optional serialized FP8 | slim path active; FP8 deliberately off |
| #1734 `2f3d40740` | VAE dispatch/compile/NVTX | defaults in all legs; decoder compile on in p1/p2 |
| #1735 `73dd105f3` | opt-in Sol-Engine H3 fusions | p2 only; disclosed non-parity |

Method: 768x1344x124, the same synth64 prompt 0 repeated three times, fixed timed seed 1000,
one untimed seed-999 warmup, VSA@0.9 tile-64, and the public scheduler's native five-point
shift-12 schedule (**four** transformer forwards). `FASTVIDEO_VSA_SM100A=1` selected the
Blackwell forward; `FASTVIDEO_FA4=1` supplied FA4 for eligible non-VSA paths. p0 is the merged
eager baseline, p1 adds `enable_torch_compile_vae`, and p2 adds
`FASTVIDEO_MINIMAX_H3_FUSIONS=1`. Model load and the one warmup are excluded from every number.

The launcher failed fast on SHA, module, and route checks. The resolved kernel was
`vsa_gate/sm100a_main/prefix/fastvideo_kernel/block_sparse_attn_sm100a.py`, all six legs
selected VSA-H3, and none logged the sm100a-to-Triton fallback. Jobs 2664/2665 are preserved but
invalid (shadowed kernel package; then a non-fail-fast preflight) and contribute no numbers.

### End-to-end / denoise (seconds, mean of 3, range in parentheses)

| leg | 1x e2e | 1x denoise | 1x peak GiB | SP-4 e2e | SP-4 denoise | SP-4 peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| p0. fully merged, eager VAE | 21.12 (20.73-21.61) | 10.71 | 81.2 | 14.30 (14.10-14.47) | 3.67 | 32.0 |
| **p1. p0 + compiled VAE** | **18.52 (18.07-18.94)** | 10.69 | 76.6 | **10.52 (10.43-10.61)** | 3.66 | 27.4 |
| p2. p1 + H3 fusions (**non-parity**) | 15.82 (15.59-16.21) | 8.58 | 76.6 | 11.38 (10.91-11.93) | 3.13 | 27.4 |

### Timed stage split (seconds, mean of 3)

| shape / leg | text | video decode | audio decode | post-decode | save |
|---|---:|---:|---:|---:|---:|
| 1x p0 | 0.56 | 8.69 | 0.18 | 0.47 | 0.35 |
| 1x p1 | 0.56 | 6.10 | 0.19 | 0.32 | 0.35 |
| 1x p2, non-parity | 0.51 | 5.85 | 0.18 | 0.21 | 0.35 |
| SP-4 p0 | 0.42 | 8.91 | 0.19 | 0.48 | 0.37 |
| SP-4 p1 | 0.42 | 5.20 | 0.26 | 0.39 | 0.35 |
| SP-4 p2, non-parity | 0.42 | 6.31 | 0.26 | 0.67 | 0.36 |

p1 is the best merged, parity-gated serving configuration: against p0 it removes 2.60 s
(12.3%) at 1x and 3.78 s (26.4%) at SP-4, while denoise is unchanged. The gain is decoder-local:
video decode falls 8.69 -> 6.10 s at 1x and 8.91 -> 5.20 s at SP-4. SP-4 p1 is now 35% denoise
and 49% video decode, so further attention-only work has limited end-to-end leverage. p2 improves
1x further, but at SP-4 its 0.53 s denoise win is outweighed by a 1.11 s decode regression plus
post-decode jitter; it is 8.2% slower end-to-end than p1 and remains a non-parity diagnostic.

Cold observations (model load / first warmup, excluded) were: 1x p0 45.6/27.5 s, p1
47.4/62.8, p2 197.2/32.9; SP-4 p0 63.4/26.8, p1 63.4/31.0, p2 71.6/31.2. The legs ran
sequentially and shared compiler caches, so these are audit data, not an independent cold-start
benchmark; notably the one 1x p2 load spent an additional 150 s in fusion-weight preparation.

### Determinism and parity envelope

- Every shape/leg produced three byte-identical same-seed MP4s. Hash prefixes: 1x p0
  `b13fd65328f1`, p1 `e9998738368c`, p2 `3997c8882652`; SP-4 p0 `f41cb9e561c4`, p1
  `acd3a2829fa5`, p2 `2c2bf768621a`. This proves repeat determinism, not cross-leg parity.
- p0 and p1 are **not byte-identical**. Their denoiser debug summaries match line-for-line and
  decoded audio PCM hashes match exactly; only the compiled video decoder changes numerics.
  On the muxed outputs, p0-vs-p1 is SSIM 0.974917 / PSNR 41.649944 dB at 1x and SSIM 0.975035 /
  PSNR 41.704310 dB at SP-4.
- The intended #1734 component gate is job 2655 on its final head `4c17beba2`; the VAE
  implementation and tests are byte-identical in merged `2f3d40740`. Compiled-vs-eager decode
  measured max/mean absolute error `1.9531e-03` / `1.0542e-04`; encode was exact, outputs were
  repeat-deterministic, and 39 VAE/compile suites passed. Thus p1 is tolerance-parity-gated,
  **not** bit-exact with p0.
- p2 is repeat-deterministic but intentionally non-parity: p1-vs-p2 output SSIM / PSNR is
  0.686541 / 18.754262 dB at 1x and 0.680668 / 18.647198 dB at SP-4; audio PCM changes too.
  Fusion engagement (`modulate,qknorm_rope,swiglu`) is present in both p2 worker logs.

Artifacts: `vsa_gate/preview_merged_20260822/{RESULTS.md,job_current_preview.sbatch,logs/job_2666.out}`;
raw JSON/log/video trees are under `runs_valid/{1x,sp4}/`, representative videos under `videos/`,
and excluded-run provenance in `INVALID_RUNS.md`. Harness SHA `d85b5cc6...`; prompt-file SHA
`b1f21b18...`.

## 6. Current matched serving matrix: base H3 and FastH3 (2026-08-22)

This section supersedes the serving headlines in sections 1 and 5. The final timing checkout was
the clean `integration/h3-vllm-parity-20260822` tree at exact SHA
`99cd355a2452ce040591fe54ef340d192e26fe48`; its merged-main base is
`2f3d4074064e4d86f99dc784ebbaa296e6f5925f`, so all nine merged H3 PRs listed in section 5 are
included. The integration tree additionally carries regional inference compile, temporal-parallel
VAE, compile-safe fusion work, the attention compile-scope correction, and packed-varlen FA4.

The isolated PR candidate for the last item is branch `perf/h3-fa4-packed-varlen-20260822` at
`f5c48f738aa95e56d691a7c5d8ee80ea4d8580d8`: performance commit
`d81513b40703cc5c61920c9f718d70a3ff2bce08` directly on merged main, plus a route-evidence/test
hardening commit. Job 2755 ran that exact PR head and finished **11 passed, 10 deselected, 0
failed**. The end-to-end timings below belong to integration SHA `99cd355a`, not to the separately
extracted PR head; do not rewrite their provenance as an exact-head PR benchmark.

### Exact workload and execution contract

- Both stacks generated 768x1344x124 with guidance 1 from synth64 prompt 0, seed 1000, one
  seed-999 warmup excluded, then three timed repeats. Model load and compile warmup are excluded.
- The real attention length is **S=38,224**: 514 text + 414 audio (`207 * 2`) + 37,296 video
  (`37 * 24 * 42`) tokens. A captured FastVideo TorchInductor input is
  `bf16[1,38224,56,128]`. vllm-omni stores an aligned S=38,272 tensor but its cumulative lengths
  are `[0,38224,38272]` and `max_seqlen=38224`; the 48-token alignment tail is not part of the
  content sequence. The earlier S=38,010 label in this ledger was stale.
- Base H3 used the official `/mnt/lustre/vlm-k1kong/models/MiniMax-H3` checkpoint and a 50-point
  sigma grid, which the scheduler turns into exactly **49 transformer forwards**. FastH3 used
  `/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-Minimax-FastH3-Preview-v0.1`, the v8
  data-free step-1400 Preview checkpoint, and its native five-point grid: exactly **four
  transformer forwards**. Each run's `schedule_contract.json` records the points and observed
  denoiser-call count.
- FastVideo used a replicated DiT at both 1x and SP-4 (`use_fsdp_inference=false`). vllm-omni was
  commit `73b623f2f7db092053c1c86fe796bed89eb3dc71` plus the local two-line profiler-only patch; its
  SP-4 leg used USP-4, text-encoder TP-4, and tile-parallel VAE. Peak memory below is the recorded
  per-process/rank maximum, not an aggregate and not a layout-normalized comparison.

**These rows are not all `torch.compile` rows.** For base H3, d0 is eager and d4 regionally
compiles 52 DiT submodules (`fullgraph=True`, `emulate_precision_casts=True`) plus the VAE decoder;
vllm-omni also uses 52-module regional DiT compile (`dynamic=True`). For FastH3, regional DiT
compile is **off in every leg** because the VSA-H3 body is not fullgraph-traceable: f0 and f2 have
an eager decoder, while f1/f3/f4 compile only the VAE decoder. VSA layers use tile-64 sm100a at
0.9 sparsity; FA4 is used only on eligible non-VSA attention paths.

Leg definitions used below:

| leg | DiT forward | VAE decoder | SP-4 VAE | H3 fusions | gate policy |
|---|---|---|---|---|---|
| d0 | eager, fixed or packed FA4 (row-local) | eager | replicated | off | reference |
| d1 | eager, fixed FA4 | compiled | replicated | off | strict |
| d2 | regional compile, fixed FA4 | eager | replicated | off | report-only |
| d3 | eager, fixed FA4 | eager | temporal parallel | off | strict |
| d4 | regional compile, fixed or packed FA4 (row-local) | compiled | temporal parallel | off | report-only |
| d5 | regional compile, fixed FA4 | compiled | temporal parallel | compile-safe | report-only |
| d6 | eager, fixed FA4 | compiled | temporal parallel | eager-body | report-only |
| f0 | eager VSA | eager | replicated | off | reference |
| f1 | eager VSA | compiled | replicated | off | strict |
| f2 | eager VSA | eager | temporal parallel | off | strict |
| f3 | eager VSA | compiled | temporal parallel | off | strict |
| f4 | eager VSA | compiled | temporal parallel | on | report-only / non-parity |

Temporal VAE parallelism is a one-rank no-op in the 1x d3/d4/d5/d6 and f2/f3/f4 rows. The d1-d6
rows were run in the fixed-length attribution matrix; only d0/d4 were rerun on the final packed
route.

### Base H3 versus vllm-omni: exact 49-forward match

Seconds are arithmetic means of three timed requests; the parenthesized interval is the request
range. The current FastVideo d0 receipts are jobs 2708_0/2708_1, d4 is jobs 2701_0/2699_1, and
the matched vllm-omni profiler rebench is job 2657.

| implementation / leg | shape | e2e s | denoise s | s/fwd | video decode s | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| vllm-omni, regional + serial VAE | 1x | 136.036 (135.944-136.160) | 127.767 | 2.607 | 6.320 | 128.61 |
| FastVideo d0, packed eager | 1x | 161.064 (160.887-161.391) | 152.105 | 3.104 | 7.598 | 77.58 |
| **FastVideo d4, packed regional + VAE compile** | **1x** | **132.468 (132.213-132.686)** | **125.750** | **2.566** | **5.429** | **72.98** |
| vllm-omni, regional + USP/TP/tile VAE | SP-4 | 40.748 (40.615-40.825) | 37.222 | 0.760 | 1.733 | 93.32 |
| FastVideo d0, packed eager | SP-4 | 59.083 (58.576-60.047) | 43.673 | 0.891 | 13.614 | 77.59 |
| **FastVideo d4, packed regional + parallel VAE** | **SP-4** | **40.587 (40.440-40.752)** | **37.131** | **0.758** | **2.013** | **73.92** |

The matched target is reached at both shapes: FastVideo d4 is 2.62% faster end-to-end and 1.58%
faster in denoise at 1x; at SP-4 the differences are 0.40% and 0.24%, respectively. The SP-4
denoise result is therefore effectively matched, not evidence of a material speed lead. d4 is a
speed/quality-reporting route, **not a parity-safe route**, because regional compile and the packed
FA4 invocation change floating-point reduction order.

### Fixed-length attribution and the packed-varlen gain

Job 2675 (`COMPLETED`, fail=0) ran the full fixed-length matrix at exact SHA
`3a8f463d4406e1f6feddbd4f3a43b4d14f172645`. Every cell used the same 49-forward contract and
three timed repeats. Values are `e2e / denoise / video-decode` seconds:

| leg | 1x | SP-4 | parity evidence against d0 |
|---|---:|---:|---|
| d0 | 184.292 / 174.863 / 8.113 | 65.282 / 53.426 / 9.749 | reference |
| d1 | 181.723 / 174.969 / 5.579 | 60.587 / 53.262 / 5.518 | strict PASS: mean/min MS-SSIM 0.988096/0.977004 (1x), 0.988234/0.980846 (SP-4) |
| d2 | 157.036 / 147.078 / 8.641 | 54.130 / 44.800 / 7.792 | report-only (regional compile) |
| d3 | 183.063 / 174.734 / 7.033 | 57.612 / 53.322 / 2.992 | exact: MS-SSIM 1.0 |
| d4 | 153.409 / 147.078 / 5.121 | 47.944 / 44.734 / 2.038 | report-only (regional compile) |
| d5 | 155.993 / 150.061 / 4.715 | 48.654 / 45.408 / 2.045 | report-only (regional + fusions) |
| d6 | 157.532 / 150.960 / 5.328 | 51.806 / 48.556 / 2.087 | report-only (eager fusions) |

Across the fixed-run and final packed-run trees, d0 1x fell by 12.60% end-to-end / 13.01% denoise
and d4 1x by 13.65% / 14.50%. The observed SP-4 changes were 9.50% / 18.26% for d0 and 15.34% /
17.00% for d4, but those full-pipeline deltas are partially confounded: job 2675 used
FSDP/sharded SP-4 weights whereas the final matrix used replicated DiT weights, and the final
tree also contains the scoped-regional-compile follow-up. Job 2695 below is the isolated evidence
for the FA4 invocation itself.

The direct FA4 API A/B in job 2695 isolates the kernel call at the exact `S=38224`, bf16, D=128
shape:

| heads | fixed-length | packed-varlen | latency change | fixed-vs-packed max / mean abs error |
|---:|---:|---:|---:|---:|
| 56 | 36.873 ms | 28.008 ms | -24.0% (1.317x) | 2.4414e-4 / 1.7990e-6 |
| 14 | 9.236 ms | 6.386 ms | -30.9% (1.446x) | 2.4414e-4 / 1.7991e-6 |

The same packed H=56 call in the older vllm-omni environment was 28.106 ms, ruling out the FA4
package version as the material remaining difference.

Packed-varlen is repeat-deterministic but **not cross-route parity**. Job 2732 compared the fixed
and packed output sets with no acceptance floor:

| dense leg | shape | mean MS-SSIM | minimum MS-SSIM |
|---|---|---:|---:|
| d0 | 1x | 0.607577 | 0.424707 |
| d0 | SP-4 | 0.820703 | 0.737233 |
| d4 | 1x | 0.737616 | 0.573518 |
| d4 | SP-4 | 0.647129 | 0.462386 |

For all four cells, fixed prompt00/01/02 hashes were identical within the fixed run and packed
prompt00/01/02 hashes were identical within the packed run. The low cross-route similarity is
therefore deterministic numerical amplification over 49 forwards, not repeat noise. A
representative manual check of dense 1x d4 prompt00 frames 0/31/62/93/123 found the same coherent
anime rooftop/pinwheel story and comparable sharpness on both routes, with modest composition and
action differences and no obvious quality regression. That is a representative visual check, not
a full quality or parity gate; parity-sensitive serving should retain fixed FA4 until a deliberate
acceptance decision is made.

### FastH3 Preview: every compatible optimization leg, exact four forwards

Jobs 2707_2/2707_3 completed with fail=0 at exact SHA `99cd355a`. All rows use VSA@0.9 tile-64
sm100a; no row uses regional DiT compile. Seconds and memory are defined as in the base table.

| leg | shape | e2e s | denoise s | s/fwd | video decode s | peak GiB | current parity status |
|---|---|---:|---:|---:|---:|---:|---|
| f0 eager | 1x | 20.754 (20.111-21.705) | 10.716 | 2.679 | 7.530 | 81.20 | reference |
| f1 + VAE compile | 1x | 18.662 (17.723-19.713) | 10.718 | 2.680 | 5.788 | 76.59 | PASS, mean/min 0.985239/0.975648 |
| f2 + parallel VAE (1x no-op) | 1x | 19.890 (19.756-20.057) | 10.706 | 2.676 | 7.519 | 81.20 | exact, 1.0/1.0 |
| f3 + VAE compile + parallel VAE | 1x | 17.779 (17.171-18.747) | 10.695 | 2.674 | 5.391 | 76.59 | PASS, mean/min 0.985227/0.976443 |
| f4 + H3 fusions | 1x | 16.488 (16.053-16.716) | 8.686 | 2.171 | 6.110 | 76.59 | report-only / non-parity |
| f0 eager | SP-4 | 19.618 (18.523-21.558) | 3.473 | 0.868 | 12.082 | 81.20 | reference |
| f1 + VAE compile | SP-4 | 13.696 (12.705-14.375) | 3.462 | 0.866 | 7.936 | 76.59 | PASS, mean/min 0.985101/0.976192 |
| f2 + parallel VAE | SP-4 | 8.371 (7.984-8.966) | 3.476 | 0.869 | 2.780 | 82.05 | exact, 1.0/1.0 |
| **f3 + VAE compile + parallel VAE** | **SP-4** | **7.318 (7.207-7.396)** | **3.463** | **0.866** | **1.997** | **77.54** | **PASS, mean/min 0.985105/0.975796** |
| f4 + H3 fusions | SP-4 | **6.910 (6.763-7.088)** | **2.959** | **0.740** | 2.077 | 77.54 | report-only / non-parity |

The current f3/f4 timing outputs received `_1.mp4` suffixes because their directories retained
files from an earlier exact-head run. The job-2707 parity tail hard-coded unsuffixed filenames;
job 2780 therefore reran the comparisons against the actual timed `_1.mp4` files. Both f3 shapes
pass the 0.95 minimum-MS-SSIM gate reported above. Current f4 remains intentionally report-only:
mean/min MS-SSIM is 0.518061/0.391710 at 1x and 0.500509/0.358064 at SP-4. All ten current timed
triplets are byte-identical within their own leg; this proves repeat determinism only.

The fastest fully strict SP-4 result is therefore f3 at 7.318 s. f4 demonstrates the all-features
speed ceiling at 6.910 s but changes model numerics and must remain report-only.

### Receipts

- Current matrix: `/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/`
  `h3_vllm_match_varlen_99cd_20260822/`; launcher `h3_varlen_final.sbatch`; terminal logs
  `logs/{final_2707_2.out,final_2707_3.out,final_2708_0.out,final_2708_1.out,retry_2701_0.out,`
  `rack2_2699_1.out}`; raw per-leg `summary.json` and `schedule_contract.json` under `runs/`.
- Fixed attribution: `vsa_gate/h3_vllm_match_20260822/logs/matrix_2675_{0,1}.out` and its `runs/`
  tree. Fixed-vs-packed comparison: current workspace `logs/parity_2732.out` plus
  `runs/dense/{1x,sp4}/{d0_eager,d4_regional_vaec_vaepar}/parity_vs_fixed.json`.
- FA4 API A/B: `vsa_gate/fa4_h3_api_ab_20260822/{logs/fa4_api_2695.out,results/}`. Runtime shape
  capture: `vsa_gate/h3_vllm_match_native_precision_172d_20260822/cache/2692_0/torchinductor/`
  `gy/cgy6er2arydb5n3d777vp7lk4cgdwycrdk4mecibwxfozf5iajlp.py`.
- vllm-omni: `vllm_omni_bench/{results_rb_1gpu_off.jsonl,results_rb_4gpu_on.jsonl,`
  `logs/rebench_2657.out,logs/server_rb_1gpu_off.log,logs/server_rb_4gpu_on.log}`.
- PR-head GPU tests: `vsa_gate/prtest_h3_fa4_varlen_2704be79/test_2755.out`.
- Corrected current FastH3 f3/f4 pairing: current workspace
  `logs/fast_current_parity_2780.out` and per-leg `parity_current_vs_eager.json` files.

## 7. Matched duration grids: T2VA and Ref2VA (2026-08-22)

This refresh extends the same prompt/seed/geometry contract to valid H3 frame counts 124, 243,
and 345: nominal 5/10/15-second buckets with exact encoded durations 5.167/10.125/14.375 seconds
at 24 fps. Every successful row is the median of three timed requests after a shape-specific
warmup; ranges are the minimum and maximum timed requests. Server boot, model load, and warmup
compile are excluded.

`num_inference_steps` means scheduler points in both implementations, not model calls. Base H3
uses 50 points and makes exactly **49 DiT forwards**; FastH3/F4 and the explicitly labeled
FastVideo Ref2VA proxy use five points and make exactly **four DiT forwards**. Timed vllm-omni
logs finish at `49/49`; FastVideo activation traces record transformer step indices
`[0,1,2,3]` for every request.

### T2VA: vllm-omni base H3 versus FastH3 F4 all-features

All T2VA rows use synth64 prompt 0 (SHA256 `04116fe2...`), seed 1000, 768x1344, guidance 1.0,
and the same released checkpoints named in section 6. vllm-omni uses dense CuTe FA4 and its
regional DiT compile path. FastH3 F4 uses eager sparse DiT, VSA@0.9 tile-64, compiled video
decoder, temporal-parallel VAE at SP-4, and H3 fusions. F4 is the all-compatible-features speed
ceiling and remains **report-only/non-parity**; its sparse DiT is not compiled.

| system / profile | GPUs | target / frames | points / fwds | attention route | e2e median (range), s | denoise median (range), s | peak MiB | job |
|---|---|---:|---:|---|---:|---:|---:|---|
| vllm-omni base | 1x | 5s / 124 | 50 / 49 | dense CuTe FA4 | 136.003 (135.944-136.160) | 127.744 (127.689-127.868) | 131698 | 2657 |
| FastH3 F4 | 1x | 5s / 124 | 5 / 4 | VSA64 sm100a CUDA | 16.694 (16.053-16.716) | 8.664 (8.608-8.785) | 78424 | 2707_2 |
| vllm-omni base | SP-4 | 5s / 124 | 50 / 49 | dense CuTe FA4 | 40.804 (40.615-40.825) | 37.222 (37.220-37.224) | 95564 | 2657 |
| FastH3 F4 | SP-4 | 5s / 124 | 5 / 4 | VSA64 sm100a CUDA | 6.879 (6.763-7.088) | 2.923 (2.917-3.039) | 79397 | 2707_3 |
| vllm-omni base | 1x | 10s / 243 | 50 / 49 | dense CuTe FA4 | 388.952 (388.799-389.511) | 371.753 (371.724-371.762) | 139572 | 2890_0 |
| FastH3 F4 | 1x | 10s / 243 | 5 / 4 | VSA64 sm100a CUDA | 31.116 (30.985-31.305) | 19.328 (19.198-19.340) | 86968 | 2885_0 |
| vllm-omni base | SP-4 | 10s / 243 | 50 / 49 | dense CuTe FA4 | 111.778 (111.306-111.989) | 104.433 (104.432-104.534) | 101036 | 2890_1 |
| FastH3 F4 | SP-4 | 10s / 243 | 5 / 4 | VSA64 sm100a CUDA | 12.045 (10.804-15.219) | 5.786 (5.782-5.883) | 79438 | 2885_1 |
| vllm-omni base | 1x | 15s / 345 | 50 / 49 | runtime failure | **N/A** | **N/A** | **N/A** | 2890_0, 2961, 2962 |
| FastH3 F4 | 1x | 15s / 345 | 5 / 4 | VSA64 corrected sm100a CUDA | 47.212 (46.729-47.978) | 29.699 (29.688-29.716) | 96037 | 2908_0 |
| vllm-omni base | SP-4 | 15s / 345 | 50 / 49 | dense CuTe FA4 | 200.308 (200.302-200.557) | 190.059 (189.862-190.192) | 113266 | 2890_1 |
| FastH3 F4 | SP-4 | 15s / 345 | 5 / 4 | VSA64 corrected sm100a CUDA | 15.468 (15.379-15.557) | 9.107 (9.098-9.119) | 79615 | 2908_1 |

vllm-omni 345f/1x is deliberately N/A, not a one-sample timing. Job 2890_0 completed one
intentionally unsaved warmup (703.305 s e2e, 678.364 s denoise, 19.634 s decode, 153654 MiB,
49/49 forwards), then all three timed requests failed with `v must be finite`. Fresh-server jobs
2961 and 2962 retained their first request but both failed with CUDA illegal-memory-access/CUBLAS
execution errors before producing media. The warmup remains diagnostic-only and cannot supply a
timing row or comparison video.

The original job-2885 345f F4 cells ran at integration SHA `99cd355a` and correctly selected the
Triton-64 fallback because the logical VSA grid had 1,743 tiles: 54.809
(54.493-54.836) / 39.998 (39.960-40.051) s e2e/denoise at 1x and 18.236
(17.880-18.517) / 11.428 (11.427-11.429) s at SP-4. They remain preserved as legacy rows; they
are not relabeled. The primary table's corrected 345f rows use private composition
`8f8529d9789b54e78b519e5c60737d29db0ceb72` (99cd plus source commit `7635a5295`) and extension
SHA256 `3f960423...`: one zero-valid transport partner makes 1,744 internal tiles without changing
the 1,743-tile logical score/mask/output geometry. GB200 gate job 2903 passed 17 tests, including
the Triton-64 oracle comparison (maximum absolute difference 0.007812).

Timing definitions differ only at the serving boundary: vllm-omni e2e is HTTP client wall time
for `/v1/videos/sync`, while FastVideo e2e surrounds `VideoGenerator.generate` including save;
denoise is `MiniMaxH3Pipeline.diffuse` versus `FASTVIDEO_STAGE_LOGGING`'s `denoising_stage`.

### T2VA comparison videos: vllm-omni base versus FastH3 F3 strict

The visual comparisons deliberately use **F3 strict**, not the F4 timing profile: VSA64,
compiled decoder, temporal-parallel VAE at SP-4, and fusions off. Each montage is 2688x768 at
24 fps with the exact source frame count; it copies the left vllm-omni AAC packets and omits the
right-side audio. Full input/output hashes, stream probes, and FFmpeg commands sit beside each
MP4 in the comparison root below.

| target | 1x MP4 (SHA256 prefix) | SP-4 MP4 (SHA256 prefix) |
|---:|---|---|
| 5s / 124f | `t2va_5s_1x_vllm_base_vs_fasth3_f3.mp4` (`d54d7de8`) | `t2va_5s_sp4_vllm_base_vs_fasth3_f3.mp4` (`3fd6e0f4`) |
| 10s / 243f | `t2va_10s_1x_vllm_base_vs_fasth3_f3.mp4` (`8516e8a0`) | `t2va_10s_sp4_vllm_base_vs_fasth3_f3.mp4` (`0cee1523`) |
| 15s / 345f | **N/A: no valid vllm-omni source MP4** | `t2va_15s_sp4_vllm_base_vs_fasth3_f3.mp4` (`aec88230`) |

The 345f F3 comparison predates the odd-tile composition and truthfully records its
Triton-64 fallback. No Ref2VA F3 montage exists: the Preview export does not contain a distilled
`transformer_ref`, so such a file would compare vllm-omni base against FastVideo base weights
while falsely labeling the right side FastH3.

### Ref2VA: genuine base grid and the FastVideo four-forward latency proxy

The Ref2VA contract uses the same prompt and seed, plus the first N frames/audio of
`vsa_gate/ref2va_grid/C/reference_15s.mp4` (SHA256 `5b74f889...`) for each target. Every saved
output passed the exact H.264 1344x768@24-fps frame contract and carries stereo 32-kHz AAC;
three repeats within each successful cell are byte-identical.

**Identity guardrail: genuine FastH3 Preview Ref2VA is N/A.** Preview manifest
`modular_model_index.json` (SHA256 `63a5c56b...`) has no distilled `transformer_ref` and resolves
that component to official `MiniMaxAI/MiniMax-H3/transformer_ref`. The first table is genuine
vllm-omni **base H3 Ref2VA**. The second table is a FastVideo **official-base transformer_ref,
four-forward F4 latency proxy only**; it is neither FastH3 nor quality-valid.

vllm-omni base uses dense CuTe FA4 plus lazy regional `torch.compile(dynamic=True)` on all 52 DiT
blocks. SP-4 additionally uses USP-4, text-encoder TP-4, and spatial-tile VAE patch parallelism 4.

| implementation | GPUs | target / frames | points / fwds | e2e median (range), s | denoise median (range), s | peak MiB | route / status |
|---|---|---:|---:|---:|---:|---:|---|
| vllm-omni base Ref2VA | 1x | 5s / 124 | 50 / 49 | 463.700 (463.462-463.907) | 441.801 (441.525-442.237) | 142652 | dense CuTe FA4 + regional compile |
| vllm-omni base Ref2VA | 1x | 10s / 243 | N/A | **N/A** | **N/A** | **N/A** | fresh one-forward warmup: CUDA illegal-address |
| vllm-omni base Ref2VA | 1x | 15s / 345 | N/A | **N/A** | **N/A** | **N/A** | fresh one-forward warmup: CUDA illegal-address |
| vllm-omni base Ref2VA | SP-4 | 5s / 124 | 50 / 49 | 135.535 (135.448-136.668) | 126.628 (126.621-127.808) | 97686 | dense CuTe FA4 + regional compile |
| vllm-omni base Ref2VA | SP-4 | 10s / 243 | 50 / 49 | 416.717 (415.745-417.267) | 399.994 (399.991-400.441) | 109316 | dense CuTe FA4 + regional compile |
| vllm-omni base Ref2VA | SP-4 | 15s / 345 | 50 / 49 | 758.643 (758.533-760.182) | 735.051 (734.863-735.209) | 125762 | dense CuTe FA4 + regional compile |

The 1x 243f/345f rows are N/A, not extrapolations. Fresh default-route job 2906 reproduced the
failures. Together, jobs 2906/2910/2912 tested FA4 dynamic/static/eager, CuDNN dynamic, and SDPA
eager at 243f; every route failed during the two-point/one-forward shape warmup while sampled
peaks stayed below the 189471-MiB GB200 capacity. These are runtime/kernel failures, not reported
OOMs and not valid 49-forward timing runs. Successful base rows come from job 2892.

The FastVideo proxy uses eager sparse DiT, VSA@0.9, compiled VAE, H3 fusions, and parallel
reference encode/decode at SP-4. It is report-only/non-parity.

| implementation | GPUs | target / frames | points / fwds | e2e median (range), s | denoise median (range), s | peak MiB | sparse route / job |
|---|---|---:|---:|---:|---:|---:|---|
| FV **base-weight proxy, not FastH3** | 1x | 5s / 124 | 5 / 4 | 65.135 (64.435-65.247) | 45.414 (45.332-45.425) | 84929 | sm100a CUDA-64 / 2877 |
| FV **base-weight proxy, not FastH3** | 1x | 10s / 243 | 5 / 4 | 164.521 (163.042-172.343) | 130.868 (130.618-130.871) | 97494 | FA4 CuTe-256 / 2894 |
| FV **base-weight proxy, not FastH3** | 1x | 15s / 345 | 5 / 4 | 292.989 (292.382-294.289) | 246.206 (246.120-246.318) | 117289 | sm100a CUDA-64 / 2901 |
| FV **base-weight proxy, not FastH3** | SP-4 | 5s / 124 | 5 / 4 | 21.733 (21.499-22.125) | 12.847 (12.814-12.884) | 82517 | sm100a CUDA-64 / 2877 |
| FV **base-weight proxy, not FastH3** | SP-4 | 10s / 243 | 5 / 4 | 48.282 (48.166-48.804) | 34.886 (34.869-35.010) | 83006 | FA4 CuTe-256 / 2894 |
| FV **base-weight proxy, not FastH3** | SP-4 | 15s / 345 | 5 / 4 | 81.666 (81.395-82.904) | 64.025 (63.997-64.149) | 89117 | FA4 CuTe-256 / 2894 |

At 243f the legacy CUDA-64 even-tile predicate rejects the exact packed Ref2VA geometry, so the
supported primary route is CuTe-256. Both 345f routes completed: 1x selected CUDA-64 at
292.989/246.206 s versus CuTe-256 at 293.343/249.344; SP-4 selected CuTe-256 at
81.666/64.025 versus CUDA-64 at 83.884/66.109. Route timings are not asserted bit-identical.

### Duration-grid receipts

- T2VA: `vsa_gate/h3_duration_grid_20260822/t2va/{RESULTS_T2VA.md,RESULTS_T2VA.json}` (SHA256
  `ab29b5c4...` / `d828f7d6...`). The machine receipt hashes every raw result, log, primary MP4,
  and validated media contract. Primary jobs: 2657, 2707, 2885, 2890, 2903, 2908, 2961, 2962.
- T2VA comparisons: `vsa_gate/h3_duration_grid_20260822/comparisons_vllm_vs_f3/`; each
  `.receipt.json` names and hashes both inputs and the output. The sibling `README.md` records
  the final 15s/1x N/A state.
- Ref2VA: `vsa_gate/ref2va_duration_grid_20260822/{RESULTS.md,RESULTS.json}` (SHA256
  `a9f7220e...` / `55509996...`). This is the sole source for the current Ref2VA duration grid;
  it contains every raw timing, MP4/ffprobe contract, route probe, failure envelope, command, and
  environment receipt. Primary jobs: 2877, 2892, 2894, 2901; failure/probe jobs 2906, 2910, 2912.
- Main T2VA code SHA is `99cd355a`; only corrected FastH3 F4 345f uses private composition
  `8f8529d9`. Ref2VA FastVideo proxy code is `99cd355a`. vllm-omni is
  `73b623f2f7db092053c1c86fe796bed89eb3dc71` plus timing-only profiler patches. Hardware is
  GB200 (189471 MiB/GPU), driver 580.82.07. No benchmark launcher overrides `HOME`; caches are
  explicit and job/task scoped.

## 8. Historical pre-merge stacked-PR integration

Historical only; sections 5-7 supersede this branch as the current preview serving result. The
detail remains here because it contains the node-pinned attribution and the dense 50-step job
2653.

Branch `integration/h3-perf-stack` on `hao-ai-lab/FastVideo`, tip `ad9cd6312` =
`origin/main` 56d4a6074 + the reviewed PR heads merged in order: #1362
(`aa95a4c18` pr1362-fixes) and #1703 (`942f7db3d` pr1703-fixes) — both content
no-ops, main's squash-merges `fca45bc8e`/`e0a3db565` already contain the review
fixes; #1732 (`ac56806af`, slim text-encoder contract + opt-in serialized FP8,
kept OFF); #1735 (`cbab605ef`, opt-in Sol-Engine fusions, includes the upstream
int64 qknorm-RoPE fix `ac98869aa`); #1734 (`dca423fd3`, VAE decode compile +
VAE attention dispatch + NVTX); #1731 (`9713ea127`, FastH3 + VSA-H3 tile-64 +
sm100a route) on top. All six merges were textually clean; the predicted
1734<->1731 denoising-stage and 1732<->1734 conditioning/loader overlaps landed
in disjoint hunks (verified by hand, documented in the merge commits).

Two blocker fixes are carried on the branch:

1. `68e6ffca9` — #1734 review F1: clone the reduce-overhead `_stitch_tiles`
   canvas at the tile-driver returns. Unfixed, the first tiled eager
   `decode()`/`encode()` on CUDA dies with the cudagraph-overwrite
   RuntimeError. Red/green proven on GB200 (job 2629 gate: unfixed tree fails
   with the exact review signature at `minimax_h3_video.py:722`; stack passes),
   CUDA regression test added (tiled decode 2 chunks + encode 2 clips,
   unmocked stitch, bitwise repeat-consistency + parity vs the eager helpers).
2. `ad9cd6312` — new blocker found benching the stack: #1732's
   `@torch.inference_mode()` on the conditioner kills the first encode when
   `text_encoder_cpu_offload=True` (the DEFAULT) because FSDP2's
   `wait_for_unshard` reads `tensor._version` ("Inference tensors do not track
   version counter", job 2594, all legs). This is review finding 2's
   unexercised-default-path gap realized. Fixed with `@torch.no_grad()` (same
   activation memory; prompt_embeds are ordinary tensors again, retiring
   review finding 7).

#1735's int64 blocker needed no carry (upstream `ac98869aa` at qknorm_rope.py:43).
#1732's other blockers stay unreachable here: cutlass m%4 is sm12x-only and
text-encoder FP8 is OFF in every leg.

Methodology = section 1 (synth64 prompt 0, seed 1000, 768x1344x124, 1 untimed
warmup + 3 timed repeats, `FASTVIDEO_STAGE_LOGGING=1`; stock executor). Jobs
2595/2629/2630/2641, node-pinned A/B for decode attribution; artifacts under
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/stack_bench/`. Legs b/c/d = FastH3
export, `num_inference_steps=5` (4 forwards), VSA@0.9 tile-64; leg a = base
dense FA4 50-step (FA4 CuTe overlay); leg d = leg b +
`FASTVIDEO_MINIMAX_H3_FUSIONS=1` (#1735 opt-in — DISCLOSED NON-PARITY
numerics, speed reference only; never enable for parity/SSIM work).

### End-to-end / denoise (seconds, mean of 3, range in parens)

| leg | 1x e2e | 1x denoise | SP-4 e2e | SP-4 denoise | sec.1 baseline (e2e 1x / SP-4) |
|---|---:|---:|---:|---:|---|
| a. dense FA4 50-step | 186.6 (186.5-186.7) | 176.39 | 66.0 (65.7-66.2) | 52.63 | 185.2-188.2 / 62.8-63.7 |
| b. FastH3 VSA-64 triton | 22.9 (21.7-25.3) | 12.48 | 17.2 (16.4-18.7) | 4.14 | 22.3-22.8 / 13.6-15.0 |
| c. FastH3 VSA-64 sm100a | 19.4 (19.0-19.7) | 10.80 | 15.9 (15.3-16.3) | 3.72 | 20.7 / 15.1 |
| d. b + H3 fusions (non-parity) | 20.2 (19.6-20.6) | 10.43 | 15.4 (14.0-16.2) | 3.63 | (new) |

### Per-stage deltas vs the section-1 baselines

| stage | stack 1x | stack SP-4 | baseline | delta |
|---|---:|---:|---|---|
| text encoding (#1732 slim + merged-#1711) | 0.53-0.55 | 0.44-0.59 | ~2 | **-1.5 s (-75%)** |
| denoising, dense FA4 (49 fwds) | 176.39 | 52.63 | 175-179 / 52.9-54.2 | parity |
| denoising, student (same kernel) | 12.48 triton / 10.80 sm100a | 4.14 / 3.72 | 12.4 / 10.7 (1x), 4.3 / 3.9 (SP-4) | parity to -5% |
| denoising, +fusions vs leg b | 10.43 | 3.63 | — | **-16% (1x) / -12% (SP-4)** |
| video VAE decode (#1703 streaming + #1734 tile compile) | 7.05-8.81 | 9.9-11.7 | 6.4-7.5 (stock SP-4 7.26) | **REGRESSION, see below** |
| audio decode + post + save | ~0.8-1.1 | ~0.9-1.2 | ~1 | parity |

**VAE decode: regression at defaults, win behind the opt-in flag (attributed).**
Same-node/same-day A/B (jobs 2630/2641, hpc-rack-3-8, identical leg-b config):

| decode path | SP-4 vdec | SP-4 e2e | 1x vdec | 1x e2e |
|---|---:|---:|---:|---:|
| pre-#1734 tree (rebased #1731 on merged main) | **8.18** (8.13-8.22) | 14.0 | — | — |
| stack, defaults (unconditional tile compile only) | 11.02 (9.90-12.38) | 17.2 | 8.81 (7.51-11.12) | 22.9 |
| stack + `enable_torch_compile_vae` (#1734's opt-in decoder compile) | **6.05** (5.86-6.34) | **12.4** (12.0-13.2) | **4.86** (4.55-5.08) | **18.7** (18.5-18.9) |

At stack DEFAULTS the only active #1734 decode-path change is the
unconditional `reduce-overhead`/`dynamic=False` compile on `_stitch_tiles` /
`_project_decoder_tile` (VAE attention resolves to SDPA in these legs — no
flash_attn on the student PYTHONPATH; the F1 clone costs one ~0.1 GB D2D per
chunk, microseconds) — and it REGRESSES decode +2.8 s SP-4 / +~1 s 1x with
extra jitter, exactly review finding F2's concern (unconditional compile
violating the opt-in contract). With the PR's intended opt-in decoder compile
the win is real and large: decode -2.1 s vs pre-#1734 same-day (8.18 -> 6.05
SP-4; 1x 4.86 vs the 6.4-7.5 baseline band), giving the best SP-4 student e2e
measured to date (12.4 s vs the 13.6-15.0 section-1 band; denoise unchanged).
Upstream ask: gate the tile compile behind `enable_torch_compile_vae` (F2) so
defaults do not regress, and ship the flag-on configuration in the H3 example.

### Sanity / gates

- CPU: union of the six PRs' test files + VSA-H3 metadata/route suites on the
  stack = **135 passed, 13 skipped (CUDA/weights-gated), 0 failed** (login
  node, empty-fastvideo_kernel shim). `fastvideo/tests/stages/test_text_encoding.py`
  carries 7 pre-existing failures that reproduce byte-identically on plain
  `origin/main` 56d4a6074 (A/B'd) — upstream, not stack-caused.
- GPU gates (job 2629): unfixed-stitch red-proof RED_CONFIRMED; stack suites
  (vae compile incl. the new CUDA regression test, streaming + pinned, all
  fusion suites, VSA-H3 metadata + sm100a route) **71 passed, 1 skipped**
  (FA4-version-string gate, environmental); sm100a kernel sanity ALL OK.
- Determinism: every leg x shape (incl. both A/B probes) produced 3
  byte-identical same-seed videos — SP-4: triton 518c20db…, sm100a 7b5b8dde…,
  fusions fc1a2447…, dense 90e27ab8…; 1x: triton fcd604eb…, sm100a 8ad1c6ce…,
  fusions 1c8de316…, dense 3ab1314d…; vae-compile probes 7c476f65…/aa26f7cb…;
  pre-#1734 probe 2b5e4c47…. Leg d is deterministic but non-parity by design —
  frame-60 inspection shows a clean, prompt-faithful render (composition
  drift only).
- Fusion engagement proof: "MiniMax H3 inference fusions enabled:
  modulate,qknorm_rope,swiglu" in both leg-d worker logs.

### Verdict

The stack is runnable and healthy end-to-end with the two carried fixes:
text encode -75%, denoise at parity everywhere same-kernel (dense 1x e2e
186.6 sits inside the 185.2-188.2 baseline band) or -12/-16% with the
disclosed non-parity fusions, and the student 1x e2e improves (sm100a 19.4
vs 20.7). The one default-configuration regression is #1734's unconditional
tile compile in VAE decode (+2.8 s SP-4 in the node-pinned A/B, +~1 s 1x
modal), which puts dense SP-4 e2e at 66.0 vs 62.8-63.7 baseline; flipping
#1734's own opt-in `enable_torch_compile_vae` more than recovers it
(SP-4 student e2e 12.4 s — best measured; decode 6.05/4.86 s). Upstream
asks: land the two carried fixes into #1734/#1732, and gate the tile compile
behind the VAE compile config (review F2) so defaults do not regress.
Serving guidance meanwhile: run the stack with `enable_torch_compile_vae`
on (parity-safe decoder compile) and, where output parity is not required,
`FASTVIDEO_MINIMAX_H3_FUSIONS=1`.

### Dense base with stack opt-ins (job 2653, node-matched)

The leg-a numbers above ran the DENSE base model (50-step) at stack defaults.
Same methodology and same node (hpc-rack-3-5) with the stack's opt-ins ON —
#1732's slim conditioner is already default-on in every leg; a2 adds #1734's
`enable_torch_compile_vae`; a3 adds #1735's `FASTVIDEO_MINIMAX_H3_FUSIONS=1`
(DISCLOSED NON-PARITY numerics — speed reference only):

| leg | 1x e2e | 1x denoise | SP-4 e2e | SP-4 denoise |
|---|---:|---:|---:|---:|
| a2. a + `enable_torch_compile_vae` (parity-safe optimum) | **181.1** (181.0-181.3) | 174.86 | **61.3** (60.6-62.3) | 52.57 |
| a3. a2 + H3 fusions (non-parity) | **157.2** (156.4-157.8) | 151.10 | **55.6** (55.2-55.8) | 47.02 |

Baselines: stack defaults (leg a) 186.6 / 66.0; pre-stack 185.2-188.2 / 62.8-63.7.

- a2 beats the PRE-STACK dense baseline at both shapes (1x -2.2% vs the 185.2
  best; SP-4 -2.3% vs 62.8) and the stack defaults by -5.5 s / -4.7 s: the
  opt-in decoder compile recovers the defaults decode regression with room to
  spare. Video VAE decode 4.98 s 1x (4.88-5.04) / 7.05 SP-4 (6.78-7.48) vs
  8.3-8.9 / 10.8-12.3 at defaults and the 6.4-7.5 pre-stack band; text encode
  0.34-0.42 s in all four legs (slim conditioner).
- Fusions on the 49-forward dense path: denoise **-13.6% at 1x** (174.86 ->
  151.10, 3.57 -> 3.08 s/fwd) and **-10.6% at SP-4** (52.57 -> 47.02) vs a2 —
  the 4-forward student saw -16%/-12% (leg d). Decode/text unchanged
  (4.90 / 6.86); engagement line present in both worker logs.
- Determinism: every leg produced 3 byte-identical same-seed videos — a2 1x
  ff70127e…, SP-4 9c453149… (parity-safe requirement PASS); a3 1x cf096256…,
  SP-4 8bd6b440… (run-to-run deterministic; numerics differ from a2 by design
  — frame-60 inspection clean and prompt-faithful, minor detail drift only).
  Videos: `stack_bench/videos/dense50_vaec{,_fus}_{1x,sp4}/`, raw runs
  `stack_bench/run_{1x,sp4}_denseopt/`.
- Net: the dense/teacher serving config on the stack is a2 — the fastest
  parity-safe dense SP-4 measured (61.3 s); where parity is not required, a3
  reaches 157.2 / 55.6 (**-15.8% vs stack defaults at both shapes**).

## Route guidance (from the measurements)

1. FastH3 Preview uses exactly four forwards. F3 (decoder compile + parallel VAE, fusions off) is
   the strict visual-comparison/serving profile; F4 is the all-features speed ceiling and remains
   report-only/non-parity. At 15s, use the corrected odd-tile sm100a-64 route for F4 rather than
   silently inheriting the older Triton-64 fallback.
2. Base H3 at 5s: packed d4 matches vllm-omni at both exact 49-forward shapes (132.468 s 1x,
   40.587 s SP-4), but it is a report-only route. Keep fixed FA4 for parity-sensitive work until
   the deterministic cross-route drift receives an explicit quality-acceptance decision. The
   duration refresh does not claim a FastVideo-base match at 10s or 15s.
3. Genuine FastH3 Preview Ref2VA is N/A because the export has no distilled `transformer_ref`.
   Never label the four-forward FastVideo base-weight latency proxy as FastH3 or quality-valid.
4. Ref2VA/long-sequence (>=100k): never Triton-256; select a supported CuTe-256 or sm100a-64
   route by exact packed geometry. P2 ref-sparsification remains the training/quality direction
   (keep 0.10 trained / 0.25 zero-finetune), not part of the section-7 serving grid.
5. Do not infer compile coverage from the headline alone: base d4 and vllm-omni regionally compile
   the DiT; FastH3 F3/F4 keep the sparse DiT eager and compile only the decoder. The Ref2VA proxy
   likewise uses eager sparse DiT.
