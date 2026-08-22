# MiniMax-H3 / FastH3 performance numbers (measured, GB200)

This is a dated results ledger. Sections 1-4 and historical section 8 preserve the measurements
through 2026-08-21 at their row-local SHAs; section 5 is the job-2666 merged-preview baseline,
section 6 is the current 5-second matched-serving and multi-node-launcher refresh, and section 7
is the 5/10/15-second T2VA and Ref2VA duration grid with row-local input contracts. Unless a row
says otherwise, FastVideo runs use the lustre venv (torch 2.12.0+cu130), driver 580.82.07,
synth64 prompt 0, seed
1000, warmup excluded, and 2-3 timed repeats. "FastH3" = the 4-forward DMD2 student (v8 data-free
step-1400 preview export) with VSA @0.9; it does not mean the live v10 training run. Raw artifacts:
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/`
`{sm100a_e2e,sp8,ref2va_grid,ref2va_duration_grid_20260822,vsa64bench,`
`preview_merged_20260822,h3_vllm_match_20260822,h3_vllm_match_varlen_99cd_20260822,`
`h3_duration_grid_20260822,public_sp8_54e5_20260822}` and `vllm_omni_bench/`; master index
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
  (`FASTVIDEO_EXTERNAL_LAUNCHER=1`, torchrun). The historical row used the unmerged worktree
  commits `c65733463`+`ac24e7e78`; the clean public extraction and current parity envelope are
  PR #1746 and job 3026 in section 6.

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

| PR and merge SHA | contribution | disposition in job 2666 |
|---|---|---|
| #1362 `fca45bc8e` | uint8 conversion before post-decode D2H | active in every leg |
| #1703 `e0a3db565` | streaming, lower-peak H3 VAE | active in every leg |
| #1711 `0462e1b0e` | stop Qwen3-VL construction after the last consumed layer | active in every leg |
| #1719 `907f2100e` | Blackwell sm100a block-sparse forward | active through the asserted kernel prefix |
| #1730 `56d4a6074` | corrected Triton sparse backward scaling | present, but inference-inert |
| #1731 `6d6a10be7` | preview example plus tile-64 sm100a H3 route | active in every leg |
| #1732 `bcffa4026` | slim conditioner plus optional serialized FP8 | slim path active; FP8 deliberately off |
| #1734 `2f3d40740` | VAE dispatch, compile, and NVTX | defaults in all legs; decoder compile on in p1/p2 |
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

| shape and leg | text | video decode | audio decode | post-decode | save |
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

This section supersedes the serving headlines in sections 1 and 5. Most attribution rows below
were measured on the clean `integration/h3-vllm-parity-20260822` tree at exact SHA
`99cd355a2452ce040591fe54ef340d192e26fe48`; it remains the historical timing/attribution
authority, while the exact public-head compositions in the final subsections are the current
public-serving acceptance authority. The `99cd355a` merged-main base is
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
| f4 | eager VSA | compiled | temporal parallel | on | report-only (non-parity) |

Temporal VAE parallelism is a one-rank no-op in the 1x d3/d4/d5/d6 and f2/f3/f4 rows. The d1-d6
rows were run in the fixed-length attribution matrix; only d0/d4 were rerun on the final packed
route.

### Base H3 versus vllm-omni: exact 49-forward match

Seconds are arithmetic means of three timed requests; the parenthesized interval is the request
range. The current FastVideo d0 receipts are jobs 2708_0/2708_1, d4 is jobs 2701_0/2699_1, and
the matched vllm-omni profiler rebench is job 2657.

| implementation and leg | shape | e2e s | denoise s | seconds per forward | video decode s | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| vllm-omni, regional + serial VAE | 1x | 136.036 (135.944-136.160) | 127.767 | 2.607 | 6.320 | 128.61 |
| FastVideo d0, packed eager | 1x | 161.064 (160.887-161.391) | 152.105 | 3.104 | 7.598 | 77.58 |
| **FastVideo d4, packed regional + VAE compile** | **1x** | **132.468 (132.213-132.686)** | **125.750** | **2.566** | **5.429** | **72.98** |
| vllm-omni, regional + USP, TP, tile VAE | SP-4 | 40.748 (40.615-40.825) | 37.222 | 0.760 | 1.733 | 93.32 |
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
three timed repeats.

| leg | 1x e2e, s | 1x denoise, s | 1x video decode, s | SP-4 e2e, s | SP-4 denoise, s | SP-4 video decode, s | parity evidence against d0 |
|---|---:|---:|---:|---:|---:|---:|---|
| d0 | 184.292 | 174.863 | 8.113 | 65.282 | 53.426 | 9.749 | reference |
| d1 | 181.723 | 174.969 | 5.579 | 60.587 | 53.262 | 5.518 | strict PASS: 1x mean 0.988096, minimum 0.977004; SP-4 mean 0.988234, minimum 0.980846 |
| d2 | 157.036 | 147.078 | 8.641 | 54.130 | 44.800 | 7.792 | report-only (regional compile) |
| d3 | 183.063 | 174.734 | 7.033 | 57.612 | 53.322 | 2.992 | exact: MS-SSIM 1.0 |
| d4 | 153.409 | 147.078 | 5.121 | 47.944 | 44.734 | 2.038 | report-only (regional compile) |
| d5 | 155.993 | 150.061 | 4.715 | 48.654 | 45.408 | 2.045 | report-only (regional + fusions) |
| d6 | 157.532 | 150.960 | 5.328 | 51.806 | 48.556 | 2.087 | report-only (eager fusions) |

Across the fixed-run and final packed-run trees, d0 1x fell by 12.60% end-to-end / 13.01% denoise
and d4 1x by 13.65% / 14.50%. The observed SP-4 changes were 9.50% / 18.26% for d0 and 15.34% /
17.00% for d4, but those full-pipeline deltas are partially confounded: job 2675 used
FSDP/sharded SP-4 weights whereas the final matrix used replicated DiT weights, and the final
tree also contains the scoped-regional-compile follow-up. Job 2695 below is the isolated evidence
for the FA4 invocation itself.

The direct FA4 API A/B in job 2695 isolates the kernel call at the exact `S=38224`, bf16, D=128
shape:

| heads | fixed-length | packed-varlen | latency change | maximum abs error | mean abs error |
|---:|---:|---:|---:|---:|---:|
| 56 | 36.873 ms | 28.008 ms | -24.0% (1.317x) | 2.4414e-4 | 1.7990e-6 |
| 14 | 9.236 ms | 6.386 ms | -30.9% (1.446x) | 2.4414e-4 | 1.7991e-6 |

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

| leg | shape | e2e s | denoise s | s/fwd | video decode s | peak GiB | parity mean | parity minimum | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| f0 eager | 1x | 20.754 (20.111-21.705) | 10.716 | 2.679 | 7.530 | 81.20 | — | — | reference |
| f1 + VAE compile | 1x | 18.662 (17.723-19.713) | 10.718 | 2.680 | 5.788 | 76.59 | 0.985239 | 0.975648 | PASS |
| f2 + parallel VAE (1x no-op) | 1x | 19.890 (19.756-20.057) | 10.706 | 2.676 | 7.519 | 81.20 | 1.0 | 1.0 | exact |
| f3 + VAE compile + parallel VAE | 1x | 17.779 (17.171-18.747) | 10.695 | 2.674 | 5.391 | 76.59 | 0.985227 | 0.976443 | PASS |
| f4 + H3 fusions | 1x | 16.488 (16.053-16.716) | 8.686 | 2.171 | 6.110 | 76.59 | 0.518061 | 0.391710 | report-only (non-parity) |
| f0 eager | SP-4 | 19.618 (18.523-21.558) | 3.473 | 0.868 | 12.082 | 81.20 | — | — | reference |
| f1 + VAE compile | SP-4 | 13.696 (12.705-14.375) | 3.462 | 0.866 | 7.936 | 76.59 | 0.985101 | 0.976192 | PASS |
| f2 + parallel VAE | SP-4 | 8.371 (7.984-8.966) | 3.476 | 0.869 | 2.780 | 82.05 | 1.0 | 1.0 | exact |
| **f3 + VAE compile + parallel VAE** | **SP-4** | **7.318 (7.207-7.396)** | **3.463** | **0.866** | **1.997** | **77.54** | **0.985105** | **0.975796** | **PASS** |
| f4 + H3 fusions | SP-4 | **6.910 (6.763-7.088)** | **2.959** | **0.740** | 2.077 | 77.54 | 0.500509 | 0.358064 | report-only (non-parity) |

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

### Exact public-head acceptance: current serving authority

Two clean internal compositions verify the public inference-only extraction without relying on
the older all-in-one `99cd355a` tree. Base H3 used
`e0bb6a5374dd944907cb48f8af444fe446e4e1a9`: public main `d3cff517c` (which includes merged
#1741) plus the reviewed contents of #1742-#1745. Its combined GB200 gate, job 2999, passed 119
focused tests with one skip and then passed the world-4 parallel-VAE GPU test on every rank (five
tests per rank). Preview used `bbc8d354892eca37a833a60821035b8e9c0ea26a`: the same public main
plus #1744 (which carries #1743) and #1745. Relative to the #1742-#1745 serving extraction, Preview
omits only #1742 because packed dense FA4 is irrelevant to its VSA attention route; it also
deliberately excludes the independently reviewed #1739 and #1740 heads. The non-shallow ancestry
audit is `provenance_audit.json` (SHA256 `08f29dd1651716a253dc2be15942c5ca25cf5481a40300a3b61e09dd92f767f6`).

Both timing jobs used 768x1344x124, prompt 0, seed 1000, one excluded warmup, and three saved timed
requests. Values are medians; parentheses give the timed range. The base d4 route uses packed-
varlen FA4, regional `fullgraph=True` compile on 52 DiT blocks, compiled VAE decode, replicated
DiT, fusions off, and temporal-parallel VAE decode at SP-4. It requested 50 scheduler points and
the scheduler contract expects 49 forwards; jobs 3008/3009 did **not** instrument a runtime
forward trace, so this evidence must not be relabeled as 49 observed calls.

| exact composition and route | GPUs | e2e median (range), s | denoise median (range), s | video decode median (range), s | peak MiB | vllm-omni e2e delta | vllm-omni denoise delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `e0bb6a5`, base d4 | 1x | **134.378 (134.234-134.451)** | **126.504 (126.415-126.660)** | **6.313 (6.047-6.342)** | 74726 | -1.20% | -0.97% |
| `e0bb6a5`, base d4 | SP-4 | **39.946 (39.930-39.960)** | **36.904 (36.855-36.910)** | **1.831 (1.812-1.850)** | 75696 | -2.10% | -0.86% |

Negative deltas mean FastVideo is faster. The base acceptance therefore reproduces the vllm-omni
match on the public extraction. Against the historical `99cd355a` rows, `e0bb6a5` is 1.44% /
0.60% slower in 1x e2e/denoise and 1.58% /
0.61% faster at SP-4. The first attempts, jobs 3006/3007, are excluded: their long Lustre
`TMPDIR` exceeded the Unix-domain socket path limit when Python returned the generated tensor.
Jobs 3008/3009 reran unchanged code and workload with a short `/tmp` directory. The combined
receipt is `public_serving_e0bb_20260822/base_acceptance/RETRY_RECEIPTS.json` (SHA256
`711ea11f6e1a4a62e52f2ab95bcaa7f16e3e138571c51388a4d0d1bc0e8866f5`); individual 1x/SP-4
receipt SHA256s are `e1c79d6c3e680641b70fa14f0b92d0b493669521d149f90288c770fb35e6577f` and
`f13224eecfcc455272d2a38f9cf7ebb304c0d216dba4e3cc65ad9eebd3f3b6d0`. The job-2999 gate log
SHA256 is `2b9fc68491eca165f936f83a1ee1692e4c986cba840ccd67b3242cd8215efffa`.

Preview f3/f4 use eager VSA@0.9 tile-64 sm100a DiT, a compiled decoder, and temporal-parallel VAE
at SP-4; f4 additionally enables H3 fusions. All four receipts record exactly four observed DiT
forwards per request, valid 124-frame media, and byte-identical repeats within the row. Parity is
mean/min MS-SSIM against the same-composition f0 eager output; f3 has a 0.95 minimum floor, while
f4 remains an ungated non-parity speed ceiling.

| exact composition and leg | GPUs | e2e median (range), s | denoise median (range), s | parity mean | parity minimum | status | receipt SHA256 |
|---|---|---:|---:|---:|---:|---|---|
| `bbc8d35`, f3 strict | 1x | **17.947 (17.887-18.407)** | **10.696 (10.696-10.715)** | 0.985266 | 0.976376 | PASS | `ad6ca5dab108...` |
| `bbc8d35`, f4 all features | 1x | **16.228 (16.202-16.385)** | **8.683 (8.663-8.703)** | 0.518112 | 0.391864 | report-only | `0c0bf53e0aad...` |
| `bbc8d35`, f3 strict | SP-4 | **6.562 (6.560-7.569)** | **3.493 (3.416-3.518)** | 0.985102 | 0.975800 | PASS | `d799c6aa774e...` |
| `bbc8d35`, f4 all features | SP-4 | **6.107 (5.829-6.480)** | **2.974 (2.842-2.991)** | 0.500538 | 0.358231 | report-only | `0a02d025660a...` |

The row receipts live under
`fasth3_preview_acceptance_1744_1745_20260822/runs/fasth3/{1x,sp4}/`
`{f3_sm100a_vaec_vaepar,f4_sm100a_vaec_vaepar_fusions}/run_receipt.json`. Their full SHA256s,
in table order, are `ad6ca5dab1089e7ec51ad462ed0eaeb23361babc0ede9773a611939aacfd42f1`,
`0c0bf53e0aad75c6b9fa847de8f9eba12ad4a1945b1c7c89b23208985b728100`,
`d799c6aa774e8f3b2f4f39f3cca5d16238570a736b480d87620767b9d66add25`, and
`0a02d025660ac5f7ffeb7f76d5f82fb6023fcf1d9f0a56e6cc122b7c793a3400`.
The adjacent `parity_vs_eager.json` files have SHA256s, in the same order,
`3cc8c5d15f79669cdb8390f645fe3954edc180d176631bf4e09abbb44ad4eff8`,
`912b10ebf32c2122dff816e0665a7ae0196d5fed3b66305892488a8954575ec2`,
`6c3040671ea61e49bbc3da4128e31c29eeb4e980d6c7e721a37ba6723b587cde`, and
`6925bc9c82eef15959f9647bac4051dab2df73c487c64afaee53070c65e6b8b6`.

### Public external launcher and genuine two-node SP-8 acceptance

[PR #1746](https://github.com/hao-ai-lab/FastVideo/pull/1746), `[feat] Add
external-launcher executor for multi-node inference`, is open at exact head
`accc5a4208bf6f78a4e70fff2f188fb33c3cd697`; GitHub reports `OPEN`, `MERGEABLE`, and
`REVIEW_REQUIRED`, with merge state `BLOCKED` and check requirements outstanding at this
refresh. It adds the opt-in `env://` executor used by `torchrun`/`srun` across hosts, binds every
process to its local
device, and keeps output ownership on rank 0. The stock multiprocess executor remains the
default. The PR contains inference runtime, tests, and inference documentation only: no training
path or configuration.

Slurm job 3026 (`COMPLETED 0:0`, 40m45s) is the terminal GPU gate for that extraction. It ran on
two real four-GPU GB200 hosts, `hpc-rack-3-3` and `hpc-rack-3-2`, not eight ranks on one host.
The exact clean execution tree was
`20d026fd27afc0b3e54bd7cf4be63dfcdfc9386c`: public main `d3cff517c`, the reviewed serving
contents represented by #1742-#1745, exact PR-head ancestor `accc5a420`, and a logging-only
follow-up. Therefore the multi-node result proves #1746 in the intended public-serving
composition; it is not mislabeled as a benchmark of the isolated PR head.

All cells used Preview F3 strict at 768x1344x124, prompt 0, seed 1000, five scheduler points,
exactly four DiT forwards, one excluded seed-999 warmup, and two saved timed repeats. The DiT
remained eager VSA@0.9 with tile-64 sm100a CUDA; temporal-parallel VAE decode used gather across
all ranks; H3 fusions, PR #1739 packed SP, and PR
#1740 Ulysses all-to-all were explicitly off. `VAE decoder` is the only route difference between
the two SP-8 rows. Medians below are medians of the two samples, and the broad inter-node denoise
ranges are retained rather than smoothed away.

| executor | GPUs | hosts | VAE decoder | e2e median, s | e2e range, s | denoise median, s | denoise range, s | video decode median, s | peak MiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| stock multiprocess | 4 | 1 | compiled | 7.320 | 7.058-7.581 | 3.384 | 3.383-3.384 | 2.329 | 79399 |
| external launcher | 4 | 1 | compiled | 10.400 | 10.146-10.654 | 3.979 | 3.574-4.384 | 4.446 | 79397 |
| external launcher | 8 | 2 | eager | 8.742 | 8.246-9.238 | 2.913 | 2.347-3.480 | 3.784 | 84464 |
| external launcher | 8 | 2 | compiled | **8.199** | 8.063-8.334 | **2.382** | 1.936-2.828 | **3.763** | 79808 |

The SP-4 launcher comparison passed its strict gate against stock multiprocess output: mean and
minimum frame MS-SSIM 0.988245 and 0.976679, mean and maximum uint8 absolute error 1.6176 and
63, and decoded audio exact. The SP-8 compiled-decoder row passed against the SP-8 eager-decoder
golden at mean and minimum frame MS-SSIM 0.985431 and 0.974849, mean and maximum uint8 absolute
error 2.1613 and 71, and exact decoded audio.
Both relationships used floors of 0.98 mean and 0.95 minimum MS-SSIM plus ceilings of 3 mean and
96 maximum uint8 error. Each cell's two timed MP4s are byte-identical within the cell; this proves
repeat determinism, not cross-route bit identity.

Route and topology checks were fail-fast. Every external row emitted rank/device receipts for
exactly ranks 0-3 or 0-7; both SP-8 rows showed local ranks 0-3 on each of the two hosts. Every
rank recorded 12 forward traces (four DiT forwards for the warmup and each of two timed
requests). The logs contain one sm100a selection and three VAE gathers per host, one decoder-
compile marker per host only in compiled cells, and zero Triton fallback markers. All outputs
passed the exact 124-frame H.264 and stereo 32-kHz AAC contract. Before timing, the exact tree
passed 108 focused worker, VSA-route, parallel-VAE, and VAE-compile tests in 17.93 seconds.

Authoritative receipts are under `vsa_gate/public_sp8_54e5_20260822/runs/gate_3026/`:

- result JSON SHA256s for stock SP-4, external SP-4, external SP-8 eager, and external SP-8
  compiled are `4af8a03c9bcc1f026a2dc1c3f3ba9645309216f60d4d54960bffd97e4fa484ee`,
  `a9f3382ce8ff42c191fe126893b1fb5d3ab15d579212ca17570e46a7b8ac817b`,
  `ca19365ff99097c653710a66bf329dd9504fb354ef01de52533d6cc6346cca8f`, and
  `10153cd63eb11eb2a985aae15805492c45e6016b7e6ef47ae898aa2b64941da1`;
- SP-4 launcher and SP-8 decoder-compile parity JSON SHA256s are
  `4c11c49a3d7e3231ebc725b9eeda0439d00de4ef7e4dfd38587f28586332ca22` and
  `88e0eda21e0f53994db6c3ae1488a5090454c9faf213ea2653402a947f527bbc`;
- the two-node runtime-provenance manifest SHA256 is
  `04539bc77bf95e74377995083738b74270ae046f2987a43cf389300f02081070`; the checked 60-file
  model manifest SHA256 is `7950a27656bf4ab5390d638ae55ee13622f78f09bc45065632a4869dfe9bfa62`;
- the terminal gate-log SHA256 is `d7f335e17ffaf808deec2a1df6b18ac49ae7fc2548efe52c623a5d89ad92ee39`,
  and the 108-test log SHA256 is `e740170934aae5f6687dbae5805e91c66c4577c4c9d6e502c6f34cd9c0aa9f8c`.

## 7. Matched duration grids: T2VA and Ref2VA (2026-08-22)

This refresh extends H3 duration benchmarking to valid frame counts 124, 243, and 345: nominal
5/10/15-second buckets with exact encoded durations 5.167/10.125/14.375 seconds at 24 fps.
Prompt/reference identities are contract-local and disclosed before each table; rows from
different input hashes are not treated as matched. Every successful row is the median of three
timed requests after a shape-specific warmup; ranges are the minimum and maximum timed requests.
Server boot, model load, and warmup compile are excluded.

`num_inference_steps` means scheduler points in both implementations, not model calls. Base H3
uses 50 points and makes exactly **49 DiT forwards**; FastH3/F4 and the explicitly labeled
FastVideo Ref2VA proxy use five points and make exactly **four DiT forwards**. Timed vllm-omni
logs finish at `49/49`; FastVideo traces record `[0,1,2,3]` for FastH3/proxy requests and
`[0,...,48]` for the official-base d4 requests that instrumented the 49-forward path.

### T2VA: vllm-omni base H3, FastVideo base d4, and FastH3 F4

All T2VA rows use synth64 prompt 0 (SHA256 `04116fe2...`), seed 1000, 768x1344, guidance 1.0,
and the same released checkpoints named in section 6. vllm-omni uses dense CuTe FA4 and its
regional DiT compile path. FastH3 F4 uses eager sparse DiT, VSA@0.9 tile-64, compiled video
decoder, temporal-parallel VAE at SP-4, and H3 fusions. F4 is the all-compatible-features speed
ceiling and remains **report-only/non-parity**; its sparse DiT is not compiled.

| system and profile | GPUs | target, s | frames | scheduler points | observed forwards | attention route | e2e median (range), s | denoise median (range), s | peak MiB | job |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| vllm-omni base | 1x | 5 | 124 | 50 | 49 | dense CuTe FA4 | 136.003 (135.944-136.160) | 127.744 (127.689-127.868) | 131698 | 2657 |
| FastH3 F4 | 1x | 5 | 124 | 5 | 4 | VSA64 sm100a CUDA | 16.694 (16.053-16.716) | 8.664 (8.608-8.785) | 78424 | 2707_2 |
| vllm-omni base | SP-4 | 5 | 124 | 50 | 49 | dense CuTe FA4 | 40.804 (40.615-40.825) | 37.222 (37.220-37.224) | 95564 | 2657 |
| FastH3 F4 | SP-4 | 5 | 124 | 5 | 4 | VSA64 sm100a CUDA | 6.879 (6.763-7.088) | 2.923 (2.917-3.039) | 79397 | 2707_3 |
| vllm-omni base | 1x | 10 | 243 | 50 | 49 | dense CuTe FA4 | 388.952 (388.799-389.511) | 371.753 (371.724-371.762) | 139572 | 2890_0 |
| FastH3 F4 | 1x | 10 | 243 | 5 | 4 | VSA64 sm100a CUDA | 31.116 (30.985-31.305) | 19.328 (19.198-19.340) | 86968 | 2885_0 |
| vllm-omni base | SP-4 | 10 | 243 | 50 | 49 | dense CuTe FA4 | 111.778 (111.306-111.989) | 104.433 (104.432-104.534) | 101036 | 2890_1 |
| FastH3 F4 | SP-4 | 10 | 243 | 5 | 4 | VSA64 sm100a CUDA | 12.045 (10.804-15.219) | 5.786 (5.782-5.883) | 79438 | 2885_1 |
| vllm-omni base | 1x | 15 | 345 | 50 | 49 | runtime failure | **N/A** | **N/A** | **N/A** | 2890_0, 2961, 2962 |
| FastH3 F4 | 1x | 15 | 345 | 5 | 4 | VSA64 corrected sm100a CUDA | 47.212 (46.729-47.978) | 29.699 (29.688-29.716) | 96037 | 2908_0 |
| vllm-omni base | SP-4 | 15 | 345 | 50 | 49 | dense CuTe FA4 | 200.308 (200.302-200.557) | 190.059 (189.862-190.192) | 113266 | 2890_1 |
| FastH3 F4 | SP-4 | 15 | 345 | 5 | 4 | VSA64 corrected sm100a CUDA | 15.468 (15.379-15.557) | 9.107 (9.098-9.119) | 79615 | 2908_1 |

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

#### FastVideo official-base d4 completion at 10s and 15s

Jobs 2963 and 2964 complete the official-base duration grid on the exact clean
`99cd355a2452ce040591fe54ef340d192e26fe48` serving tree. These are base MiniMax-H3 rows, not
FastH3: every request used the same section-7 prompt/seed/geometry contract and a 50-point sigma
grid with exactly **49 observed DiT forwards**. Values are medians of three saved requests after
one excluded shape-specific warmup; parentheses are the timed-request range. The three-copy
FastVideo prompt JSON has SHA256 `b1f21b1832f38af42fb631a243da8803c00cf48baa4aa42ce3cc448fb9008b0b`.

| target, s | frames | FastVideo d4 1x e2e, s | FastVideo d4 1x denoise, s | vllm-omni 1x e2e, s | vllm-omni 1x denoise, s | FastVideo d4 SP-4 e2e, s | FastVideo d4 SP-4 denoise, s | vllm-omni SP-4 e2e, s | vllm-omni SP-4 denoise, s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 243 | **377.372 (377.217-378.109)** | **366.493 (366.340-366.508)** | 388.952 (388.799-389.511) | 371.753 (371.724-371.762) | **108.714 (108.418-108.811)** | **103.379 (103.275-103.501)** | 111.778 (111.306-111.989) | 104.433 (104.432-104.534) |
| 15 | 345 | **678.690 (678.338-679.394)** | **659.523 (659.493-659.570)** | **N/A** | **N/A** | **193.068 (192.838-193.429)** | **186.429 (186.262-186.472)** | 200.308 (200.302-200.557) | 190.059 (189.862-190.192) |

The 1x and SP-4 results stay in separate columns because SP-4 uses four-rank temporal-parallel
VAE decode while the same flag is a one-rank no-op at 1x. The shared d4 profile is a replicated
DiT with packed-varlen FA4, regional `torch.compile(fullgraph=True)` on all 52 DiT submodules, a
compiled VAE decoder, and H3 fusions off. It remains **report-only/non-parity** because regional
compile and packed FA4 change floating-point reduction order.

| target, s | frames | 1x job | 1x peak MiB | 1x repeat MP4 SHA256 | SP-4 job | SP-4 peak MiB | SP-4 repeat MP4 SHA256 |
|---:|---:|---:|---:|---|---:|---:|---|
| 10 | 243 | 2964_0 | 74767 | `24a57c8181d6...` | 2963_0 | 75736 | `cc38c1e9c1fa...` |
| 15 | 345 | 2964_1 | 77771 | `b4665badd9a7...` | 2963_1 | 75770 | `79ad65b9050b...` |

All twelve timed MP4s decode as H.264 1344x768 at 24 fps with the exact 243/345 frame count and
stereo 32-kHz AAC. The three fixed-seed files in each cell are byte-identical; that proves repeat
determinism, not cross-route parity. Against the matched vllm-omni cells, FastVideo is 2.98% e2e /
1.41% denoise faster at 10s 1x, 2.74% / 1.01% faster at 10s SP-4, and 3.61% / 1.91% faster at
15s SP-4. vllm-omni 15s 1x remains unsupported after the preserved runtime failures above, so the
valid 678.690/659.523-second FastVideo result is one-sided and makes no match or speedup claim.

### T2VA comparison videos: vllm-omni base versus FastH3 F3 strict

The visual comparisons deliberately use **F3 strict**, not the F4 timing profile: VSA64,
compiled decoder, temporal-parallel VAE at SP-4, and fusions off. Each montage is 2688x768 at
24 fps with the exact source frame count; it copies the left vllm-omni AAC packets and omits the
right-side audio. Full input/output hashes, stream probes, and FFmpeg commands sit beside each
MP4 in the comparison root below.

| target, s | frames | 1x MP4 (SHA256 prefix) | SP-4 MP4 (SHA256 prefix) |
|---:|---:|---|---|
| 5 | 124 | `t2va_5s_1x_vllm_base_vs_fasth3_f3.mp4` (`d54d7de8`) | `t2va_5s_sp4_vllm_base_vs_fasth3_f3.mp4` (`3fd6e0f4`) |
| 10 | 243 | `t2va_10s_1x_vllm_base_vs_fasth3_f3.mp4` (`8516e8a0`) | `t2va_10s_sp4_vllm_base_vs_fasth3_f3.mp4` (`0cee1523`) |
| 15 | 345 | **N/A: no valid vllm-omni source MP4** | `t2va_15s_sp4_vllm_base_vs_fasth3_f3.mp4` (`aec88230`) |

The 345f F3 comparison predates the odd-tile composition and truthfully records its
Triton-64 fallback. No Ref2VA F3 montage exists: the Preview export does not contain distilled
`transformer_ref` weights. Aliasing its T2VA student into that role would be an untrained
cross-variant transplant, while using the advertised external component would compare two base
Ref2VA models and falsely label the right side FastH3.

### Ref2VA: genuine base grid and the FastVideo four-forward latency proxy

The original matched vllm-omni/proxy contract uses the same prompt and seed, plus the first N
frames/audio of `vsa_gate/ref2va_grid/C/reference_15s.mp4` (SHA256 `5b74f889...`) for each
target. The genuine FastVideo official-base d4 completion below is deliberately a separate
row-local job-3002 contract and discloses its different prompt/reference hashes; its times must
not be compared directly with the older grid. Every saved output passed the exact H.264
1344x768@24-fps frame contract and carries stereo 32-kHz AAC; three repeats within each
successful cell are byte-identical.

**Identity guardrail: genuine FastH3 Preview Ref2VA is N/A.** Preview manifest
`modular_model_index.json` (SHA256 `63a5c56b...`) advertises
`MiniMaxAI/MiniMax-H3/transformer_ref`, but the release contains no physical `transformer_ref/`
and no distilled weights for that role. The current FastVideo loader selects
`<model_path>/transformer_ref` and does not follow the nested component source metadata, so the
standalone Preview snapshot cannot transparently materialize the advertised base component.
The raw step-1400 export used by the latency-proxy harness instead has `transformer_ref` linked to
the official base component. The vllm-omni and FastVideo d4 tables below are genuine **base H3
Ref2VA** runs. The final table is a FastVideo **official-base transformer_ref, four-forward F4
latency proxy only**; it is neither FastH3 nor quality-valid.

The identity audit also compared the official base components directly: `transformer` and
`transformer_ref` expose the same 638-key architecture but all 14 corresponding shard SHA256s
differ, and sampled tensors differ. FastVideo's Ref2VA preset maps `transformer` to
`transformer_ref` and loads that component as the sole denoiser; it does not combine the two
DiTs. Copying the Preview T2VA student into that slot would therefore be an untrained
cross-variant transplant, while materializing the manifest's external component would simply
run the 50-step base Ref2VA model. Neither is a genuine FastH3 measurement.

Consequently every requested Preview Ref2VA cell is explicitly N/A; none is silently replaced
by the official-base component or the latency proxy:

| requested checkpoint role | GPUs | target, s | frames | result | reason |
|---|---:|---:|---:|---|---|
| FastH3 Preview `transformer_ref` | 1 | 5 | 124 | **N/A** | no distilled Preview Ref2VA weights |
| FastH3 Preview `transformer_ref` | 1 | 10 | 243 | **N/A** | no distilled Preview Ref2VA weights |
| FastH3 Preview `transformer_ref` | 1 | 15 | 345 | **N/A** | no distilled Preview Ref2VA weights |
| FastH3 Preview `transformer_ref` | 4 | 5 | 124 | **N/A** | no distilled Preview Ref2VA weights |
| FastH3 Preview `transformer_ref` | 4 | 10 | 243 | **N/A** | no distilled Preview Ref2VA weights |
| FastH3 Preview `transformer_ref` | 4 | 15 | 345 | **N/A** | no distilled Preview Ref2VA weights |

vllm-omni base uses dense CuTe FA4 plus lazy regional `torch.compile(dynamic=True)` on all 52 DiT
blocks. SP-4 additionally uses USP-4, text-encoder TP-4, and spatial-tile VAE patch parallelism 4.

| implementation | GPUs | target, s | frames | scheduler points | observed forwards | e2e median (range), s | denoise median (range), s | peak MiB | route | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| vllm-omni base Ref2VA | 1 | 5 | 124 | 50 | 49 | 463.700 (463.462-463.907) | 441.801 (441.525-442.237) | 142652 | dense CuTe FA4 + regional compile | PASS |
| vllm-omni base Ref2VA | 1 | 10 | 243 | N/A | N/A | **N/A** | **N/A** | **N/A** | runtime probes | fresh one-forward warmup: CUDA illegal-address |
| vllm-omni base Ref2VA | 1 | 15 | 345 | N/A | N/A | **N/A** | **N/A** | **N/A** | runtime probes | fresh one-forward warmup: CUDA illegal-address |
| vllm-omni base Ref2VA | 4 | 5 | 124 | 50 | 49 | 135.535 (135.448-136.668) | 126.628 (126.621-127.808) | 97686 | dense CuTe FA4 + regional compile | PASS |
| vllm-omni base Ref2VA | 4 | 10 | 243 | 50 | 49 | 416.717 (415.745-417.267) | 399.994 (399.991-400.441) | 109316 | dense CuTe FA4 + regional compile | PASS |
| vllm-omni base Ref2VA | 4 | 15 | 345 | 50 | 49 | 758.643 (758.533-760.182) | 735.051 (734.863-735.209) | 125762 | dense CuTe FA4 + regional compile | PASS |

The 1x 243f/345f rows are N/A, not extrapolations. Fresh default-route job 2906 reproduced the
failures. Together, jobs 2906/2910/2912 tested FA4 dynamic/static/eager, CuDNN dynamic, and SDPA
eager at 243f; every route failed during the two-point/one-forward shape warmup while sampled
peaks stayed below the 189471-MiB GB200 capacity. These are runtime/kernel failures, not reported
OOMs and not valid 49-forward timing runs. Successful base rows come from job 2892.

#### FastVideo official-base d4 on the separate job-3002 input contract

This is a genuine 50-point base Ref2VA grid, not the four-forward proxy. Slurm array job 3002
completed all six cells with exit code 0 at exact clean code SHA
`99cd355a2452ce040591fe54ef340d192e26fe48`. Every request loaded the official
`/mnt/lustre/vlm-k1kong/models/MiniMax-H3/transformer_ref` component as the sole denoiser; the
Preview checkpoint was not used. Each cell ran one excluded full warmup followed by three saved
seed-1000 requests. Activation traces prove 49 DiT forwards per request and four requests per
rank, rather than inferring the forward count from the 50 scheduler points.

The row-local prompt is `A cinematic drone shot over coastal cliffs at sunrise, golden light,
gentle ocean waves, ultra detailed.` (SHA256
`92a4224855ef1ea61602eb4debdb1a32dbd9a301fbdece1623ef2af4cdb06e60`), **not** synth64
prompt 0. The reference is
`ref2va_grid/B/videos/ref_372f_768x1344_24fps.mp4` (SHA256
`5bce6f81e97ec726ad6143154a258bd149351880937ba10c78fa6d6ed0e0e636`), a byte-pinned
372-frame VFR H.264/AAC source. Both hashes differ from the older vllm-omni/proxy contract.
Accordingly these rows establish FastVideo base behavior across durations and GPU counts, but
make **no direct vllm-omni speedup or match claim**.

| implementation | GPUs | target, s | frames | scheduler points | observed forwards | e2e median, s | e2e minimum, s | e2e maximum, s | denoise median, s | denoise minimum, s | denoise maximum, s | reference encode median, s | video decode median, s | peak MiB | job |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FastVideo official-base d4 Ref2VA | 1 | 5 | 124 | 50 | 49 | **455.290** | 455.216 | 455.662 | **436.043** | 435.882 | 436.116 | 10.358 | 6.205 | 78949 | 3002_0 |
| FastVideo official-base d4 Ref2VA | 4 | 5 | 124 | 50 | 49 | **125.354** | 124.526 | 125.417 | **117.453** | 117.345 | 117.515 | 2.946 | 1.915 | 78799 | 3002_3 |
| FastVideo official-base d4 Ref2VA | 1 | 10 | 243 | 50 | 49 | **1437.518** | 1437.463 | 1437.970 | **1405.001** | 1404.674 | 1405.095 | 19.053 | 9.837 | 85308 | 3002_1 |
| FastVideo official-base d4 Ref2VA | 4 | 10 | 243 | 50 | 49 | **381.834** | 381.676 | 382.284 | **369.190** | 369.080 | 369.253 | 5.615 | 3.500 | 79015 | 3002_3 |
| FastVideo official-base d4 Ref2VA | 1 | 15 | 345 | 50 | 49 | **2686.231** | 2683.925 | 2686.743 | **2637.297** | 2637.215 | 2637.369 | 26.637 | 17.454 | 93762 | 3002_2 |
| FastVideo official-base d4 Ref2VA | 4 | 15 | 345 | 50 | 49 | **700.342** | 700.178 | 700.541 | **683.647** | 683.536 | 683.758 | 7.978 | 4.376 | 79257 | 3002_3 |

All rows use replicated DiT weights, packed-varlen FA4, regional
`torch.compile(fullgraph=True)` on 52 DiT submodules, a compiled VAE decoder, and no H3 fusions.
At four GPUs, both reference encode and target decode use four-rank temporal parallelism; the
same flags are one-rank no-ops at one GPU. Route gates found no SDPA fallback, FSDP transformer,
regional-compile disable, or H3-fusion marker. Every rank produced 196 trace records
(`49 forwards × 4 requests`), every timed triplet was byte-identical within its cell, and all
outputs passed the strict frame, codec, resolution, fps, and audio checks.

The terminal machine receipt is `results_summary_job3002.json` under
`h3_duration_grid_20260822/ref2va/fastvideo_official_base_99cd_d4_20260822/` (SHA256
`86d8ef86d222ee1c1cf6a70a431a72753de3dbf72200c7e31a6ffd188587bcd2`). Its all-or-nothing
builder SHA256 is `628353860aee63d2b17ccfc5a1ce85d3d6e1f3c82efaf09e3e963cc890fa5420`;
the timed harness SHA256 is `b6f1fe05d1ae225e49e0f0a5af0aabbdab5094053d29c76f59e94f24d77b217e`.
The six cell-receipt SHA256s, in table order, are
`4184fb7a808773a6a980cb0fc540af22d74851d2c23e6dc6bd929b8c130c11b6`,
`14a5801baff1de9458d23c471782dbcd3f0a778ba55a48872ec55cf50c76dccc`,
`03c66272d5681b701ab251ec6bb48e3ad8e5499aeccca404af9a477dea27cc6c`,
`ad557f62fa9fcf20ecf028473b0cdf01988c8b16fece5332a2dfa04207abce83`,
`fa5aa5fd0a447bff5f133ac76573c6847f2527bf95bf47f6612fbf59bd88390f`, and
`03d4cdb6d47157e693c70ac4bb8f572087405649f78b3a77d7ee3b462d527775`.
The source is VFR while generated media are strict 24-fps CFR; a post-submission validator-only
correction fixed source-media validation without changing the byte-pinned harness or timed path.

The FastVideo proxy uses eager sparse DiT, VSA@0.9, compiled VAE, H3 fusions, and parallel
reference encode/decode at SP-4. It is report-only/non-parity.

| implementation | GPUs | target, s | frames | scheduler points | observed forwards | e2e median (range), s | denoise median (range), s | peak MiB | sparse route | job |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| FV **base-weight proxy, not FastH3** | 1 | 5 | 124 | 5 | 4 | 65.135 (64.435-65.247) | 45.414 (45.332-45.425) | 84929 | sm100a CUDA-64 | 2877 |
| FV **base-weight proxy, not FastH3** | 1 | 10 | 243 | 5 | 4 | 164.521 (163.042-172.343) | 130.868 (130.618-130.871) | 97494 | FA4 CuTe-256 | 2894 |
| FV **base-weight proxy, not FastH3** | 1 | 15 | 345 | 5 | 4 | 292.989 (292.382-294.289) | 246.206 (246.120-246.318) | 117289 | sm100a CUDA-64 | 2901 |
| FV **base-weight proxy, not FastH3** | 4 | 5 | 124 | 5 | 4 | 21.733 (21.499-22.125) | 12.847 (12.814-12.884) | 82517 | sm100a CUDA-64 | 2877 |
| FV **base-weight proxy, not FastH3** | 4 | 10 | 243 | 5 | 4 | 48.282 (48.166-48.804) | 34.886 (34.869-35.010) | 83006 | FA4 CuTe-256 | 2894 |
| FV **base-weight proxy, not FastH3** | 4 | 15 | 345 | 5 | 4 | 81.666 (81.395-82.904) | 64.025 (63.997-64.149) | 89117 | FA4 CuTe-256 | 2894 |

At 243f the legacy CUDA-64 even-tile predicate rejects the exact packed Ref2VA geometry, so the
supported primary route is CuTe-256. Both 345f routes completed: 1x selected CUDA-64 at
292.989/246.206 s versus CuTe-256 at 293.343/249.344; SP-4 selected CuTe-256 at
81.666/64.025 versus CUDA-64 at 83.884/66.109. Route timings are not asserted bit-identical.

### Duration-grid receipts

- T2VA: `vsa_gate/h3_duration_grid_20260822/t2va/{RESULTS_T2VA.md,RESULTS_T2VA.json}` (SHA256
  `ab29b5c4...` / `d828f7d6...`). The machine receipt hashes every raw result, log, primary MP4,
  and validated media contract. Primary jobs: 2657, 2707, 2885, 2890, 2903, 2908, 2961, 2962.
- FastVideo official-base duration completion: SP-4 job 2963 is under
  `t2va/fastvideo_base_duration_99cd_20260822/` (`RESULTS.md` SHA256 `3f9a82ae...`; 243f/345f
  receipt SHA256 `a63716bc...` / `33f4076c...`). The 1x job-2964 ledger is under
  `h3_duration_grid_20260822/fastvideo_base_1x_10s_15s/` (`RESULTS_FASTVIDEO_BASE_1X.md` /
  `.json` SHA256 `1709d222...` / `fd089d84...`; 243f/345f receipt SHA256 `d5ad5b78...` /
  `822f350d...`). Both workspaces retain the launchers, terminal logs, raw timing JSON,
  schedule contracts, warmups, and all timed MP4s.
- T2VA comparisons: `vsa_gate/h3_duration_grid_20260822/comparisons_vllm_vs_f3/`; each
  `.receipt.json` names and hashes both inputs and the output. The sibling `README.md` records
  the final 15s/1x N/A state.
- Original matched-input Ref2VA vllm-omni/proxy grid:
  `vsa_gate/ref2va_duration_grid_20260822/{RESULTS.md,RESULTS.json}` (SHA256 `a9f7220e...` and
  `55509996...`). It contains every raw timing, MP4/ffprobe contract, route probe, failure
  envelope, command, and environment receipt. Primary jobs: 2877, 2892, 2894, 2901;
  failure/probe jobs 2906, 2910, 2912.
- FastVideo official-base d4 Ref2VA, separate row-local contract:
  `vsa_gate/h3_duration_grid_20260822/ref2va/fastvideo_official_base_99cd_d4_20260822/`.
  `results_summary_job3002.json` has SHA256 `86d8ef86d222ee1c1cf6a70a431a72753de3dbf72200c7e31a6ffd188587bcd2`
  and hashes all six terminal cell receipts. Primary array job: 3002.
- Main T2VA code SHA is `99cd355a`; only corrected FastH3 F4 345f uses private composition
  `8f8529d9`. Ref2VA FastVideo proxy and official-base d4 code are `99cd355a`, under the distinct
  input contracts disclosed above. vllm-omni is
  `73b623f2f7db092053c1c86fe796bed89eb3dc71` plus timing-only profiler patches. Hardware is
  GB200 (189471 MiB/GPU), driver 580.82.07. No benchmark launcher overrides `HOME`; caches are
  explicit and job/task scoped.

### Public inference lineage and excluded PR #1740 factor

The public-serving extraction is intentionally split by concern. As of this refresh, #1741
(regional inference compile) is merged at public main `d3cff517c`; #1742 (packed-varlen FA4)
is conflict-free at `9eb7b5d3a`; #1743 (schema-inventory repair) is at `d1ee99ac2`; #1744
(parallel VAE) is at `88e241a75`; #1745 (odd-tile sm100a) is at `82a5b0db6`; and #1746
(external multi-node launcher) is open, mergeable, and awaiting review at `accc5a420`. None of
these public PRs contains a training path or training configuration. The exact
`e0bb6a5`/`bbc8d35` acceptance and the job-3026 `20d026fd2` multi-node composition above are the
current public-serving authorities; `99cd355a` remains the historical timing and feature-
attribution authority for rows that have not been rerun on those compositions.

PR #1740 (fused Ulysses NVLink all-to-all) was re-audited after the public-head acceptance and is
unchanged at exact head `61ab307aaf469969d2b72d36b39a4d835a19867f`; it remains an explicit
**excluded factor**, not a missing timing. Exact-head review found two distributed-correctness
blockers: up to 36 CUDA CTAs reuse one NCCL LSA barrier index, and only the initial capability
decision is voted while allocation, build, per-call guards, and buffer growth can diverge
rank-locally. The current benchmark kernel prefix also lacks the new communication symbol, and
the raw pybind path is not traceable inside base d4's `fullgraph=True` regions. Running a timing
grid before those issues are fixed risks a mismatched collective hang and would not prove
engagement.

| topology | #1740 grid status | reason |
|---|---|---|
| 1x | N/A (intended no-op) | no Ulysses collective or allocation exists at world size one |
| SP-4, one tray | excluded (not run) | potentially useful topology, but current head is not group-atomic or safely initialized |
| SP-8, two trays | excluded (not run) | current NVL72 layout is expected to expose LSA teams of four, requiring unanimous 0/8 fallback |

No number in this document includes #1740. PR #1739's packed-SP route also bypasses #1740, and
Preview VSA rejects packed SP, so those factors must never be described as additive.

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

| leg | 1x e2e | 1x denoise | SP-4 e2e | SP-4 denoise | section-1 baseline 1x e2e | section-1 baseline SP-4 e2e |
|---|---:|---:|---:|---:|---:|---:|
| a. dense FA4 50-step | 186.6 (186.5-186.7) | 176.39 | 66.0 (65.7-66.2) | 52.63 | 185.2-188.2 | 62.8-63.7 |
| b. FastH3 VSA-64 triton | 22.9 (21.7-25.3) | 12.48 | 17.2 (16.4-18.7) | 4.14 | 22.3-22.8 | 13.6-15.0 |
| c. FastH3 VSA-64 sm100a | 19.4 (19.0-19.7) | 10.80 | 15.9 (15.3-16.3) | 3.72 | 20.7 | 15.1 |
| d. b + H3 fusions (non-parity) | 20.2 (19.6-20.6) | 10.43 | 15.4 (14.0-16.2) | 3.63 | — | — |

### Per-stage deltas vs the section-1 baselines

| stage | stack 1x | stack SP-4 | baseline 1x | baseline SP-4 | delta 1x | delta SP-4 |
|---|---:|---:|---:|---:|---|---|
| text encoding (#1732 slim + merged-#1711) | 0.53-0.55 | 0.44-0.59 | ~2 | ~2 | **about -1.5 s (-75%)** | **about -1.5 s (-75%)** |
| denoising, dense FA4 (49 fwds) | 176.39 | 52.63 | 175-179 | 52.9-54.2 | parity | parity |
| denoising, student, Triton | 12.48 | 4.14 | 12.4 | 4.3 | parity to -5% | parity to -5% |
| denoising, student, sm100a | 10.80 | 3.72 | 10.7 | 3.9 | parity to -5% | parity to -5% |
| denoising, +fusions vs leg b | 10.43 | 3.63 | — | — | **-16%** | **-12%** |
| video VAE decode (#1703 streaming + #1734 tile compile) | 7.05-8.81 | 9.9-11.7 | 6.4-7.5 | 7.26 stock | **REGRESSION, see below** | **REGRESSION, see below** |
| audio decode + post + save | ~0.8-1.1 | ~0.9-1.2 | ~1 | ~1 | parity | parity |

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
2. Base H3 packed d4 matches or slightly beats every supported matched vllm-omni cell: 5s and
   10s at both 1x and SP-4, plus 15s at SP-4, all on the exact 49-forward contract. The valid
   FastVideo 15s 1x result is one-sided because the vllm-omni comparator is N/A. d4 remains a
   report-only route; keep fixed FA4 for parity-sensitive work until the deterministic
   cross-route drift receives an explicit quality-acceptance decision.
3. Genuine FastH3 Preview Ref2VA is N/A because the export has no distilled `transformer_ref`.
   Never label the four-forward FastVideo base-weight latency proxy as FastH3 or quality-valid.
4. Ref2VA/long-sequence (>=100k): never Triton-256; select a supported CuTe-256 or sm100a-64
   route by exact packed geometry. P2 ref-sparsification remains the training/quality direction
   (keep 0.10 trained / 0.25 zero-finetune), not part of the section-7 serving grid.
5. Do not infer compile coverage from the headline alone: base d4 and vllm-omni regionally compile
   the DiT; FastH3 F3/F4 keep the sparse DiT eager and compile only the decoder. The Ref2VA proxy
   likewise uses eager sparse DiT.
