# MiniMax-H3 / FastH3 performance numbers (measured, GB200)

Consolidated inference/serving measurements as of 2026-08-21. All FastVideo numbers: branch
`h3-dmd` (tip `6b51634ee` era), lustre venv (torch 2.12.0+cu130), driver 580.82.07, synth64
prompt 0, seed 1000, warmup excluded, >=2-3 timed repeats (spreads <=1% unless noted). "FastH3"
= the 4-forward DMD2 student (step-1400 export) with VSA @0.9. Raw artifacts:
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/{sm100a_e2e,sp8,ref2va_grid,vsa64bench}` and
`vllm_omni_bench/`; master index `~/h3-results-index.md`.

## 1. T2VA @ 5s (768x1344, 124 frames, S=38,010)

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

Base `transformer_ref` weights; full grid in `vsa_gate/ref2va_grid/ROOFLINE.md`.

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

## 5. Stacked-PR integration branch (1362+1703+1732+1734+1735+1731)

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

## Route guidance (from the measurements)

1. Student serving at 5s: SP-4, VSA-64 sm100a (`FASTVIDEO_VSA_SM100A=1`), 4 forwards
   (`num_inference_steps=5` upstream-scheduler convention).
2. Teacher/50-step at 5s: SP-8 external launcher if available, else SP-4 dense FA4.
3. Ref2VA/long-sequence (>=100k): never Triton-256; CuTe-256 or sm100a-64; adopt P2
   ref-sparsification (keep 0.10 trained / 0.25 zero-finetune).
4. The next e2e wins are NOT attention: parallel VAE decode+encode, regional compile for
   inference, encoder residency (#1711's 13.7 GB saving).
