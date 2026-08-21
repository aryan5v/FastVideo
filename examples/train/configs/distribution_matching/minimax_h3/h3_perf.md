# MiniMax-H3 / FastH3 performance numbers (measured, GB200)

Consolidated inference/serving measurements as of 2026-08-21. All FastVideo numbers: branch
`h3-dmd` (tip `6b51634ee` era), lustre venv (torch 2.12.0+cu130), driver 580.82.07, synth64
prompt 0, seed 1000, warmup excluded, >=2-3 timed repeats (spreads <=1% unless noted). "FastH3"
= the 4-forward DMD2 student (step-1400 export) with VSA @0.9. Raw artifacts:
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/{sm100a_e2e,sp8,ref2va_grid,vsa64bench}` and
`vllm_omni_bench/`; master index `~/h3-results-index.md`.

## 1. T2VA @ 5s (768x1344, 124 frames, S=38,010)

### End-to-end (seconds)

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
- Sharding note: multi-node inference requires the external-launcher path
  (`FASTVIDEO_EXTERNAL_LAUNCHER=1`, torchrun; byte-identical gate vs stock executor; unmerged
  worktree commits `c65733463`+`ac24e7e78`).

### Cross-stack: vllm-omni (same workload, FA4 verified, their commit 73b623f2)

| config | e2e | FastVideo equivalent | ratio |
|---|---:|---:|---:|
| 1 GPU dense FA4 50-step (no offload) | **135.8 s** | 188.2 s | 0.72x |
| 1 GPU + CPU offload | 140.7 s (82.6 GB peak) | — | — |
| 4 GPU (USP-4 + text-encoder TP-4 + VAE patch-parallel + regional compile) | **40.7 s** | 62.8 s | 0.65x |

Their 4-GPU margin comes from regional compile + parallel VAE decode + encoder TP — our #1718
compile port (unenabled) and the #1703-seam parallel decode (unbuilt) are the corresponding levers.

## 2. Ref2VA @ 15s (345 frames, 768x1344 ref + target, S=220,628)

Base `transformer_ref` weights; full grid in `vsa_gate/ref2va_grid/ROOFLINE.md`.

| leg (SP-4 unless noted) | fwds | DiT s/fwd | e2e |
|---|---:|---:|---:|
| dense FA4 50-step, 1x GPU (fits: 99.6 GiB) | 49 | 81.5 | 4046 s |
| dense FA4 50-step | 49 | 21.0 | 1084 s |
| VSA@0.9 **Triton-256, current policy — DO NOT USE at this scale** | 4 | 37.6 (1.79x SLOWER than dense) | 203 s |
| VSA@0.9 tile-64 sm100a, current policy (P1) | 4 | 17.3 (1.22x) | 121 s |
| **P2 ref-sparsified keep-0.10, CuTe-256 (50-step)** | 49 | **9.2 (2.34x denoise, 2.17x e2e)** | **513 s** |

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

Triton-64 saturates ~536 TF from 65k. H3 layer path (56 heads, prod 5s shape, sparsity 0.9):
19.2 ms (Triton-64) → 11.4 ms (sm100a, 1.68x). Dense-50-vs-3-step legacy headline (VSA-256,
124f): 4.66x e2e / 14.1x denoise (job 2432 era).

## 4. Training throughput (v8, for context)

Job 2486: 32 GPUs, global batch 64, ~67-69 s/step (carried backward-simulation walk + VSA-64
Triton student; v7 was ~81 s/step, v6 dense ~same MFU). MFU tables and accounting:
`h3_dmd.md` MFU section + `scripts/train/mfu_calc_minimax_h3.py`.

## Route guidance (from the measurements)

1. Student serving at 5s: SP-4, VSA-64 sm100a (`FASTVIDEO_VSA_SM100A=1`), 4 forwards
   (`num_inference_steps=5` upstream-scheduler convention).
2. Teacher/50-step at 5s: SP-8 external launcher if available, else SP-4 dense FA4.
3. Ref2VA/long-sequence (>=100k): never Triton-256; CuTe-256 or sm100a-64; adopt P2
   ref-sparsification (keep 0.10 trained / 0.25 zero-finetune).
4. The next e2e wins are NOT attention: parallel VAE decode+encode, regional compile for
   inference, encoder residency (#1711's 13.7 GB saving).
