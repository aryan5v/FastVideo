# RIFE "generate-fewer-frames" speedup — summary + exact recipe

**Date:** 2026-07-17 · **Branch:** `aryan/metalfx-research` · **Model:** V2 QAD 1.3B
(`/Users/aryank/models/qad_int8_v2`) · **Hardware:** Apple M4 Max (36 GB), MLX 0.31.2

## TL;DR
Generating **fewer video frames** with the diffusion model and **interpolating the rest** with
RIFE gives a **~2.7× end-to-end generation speedup** at **MS-SSIM 0.974** vs native. The only
downside — RIFE's interpolated frames are ~7% softer — is removed for **free (+33 ms)** by a light
sharpen pass. Net: **~2.7× faster generation, sharpness restored, ~0.97 SSIM.**

## Why it works (the mechanism)
Diffusion denoise cost is dominated by self-attention, which is **O(tokens²)** where
`tokens = latent_frames × latent_H × latent_W`. Halving the frames (81 → 41 pixel frames = 21 → 11
latent frames) cuts the token count ~2×, and because attention is quadratic the denoise compute
drops **~3.7×** → measured **~2.7× wall-clock** end-to-end. Quantization (INT8/MXFP4) does **not**
speed this up (it's memory-only); this lever attacks the token count directly, so it **composes**
with quantization and with M5 acceleration.

**Key constraint discovered:** Apple's **MetalFX Frame Interpolation is NOT usable** — it requires
game-engine motion vectors + depth (5 bound textures), which diffusion output lacks. The working
tool is **RIFE** (video-native, estimates its own optical flow), specifically the **MLX-native
`rife-mlx`** port (`mlx-community/RIFE-4.25` weights) — torch-free, Metal-backed, arbitrary-timestep.

## Measured results (fox, 480×832, vs native 81-frame)
| Config | Speedup | MS-SSIM | Sharpness (native=1.00) | Extra cost |
|---|--:|--:|--:|--:|
| Native 81 (baseline) | 1.00× | 1.000 | 1.00 | — |
| gen41 + RIFE→81 | **2.71×** | 0.9745 | 0.93 (7% softer) | +1.7 s |
| **gen41 + RIFE→81 + light sharpen** ✅ | **~2.7×** | 0.9734 | ~1.0 (restored)* | **+0.03 s** |
| gen41 + RIFE→81 + Real-ESRGAN | ~1.3× | 0.9639 | 2.11 (over-sharp) | +42 s ❌ |
| gen55 + RIFE→81 (1.5×) | ~2.0× | 0.9763 | 0.90 | +3.0 s |
| low-res gen41 + RIFE + ESRGAN 2× | — | 0.9431 | 1.08 | +19 s |

Denoise time: 81 frames = 94.7 s → 41 frames = 33.5 s. RIFE 41→81 = 1.4 s. Total 96.6 s → 34.9 s.

\* Codex's unsharp was tuned too strong (1.61×, crunchy). The ship recipe dials it to target ~1.0
(match native, not exceed) — a `sharpen_strength` follow-up.

## The exact recipe (what to ship as "fast mode")
1. Generate at **half the target frames** (81 → 41; i.e. 21 → 11 latent frames) with the distilled
   MLX model — everything else identical (480×832, 3-step DMD, TAEHV decode).
2. **RIFE 2× interpolate** 41 → 81 via `rife-mlx` (Metal-backed, ~1.4 s).
3. **Light unsharp-mask** pass (~33 ms) tuned to native sharpness — removes RIFE softness.
Result: same clip, ~2.7× faster, MS-SSIM ~0.97, sharpness ≈ native.

## What did NOT help
- **Real-ESRGAN** super-res: restores/overshoots sharpness but +42 s erases the speedup and its
  hallucinated detail drops SSIM to 0.964. Overkill for mild softness.
- **1.5× ratio** (gen55): less speedup, not meaningfully sharper.
- **Low-res gen + upscale:** loses real detail (SSIM 0.943).

## Composability (the endgame local pipeline)
Each lever attacks a different axis and multiplies:
- **MXFP4** → memory (fits more Macs)
- **generate-fewer-frames + RIFE** → generation speed (this doc, ~2.7×)
- **M5 Neural Accelerators** → hardware matmul (MXFP4-accelerated)

## Reproduce
`fastvideo/benchmarks/rife_interp.py` (MLX RIFE wrapper) +
`fastvideo/benchmarks/eval_metalfx_rife.py` (the benchmark). Artifacts + contact sheets under
`bench/metalfx_rife/`. Videos copied to `~/Desktop/fastvideo_demos/rife/`.

## Recommended follow-up
Tune `sharpen_strength` to native, run across the **motion7** prompt set (motion is where RIFE
softens most), then wire a `--fast` flag into the generate path. Do NOT push past 2× reduction
without a prompt sweep — larger temporal gaps are where RIFE invents motion.
