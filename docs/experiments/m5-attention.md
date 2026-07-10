# Experiment: Windowed (local) attention scaling for MLX Wan denoise

**Date:** 2026-07-10  
**Machine:** Apple M4 Max  
**MLX:** 0.31.2  
**Code:** `fastvideo/mlx_runtime/windowed_attention.py` + `fastvideo/mlx_runtime/attn_scaling_probe.py`  
**JSON:** `bench/accel/attn_m4.json`

## Motivation

A single full self-attention call at the real Wan denoise shape `(B=1, H=12, S=32760, D=128)` is
~0.6–1.0 s on Apple Silicon; with ~30 DiT layers that is essentially 100% of per-step time.
Full attention is **O(S²)**. This study measures how much a **chunked symmetric sliding-window**
attention (optional global sinks) saves at our shapes, and how badly it approximates dense
attention on fixed-seed random Q/K/V.

## Method

- **Full:** `mx.fast.scaled_dot_product_attention` (reference).
- **Windowed:** query tiles run SDPA only against the local key slice
  `[i − ⌊W/2⌋, i + ⌊W/2⌋]` (symmetric, non-causal) plus the first `sink` keys.
  Implementation is **chunked** — not a full-size additive mask — so work is
  `O(S · (W + sink) · D)`, not `O(S²)`.
- Shapes: `B=1, H=12, D=128`, fp16; `S ∈ {8192, 16384, 32760}`, `W ∈ {1024, 2048, 4096}`, `sink=0`.
- Warmup 2 + 5 timed iters; `mx.eval` each call. Approximation metric: **mean cosine
  similarity** between windowed and full outputs on the same Q/K/V.

## Results (this machine)

### Full attention scales as O(S²)

| S     | full s/call | ratio vs S=8192 |
|-------|-------------|-----------------|
| 8192  | 0.0385      | 1.0×            |
| 16384 | 0.1526      | 4.0× (S×2 → ~4×) |
| 32760 | 0.6174      | 16.0× (S×4 → ~16×) |

Wall time tracks S² cleanly on M4 Max Metal SDPA.

### Windowed speedup and approximation error

| S     | window | full_s | win_s  | speedup | mean_cos |
|-------|--------|--------|--------|---------|----------|
| 8192  | 1024   | 0.0385 | 0.0073 | 5.24×   | 0.419    |
| 8192  | 2048   | 0.0386 | 0.0115 | 3.36×   | 0.534    |
| 8192  | 4096   | 0.0384 | 0.0189 | 2.03×   | 0.692    |
| 16384 | 1024   | 0.1526 | 0.0147 | 10.37×  | 0.302    |
| 16384 | 2048   | 0.1523 | 0.0236 | 6.44×   | 0.384    |
| 16384 | 4096   | 0.1531 | 0.0404 | 3.79×   | 0.513    |
| 32760 | 1024   | 0.6174 | 0.0295 | **20.91×** | 0.215 |
| 32760 | 2048   | 0.6218 | 0.0484 | **12.86×** | 0.273 |
| 32760 | 4096   | 0.6242 | 0.0837 | **7.45×**  | 0.367 |

At the production sequence length **S=32760**, windowed attention is **7–21× faster** than full
SDPA depending on W. Mean cosine vs full on **i.i.d. Gaussian Q/K/V is low** (0.22–0.37 at that
S): windowed attention is a real approximation, not a free lunch. Random QKV is a harsh proxy
(attention mass is globally diffuse); real DiT activations may be more local, but this metric
honestly bounds “how close is the operator to dense attention” without a full video eval.

## Recommendation

**Worth prototyping into the dense DiT for speed**, but **not as a drop-in quality-neutral swap**
without video-level validation. At S≈33k the FLOP/runtime lever is huge (≈7× even at W=4096;
≈13× at W=2048). Prefer starting at **W=2048 or 4096** (better cosine than W=1024, still
double-digit speedup) and optionally a small sink (e.g. 64–256) for global tokens. Treat
windowed attention as a **quality/speed knob**: ship only after SSIM / human checks on real
denoise, not based on this microbench alone. If quality regresses, keep full attention for
early layers or alternate layers, or fall back to full at short S where speedup is modest.
