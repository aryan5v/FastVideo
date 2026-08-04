# W8A8 Fused INT8 GEMM for Metal — Design, Calibration, and M5 Speed Gate

Status: **CLOSED — Gate 2 NO-GO** (2026-08-04).  
Keep this file as the permanent receipt. Do not restart Gate 3 (W8A8 QAT)
unless MLX grows a real int8×int8 path or a lower-level kernel beats fp16.

Related: launch blog draft §"We Tried to Make INT8 Fast"
(`docs/blog/draft_fastwan_14b_int8_apple_silicon.md`), next-wins item 6
(`docs/design/mlx_runtime_next_wins.md`).

---

## Thesis (what we set out to prove)

Weight-only quantization cannot buy speed on Metal: `mx.quantized_matmul`
dequantizes to fp16 and runs fp16 arithmetic, so the integer matrix units
sit idle (M5 survey + `mac_quantization_findings.md`). The only route to
**quantization-derived speed** is quantizing **activations as well as
weights** (W8A8) so an integer compute path can engage.

Ideogram shipped fused INT8 GEMM for consumer GPUs. Nobody had done it for
diffusion on Metal against M5 Neural Accelerators. We tried.

Ship bar (pre-registered): fused W8A8 must beat fp16 GEMM by a **meaningful
margin** at DiT shapes; **≤1.2× is not worth shipping**.

---

## Gate 1 — activation calibration: **PASS**

| Artifact | Job | Result |
|---|---|---|
| `w8a8/calib_5b.json` | GB200 **1112** | 306 Linears, 64 samples × 3 DMD steps, 720p latent, randn context |
| `w8a8/calib_5b_realctx.json` | GB200 **1118** | same geometry, **real UMT5** embeds (8 prompts) |
| `w8a8/calib_14b.json` | GB200 **1117** | 406 Linears on `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| `w8a8/scales_5b.json` | CPU export | per-layer scales + `fp16_keep` list |

Findings (still valid even after Gate 2 NO-GO):

1. **Per-timestep drift is modest.** Median `absmax(t522)/absmax(t1000) ≈ 0.90`
   on 5B. One scale per layer (or 3 step-buckets) is enough for the 3-step
   DMD schedule.
2. **Bulk fits int8 with per-token scales.** Self-attn + FFN look healthy
   (`p999/absmax` median ~0.6–0.7).
3. **Outliers live in the text path.** Worst layers are almost all
   `attn2.to_q` (and some `condition_embedder.*`). Real UMT5 context makes
   this **worse**, not better: median absmax drops (~0.76×) but
   `attn2.to_q` absmax inflates up to **~3.7×** vs randn context.
4. **Scale export:** 22 / 306 layers (7.2%) marked `fp16_keep` at
   `p999/absmax < 0.25` — **all** `attn2_cross`. Self-attn and FFN: zero
   keeps. Same pattern on 14B (worst p999/absmax ~0.03–0.06 on early
   `attn2.to_q`).

**Gate 1 verdict:** W8A8 is *numerically plannable* at int8 with per-token
activation scales and fp16 for cross-attn conditioning projections. That is
necessary but not sufficient — the kernel still has to win on wall-clock.

---

## Gate 2 — integer GEMM kernel: **NO-GO on M5**

### Implementation

Prototype: `fastvideo/mlx_runtime/w8a8_gemm.py` on branch
`mlx-two-pass-refine` @ `df5797fc`.

- Custom `mx.fast.metal_kernel`: int8×int8 → int32 accumulate →
  `float(acc) * scale_a[m] * scale_b[n]`
- Schemes: **naive** (1 thread / output) and **tiled** (8×8 TG, TK=32)
- API matches Gate 1: per-token act scales + per-out-channel weight scales
- Correctness oracle: dequantized float64 reference
- Microbench vs fp16 `x @ w.T` and weight-only `mx.quantized_matmul`

### Correctness (M2 and M5)

| Machine | Naive max abs err | Tiled max abs err |
|---|---|---|
| M2 8 GB (dev) | 3.0e-6 | 3.0e-6 |
| **M5 24 GB (friend machine)** | **3.016e-6** | **3.016e-6** |

Commit probed on M5: `df5797fc174060f26ab58084867ad1f6a78d5846`.  
MLX 0.32.0, macOS 26.5.2, Metal available, `metal_kernel` smoke PASS.

### Speed — M5 24 GB (MacBook Air Mac17,3), median ms

Speedup = `fp16_ms / w8a8_ms`. Ship bar = **≥ 1.2×**.

| Shape (M×N×K) | fp16 GEMM | W8A8 naive | W8A8 tiled | Weight-only QMM | Best fused vs fp16 |
|---|---:|---:|---:|---:|---:|
| 128×128×128 | 0.924 | 1.173 | 1.087 | 0.872 | **0.85×** |
| 256×256×256 | 1.024 | 6.098 | 5.985 | 1.401 | 0.17× |
| 512×512×512 | 0.393 | 4.851 | 22.395 | 14.283 | 0.08× |
| 1024×1024×1024 | 1.820 | 73.647 | 81.066 | 3.056 | 0.02× |
| 2048×2048×2048 | 6.399 | 316.5 | 365.9 | 11.033 | 0.02× |
| **1024×5120×5120** (DiT-ish) | 8.708 | 447.7 | 677.3 | 16.251 | **0.02×** |
| **4096×5120×5120** | 52.64 | 1909 | 1341 | 23.64 | **0.04×** |
| **8192×5120×5120** | 52.74 | 1755 | 3889 | 115.2 | **0.03×** |
| 512×3072×3072 | 2.783 | 138.4 | 156.7 | 2.028 | 0.02× |

No shape skipped or OOMed. Best fused speedup anywhere: **0.85×** (tiled,
128³) — still under the 1.2× bar. At every Wan Linear-class shape the fused
kernel is roughly **25–100× slower than fp16**.

### Why (mechanism, not vibes)

1. `mx.fast.metal_kernel` runs a **hand-written scalar MAC** loop in shader
   code. It does **not** bind the M5 Neural Accelerator integer tensor path
   (MLX 0.32 has no public int8×int8 NA API).
2. Apple's fp16 GEMM is a **vendor-tuned** path. Scalar int8 cannot beat it
   by burning fewer bits when the fp16 unit is already saturated and the
   int8 path is not hardware-accelerated.
3. Weight-only `mx.quantized_matmul` stays near fp16 (sometimes slightly
   faster at small shapes, sometimes slower) — consistent with
   dequant-then-fp16, not integer GEMM.

### Gate 2 verdict

**NO-GO.** Correctness is production-grade; speed is not. Do not productize
this kernel. Do not spend GB200 on W8A8 QAT (Gate 3) until a kernel exists
that clears 1.2× fp16 at DiT shapes.

Fallback (documented, not pursued as a release feature): per-token act
quant into the existing weight-only path — small memory win, **zero** speed
win.

---

## Gate 3 — W8A8 QAT: **BLOCKED / cancelled**

Was: add activation fake-quant to `MLXQuantizationAwareCallback`, train on
GB200 with Gate-1 scales.  
Now: **cancelled** as a speed path. Reopen only if Gate 2 is re-lit by a
new MLX/Metal capability.

Weight-only INT8 QAD (the 5B/14B launch trains) is **orthogonal** and
continues — that is the memory/quality deploy grid, not this speed bet.

---

## What we still ship because of this work

Even a NO-GO is a launch asset:

- **Calib corpus** for any future act-quant work (5B randn, 5B realctx, 14B).
- **`fp16_keep` recipe** for cross-attn projections (empirically ~7% of
  Linears on 5B).
- **Prototype kernel + bench harness** (`w8a8_gemm.py`) as a regression
  oracle if MLX ever exposes int8×int8 NA matmul.
- **Blog-grade receipts** that INT8-on-Mac is a *memory* decision, and that
  we measured the ambitious alternative rather than hand-waving it away.

Mac speed levers that remain in force (pipeline, not GEMM format):

- RIFE `--fast`, `--fast-spatial`, two-pass `--refine`, prompt enhance,
  `mx.compile`, AdaLN cache, sequenced residency, 1–2 step distill.

---

## References

- Prototype: `fastvideo/mlx_runtime/w8a8_gemm.py` (`mlx-two-pass-refine` @ `df5797fc`)
- Calib + scales: cluster `$T/w8a8/` under `wan14b-qad-int8-20260803`
- Format decision / traffic: `mac_quantization_findings.md`, `m5-quant-backends.md`
- Next-wins ranking: `mlx_runtime_next_wins.md` §6
- Ideogram INT8 GEMM precedent: https://arxiv.org/html/2606.14598v1
- M5 probe machine: MacBook Air (Mac17,3), M5, 24 GB, macOS 26.5.2, MLX 0.32.0
