# M5 Survey Results — 2026-08-03

Machine: Apple M5, 24 GB unified, macOS 26.5.2 (25F84), MLX 0.32.0. Both
accelerator gates passed. All probed modes supported. Checkpoints:
`FastVideo/FastWan2.1-T2V-1.3B-Diffusers` and `FastVideo/FastWan-QAD-1.3B`.

## Reconstruction error

| Mode | plain rel L2 | QAD rel L2 | vs int8 | bits/param | 14B weights |
|---|---|---|---|---|---|
| **int8** | **0.00546** | 0.00546 | 1.0× | 8.50 | 14.9 GB |
| mxfp8 | 0.06793 | 0.06826 | 12.4× | 8.25 | 14.4 GB |
| int4 | 0.09191 | 0.09195 | 16.8× | 4.50 | 7.9 GB |
| nvfp4 | 0.10287 | 0.10290 | 18.8× | 4.50 | 7.9 GB |
| mxfp4 | 0.12089 | 0.12102 | 22.1× | 4.25 | 7.4 GB |

## Throughput

| Shape | int8 | mxfp4 | nvfp4 |
|---|---|---|---|
| 8192×1536 | 3.469 ms | 3.416 ms | 3.465 ms |
| 8192×5120 | 43.461 ms | 42.142 ms | 41.671 ms |

## Four conclusions

### 1. Integer formats beat float formats at equal bit width — decisively

int8 (8.5 bits) is **12.4× more accurate than mxfp8** (8.25 bits). Nearly the
same budget, an order of magnitude apart. Same story at 4 bits: affine int4
(4.5) beats nvfp4 (4.5) beats mxfp4 (4.25).

The mechanism is visible in the numbers. E4M3 carries 3 mantissa bits, so ~6%
relative precision per element — and mxfp8's measured 6.8% error is essentially
that figure. Affine int8 spends all 8 bits on 256 *uniform* levels across a
group's actual range, with a scale **and** a bias to fit it. For weight
distributions, which per-group scaling has already normalised, float formats
spend their bits on dynamic range that is no longer needed.

**This inverts the recommendation from the literature review.** Published
comparisons rank NVFP4 above MXFP4 — confirmed here — but they compare MX
formats against each other, not against affine int4 at group 64. Affine wins,
and nothing in the earlier research covered that case.

### 2. Affine error scales as 2^-bits, cleanly

int8 → int4 is 4 bits and gives a measured **16.8×** error increase against a
theoretical 16×. That is a tight enough fit to extrapolate from:

| Mode | predicted rel L2 | bits/param | 14B weights |
|---|---|---|---|
| int7 | ~0.011 | 7.5 | 13.1 GB |
| **int6** | **~0.022** | **6.5** | **11.4 GB** |
| int5 | ~0.044 | 5.5 | 9.6 GB |

**int6 is the gap in this survey and the most valuable thing left to measure.**
Predicted ~4× int8's error but ~4× better than int4 and ~5× better than nvfp4,
at 6.5 bits/param — which puts a 14B at **11.4 GB, comfortably inside a 24 GB
Mac, on the one grid already proven to hold up.** It was in the test plan; the
run used the earlier mode list and skipped it.

### 3. 4-bit buys no speed on M5 — it is purely a capacity play

All three modes land within noise: 1.5% apart at 1536, 4% at 5120.

The reason is bandwidth accounting. At 8192 tokens and dim 1536, weights are
only **8.6%** of memory traffic at fp16; activations dominate. Dropping weights
to 4-bit can save at most ~7% of traffic, and the measured difference matches.
At dim 5120 weights reach 23.8% and the gain grows to 4% — still small.

This is why the LLM quantization story does not transfer. LLM *decode* runs one
token at a time, so weights are nearly 100% of traffic and 4-bit is a large
speedup — that is the Ollama NVFP4 result. A video DiT at 8k+ tokens is in the
opposite regime.

**Corollary: quantization on Metal is for fitting a bigger model, not for
running it faster.** Speed has to come from step count, sparse attention, or
resolution.

### 4. The QAD checkpoint is not distributionally different

nvfp4 error 0.10290 (QAD) vs 0.10287 (plain) — a 0.03% difference, and the same
across every mode. The two checkpoints have nearly identical weight
distributions, consistent with QAD being a light finetune of the distilled
model.

**But this test cannot settle whether QAD's quality transfers.** QAT does not
optimise weights toward low reconstruction error; it trains the network to
produce good *outputs* despite quantization error. A QAT'd model can show
identical weight error and much better output. The reconstruction proxy is
silent here — only generating from it answers the question.

## Decision

**Ship affine int8. Measure int6 as the path to 14B on 24 GB. Drop the MX
formats.**

- mxfp4 was the **worst** option tested, and gains nothing on speed. The earlier
  mxfp4 run failures now have a partial format explanation on top of the
  uncommitted-quantizer problem.
- nvfp4 loses to affine int4 at identical memory. There is no reason to prefer
  it on this hardware.
- No 4-bit format is worth a distillation run when int6 likely delivers 14B in
  11.4 GB at ~4× int8 error rather than ~19×.

## Next

1. **Re-run the survey with `--modes int8 int7 int6 int5 int4`.** Minutes, and
   it decides the 14B story.
2. **Generate from the QAD checkpoint in MLX at int8** to settle conclusion 4,
   which reconstruction error cannot.
3. If int6 holds, distill the 14B at int6 using `dmd2_t2v_14b_mlx_int8.yaml`
   with `bits: 6` — the affine QAT path is already committed and parity-tested.
