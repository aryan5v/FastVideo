# Apple Silicon Launch — Near-Term Checklist

Companion to `apple_silicon_minimax_h3.md`. This is the ship-in-days plan for
FastWan on Metal. No H3 dependency.

**Core call: the launch does not need mxfp4.** Ship int8 at both sizes, add
mxfp4 as a follow-up once the QAT path is fixed.

## What exists right now

| Asset | State |
|---|---|
| 1.3B int8 QAD | **done, quality accepted** — shippable today |
| 1.3B mxfp4 QAD | done, bad output — QAT path suspect |
| 14B mxfp4 QAD | done, bad output — same suspect path |
| 14B int8 QAD | **never run — the only thing gating a 14B launch** |
| MLX runtime at 14B | proven — the mxfp4 runs loaded and ran |

The runtime already handles 14B. The single missing artifact is 14B at int8.

## Day 1

**Ship 1.3B int8.** It is finished and the quality is accepted. Do not hold it
behind the 14B work.

**Start the 14B int8 QAD run immediately** — it is the long pole and the GPUs
are otherwise idle.

| GPUs | Wall-clock |
|---|---|
| 16 | ~39 h |
| 32 | **~20 h** |

Reuse `dmd2_t2v_mlx_int8_v2.yaml` — the exact recipe that produced the liked
1.3B result — with the student and critic pointed at `fastwan14b_distilled` and
`hsdp_shard_dim` set for the new GPU count.

**In parallel, no GPUs needed:**

- **14B int8 PTQ.** Quantize the existing 14B weights to affine int8 and look at
  the output. A few hours. 8-bit PTQ is forgiving and larger models PTQ better,
  so this may just work — in which case 14B ships without waiting on the QAD run.
- **STE diagnostic.** Log per-layer grid saturation and STE gradient-passthrough
  rate on the int8 path vs the mxfp4 path. The matched control shows mxfp4 with
  0.47× the student gradient norm at *lower* loss, which points at a clipped STE
  zeroing gradients at saturation. This decides whether mxfp4 is salvageable.

## Day 2–3

- 14B int8 lands → validate on Mac → this is the headline model.
- STE diagnostic reports. If it is a clipped-STE bug, the fix is small.
- Then, and only then, a corrected mxfp4 run. 1.3B ~16 h on 8 GPUs; 14B ~20 h on
  32. They can run concurrently.

## Launch shape

| Tier | Model | Weights | Mac |
|---|---|---|---|
| Fast | 1.3B int8 | ~1.4 GB | 16 GB+, any generation |
| Quality | 14B int8 | ~15 GB | 32 GB+ |
| *Follow-up* | 14B mxfp4 | ~7.4 GB | 24 GB, M5+ |

The 1.3B → 14B jump is what users will actually notice. mxfp4's contribution is
reach — it moves the 14B from a 32 GB machine to a 24 GB one — not output
quality.

## Making mxfp4 work — check the recipe before blaming the format

The int8 config that produced quality you liked is `dmd2_t2v_mlx_int8_**v2**`.
There is a v1, it failed, and v2's own header says exactly why:

> - Student and critic initialize from `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
>   (already 3-step DMD-distilled): training becomes "adapt a good 3-step model
>   to the INT8 grid" instead of learning distillation from scratch through a
>   quantizer. The teacher stays the base 50-step Wan2.1.
> - `gradient_accumulation_steps: 4` (effective batch 16): run 1's global batch
>   of 4 left the critic noisy and **the student under-trained (weights moved
>   only ~0.2% from init)**; motion coherence is the expected beneficiary.

**That failure description is the signature we observed in the mxfp4 runs** —
low student gradient norm, low loss, poor output. v1 failed the same way, on
int8, for reasons that had nothing to do with the numeric format.

The mxfp4 configs (`dmd2_t2v_1p3b_mlx_mxfp4.yaml`,
`dmd2_t2v_14b_mlx_mxfp4.yaml`) are **not in the committed tree** at the run
commit `411cfa7e` — they were local working-tree files. So whether they inherited
v2's two fixes or branched from v1 is unverified.

### The likeliest root cause: the mxfp4 fake-quantizer is uncommitted and untested

At the run commit `411cfa7e`:

- `fastvideo/layers/quantization/mlx_affine_qat.py` implements **the affine grid
  only** — `mx.quantize(..., mode="affine")`.
- `fastvideo/train/callbacks/mlx_qat.py` accepts **only `bits` and
  `group_size`**. There is no `mode` parameter.
- `fastvideo/tests/mlx/test_mlx_affine_qat_parity.py` tests **affine only** —
  every assertion passes `mode="affine"`.
- The only files mentioning mxfp4 at that commit are inference-side:
  `mlx_runtime/fastwan.py`, the benchmarks, the capability probe. **No
  training-side MX fake-quantizer exists in the tree.**

But the mxfp4 runs logged `mode: mxfp4` with no `bits`/`group_size`, a different
key set from the int8 run's `bits: 8, group_size: 64`. So they used a
**locally-modified, uncommitted QAT callback and MX quantizer** — code that is
unversioned, unreviewed, and has no parity test against
`mx.quantize(..., mode="mxfp4")`.

That single component is shared by both failed runs and absent from the working
one. It explains every observation at once: why both mxfp4 runs failed the same
way, why int8 did not, and why the student gradient norm was halved. A buggy
fake-quant or saturation-clipped STE produces exactly that signature.

**Find that code first.** It is on the training box under
`/raid/arkumar/FastVideo-apple-qad/`, not in git. Commit it, then write the
parity test — the MX sibling of `test_mlx_affine_qat_parity.py`, asserting
bitwise agreement with `mx.quantize(..., mode="mxfp4")` on codes, scales, and
the dequantized reconstruction. Until that test passes, no mxfp4 result means
anything.

### On EMA

v1's failure was EMA, and it is worth checking whether that recurred — but the
mechanism differs here. `mlx_qat.py` swaps the fake-quantized weight in only for
the duration of each `forward`, restoring the master immediately after, and the
EMA callback updates in `on_training_step_end`, outside forwards. So **EMA
tracks master weights, not quantized ones.** EMA-of-quantized-weights is not the
bug.

What remains real: validation computes `quantize(EMA(w))`, while the student was
trained so that `quantize(w_t)` is good at each step. Quantization is nearly
linear on a fine grid and sharply nonlinear on a coarse one, so
`quantize(EMA(w))` can diverge from what training optimized much more at mxfp4
than at int8 — a genuinely format-dependent effect.

Also note `decay: 0.98` in all three configs against the callback's documented
default of `0.9999`. With `generator_update_interval: 5` the EMA updates five
times per student change, so the effective window is ~10 student updates —
short enough that EMA should track raw closely.

**Free test: evaluate the existing mxfp4 checkpoint on raw (non-EMA) weights.**
Pure inference, no training. If raw looks better than EMA, that is the answer.

### Then, minutes, no GPUs

1. **Open the mxfp4 configs.** Confirm `gradient_accumulation_steps: 4` and that
   student/critic `init_from` points at the already-distilled FastWan, not base
   Wan. If either is missing, the mxfp4 runs used the known-bad recipe and the
   format was never fairly tested.
2. **Measure how far the student moved from init.** v2's own diagnostic. If the
   mxfp4 student moved ~0.2%, that is conclusively the v1 failure mode.

Partial evidence that grad-accum *was* carried over: v2's header notes ~4×
wall-time per step versus v1, and `step_time_sec` is 49.2 (int8 v2) against 51.2
(mxfp4) — close enough to suggest matched work per step. Not conclusive, since
fake-quant adds its own cost. Check the file rather than infer.

Only if both come back clean do the STE and grid-parity checks become the
leading hypotheses.

### Do not retrain from scratch

The v2 lesson runs directly against that instinct. Learning distillation
*through* a quantizer from base weights is what v1 did, and it failed. The fix
was to start from an already-distilled model and adapt it to the grid. Keep that
structure for mxfp4.

### Iterate on 1.3B, not 14B

~16 h per run on 8 GPUs against ~20 h on 32 for the 14B — roughly 5× cheaper per
experiment, on the model whose MLX runtime is best proven. Get mxfp4 working
there, then scale the recipe to 14B once it does.

## Assets — public, already registered

| Kind | ID |
|---|---|
| Dataset | `FastVideo/Wan-Syn_77x448x832_600k` (600k samples, 77×448×832, ~1 TB) |
| Validation | `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json` |
| Base recipe | `examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml` |
| MLX recipe | `dmd2_t2v_mlx_int8_v2.yaml` — the one that works; branch all new configs from this |
| Teacher 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Teacher 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Student init 1.3B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` |
| Student init 14B | `FastVideo/FastWan2.1-T2V-14B-480P-Diffusers` |
| Alternative base | `FastVideo/FastWan2.2-TI2V-5B-Diffusers` — newer and better than 2.1, but the 5B MLX port is not parity-green (`FIVE_B_MODEL_REPO = None`), so it adds runtime risk |

Recipe defaults worth knowing: 77 frames at 448×832, `num_latent_t: 20`,
`train_batch_size: 1`, lr 2e-6, betas `[0.0, 0.999]`, `max_train_steps: 4000`,
`generator_update_interval: 5`, MLX variants on the 3-step schedule
`[1000, 757, 522]`.

## Risks

- **14B int8 quality is unmeasured.** Low risk: int8 is proven at 1.3B, 8-bit is
  far more forgiving than 4-bit, and the runtime already handles 14B. But it is
  an assumption until the run lands, so do not commit to a date before it does.
- **14B int8 is ~15 GB of weights**, marginal on a 24 GB Mac even with sequenced
  residency and a raised wired limit. Position it as 32 GB+ and let mxfp4 claim
  24 GB later.
- **Do not start another mxfp4 run before the STE diagnostic.** 26 GPU-days per
  14B attempt is too expensive to spend against an unverified path twice.
