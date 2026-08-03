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

## Risks

- **14B int8 quality is unmeasured.** Low risk: int8 is proven at 1.3B, 8-bit is
  far more forgiving than 4-bit, and the runtime already handles 14B. But it is
  an assumption until the run lands, so do not commit to a date before it does.
- **14B int8 is ~15 GB of weights**, marginal on a 24 GB Mac even with sequenced
  residency and a raised wired limit. Position it as 32 GB+ and let mxfp4 claim
  24 GB later.
- **Do not start another mxfp4 run before the STE diagnostic.** 26 GPU-days per
  14B attempt is too expensive to spend against an unverified path twice.
