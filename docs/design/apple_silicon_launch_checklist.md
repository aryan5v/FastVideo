# Apple Silicon Launch — Action Plan

Written fresh from what the runs actually showed. Supersedes the quantization
strategy in `apple_silicon_minimax_h3.md`; that document remains the reference
for the H3 track, which is not on this critical path.

**Ship int8. mxfp4 is a follow-up, and it has never actually been tested.**

## What we know

| | |
|---|---|
| 1.3B int8 QAD | **works, quality accepted** — shippable now |
| 14B int8 QAD | **never run** — the only missing launch artifact |
| 1.3B / 14B mxfp4 QAD | both bad, but both used an **uncommitted, untested** quantizer |
| MLX runtime at 14B | proven — the mxfp4 runs loaded and ran |
| Working recipe | `dmd2_t2v_mlx_int8_v2.yaml` |
| Measured cost | 14B QAD = 26 GPU-days; 1.3B QAD = 5.3 GPU-days |
| Available | up to 36× B200 |

Two things carry most of the weight.

**The recipe that works is v2, and v2 exists because v1 failed.** v1 learned
distillation from scratch through a quantizer and had a broken EMA checkpoint.
v2 fixed both: initialize student and critic from the already-distilled FastWan,
raise effective batch to 16, use the world-size-portable EMA state. Every new
config branches from v2.

**The mxfp4 runs were never a fair test of mxfp4.** At run commit `411cfa7e`
the tree contains no training-side MX fake-quantizer at all — `mlx_affine_qat.py`
is affine-only, `mlx_qat.py` takes only `bits`/`group_size`, and the parity test
asserts `mode="affine"` throughout. The runs logged `mode: mxfp4`, so they ran on
local uncommitted code with no parity test. That component is shared by both
failures and absent from the working run, and it accounts for the identical
failure mode across sizes and the halved student gradient norm.

## Day 1

**1. Ship 1.3B int8.** Done, accepted, don't hold it.

**2. Launch the 14B int8 run.** Config is written and ready:

```bash
NUM_GPUS=32 bash examples/train/run.sh \
  examples/train/configs/distribution_matching/wan/dmd2_t2v_14b_mlx_int8.yaml
```

~20 h on 32 GPUs. It is a mechanical scale-up of v2 — same recipe, same
committed and parity-tested int8 quantizer, 14B checkpoints, 4×8 HSDP mesh.
This is the whole critical path; start it before anything else.

**3. In parallel — no GPUs, all of it:**

- **14B int8 PTQ.** Quantize the existing 14B weights to affine int8 and look.
  A few hours. If it holds, 14B ships without waiting for the run above.
- **Recover the mxfp4 quantizer** from `/raid/arkumar/FastVideo-apple-qad/` and
  commit it. It is not in git.
- **Write the MX parity test** — the sibling of
  `test_mlx_affine_qat_parity.py`, asserting bitwise agreement with
  `mx.quantize(..., mode="mxfp4")` on codes, scales, and dequantized output.
  **Until this passes, no mxfp4 result means anything.**
- **Evaluate the existing mxfp4 checkpoint on raw, non-EMA weights.** Pure
  inference. Validation computes `quantize(EMA(w))` while training optimized
  `quantize(w_t)`; that gap is nearly linear at int8 and sharply nonlinear at
  4-bit. If raw beats EMA, that is the answer.

## Day 2–3

- 14B int8 lands → validate on Mac → headline model.
- MX parity test reports. If the quantizer is wrong, the two failed runs are
  void and mxfp4 is untested rather than disproven.
- Only then, one corrected 1.3B mxfp4 run: ~16 h on 8 GPUs. Iterate at 1.3B, not
  14B — 5× cheaper per experiment, on the best-proven runtime.

## Launch shape

| Tier | Model | Weights | Mac |
|---|---|---|---|
| Fast | 1.3B int8 | ~1.4 GB | 16 GB+, any chip |
| Quality | 14B int8 | ~15 GB | 32 GB+ |
| Follow-up | 14B mxfp4 | ~7.4 GB | 24 GB, M5+ |

The 1.3B → 14B jump is what users notice. mxfp4 buys reach — 14B from a 32 GB
machine down to a 24 GB one — not better output.

## Rules

- **Every new config branches from `dmd2_t2v_mlx_int8_v2.yaml`.** Never from v1,
  never from scratch. Adapting an already-distilled model to a grid is the thing
  that works; learning distillation through a quantizer is the thing that failed.
- **No more mxfp4 GPU time until the parity test passes.** 26 GPU-days per 14B
  attempt is too expensive to spend twice against unverified code.
- **Debug at 1.3B, ship at 14B.**

## Assets

| Kind | ID |
|---|---|
| Dataset | `FastVideo/Wan-Syn_77x448x832_600k` (600k, 77×448×832, ~1 TB) |
| Validation | `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json` |
| Teacher 1.3B / 14B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` / `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Student init 1.3B / 14B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` / `FastVideo/FastWan2.1-T2V-14B-480P-Diffusers` |

## Risks

- **14B int8 quality is unmeasured.** Low risk — int8 is proven at 1.3B, 8-bit
  is far more forgiving than 4-bit, and the runtime already handles 14B. But it
  is an assumption until the run lands; don't commit to a date before it does.
- **14B int8 is ~15 GB of weights**, marginal on a 24 GB Mac even with sequenced
  residency and a raised wired limit. Position it as 32 GB+; let mxfp4 claim
  24 GB later.
- **The 4×8 HSDP mesh at 14B is untried here.** v2 ran 4-GPU pure FSDP. If the
  student, frozen teacher, and critic do not fit at `hsdp_shard_dim: 8`, raise
  the shard dim and lower replication.

## Not on this path

The H3 CUDA port is blocked on Hugging Face egress from the porting environment
(`tests/local_tests/minimax_h3/PORT_STATUS.md`, I001). It proceeds independently
and gates nothing here.
