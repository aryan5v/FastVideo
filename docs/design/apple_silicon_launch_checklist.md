# Apple Silicon Launch — MXFP4 Plan

Make mxfp4 work, distill on it, harden the runtime, launch. Supersedes the
quantization strategy in `apple_silicon_minimax_h3.md`, which remains the
reference for the H3 track (not on this path).

## Why mxfp4 is worth fixing rather than abandoning

mxfp4 puts a 14B model in ~7.4 GB against ~15 GB at int8. That is the
difference between 14B needing a 32 GB Mac and running comfortably on a 24 GB
one, and int8 cannot get there. On M5 the format also has a hardware path
through the GPU neural accelerators that affine int8 does not.

**And it has never actually been tested.** Both prior mxfp4 runs used an
uncommitted fake-quantizer with no parity test against the deploy grid. The
matched 1.3B control is the tell: at *lower* loss, the mxfp4 run showed **0.47×
the student gradient norm** of the int8 run. A model converged somewhere worse
shows comparable gradients and higher loss. Low loss with halved gradients is a
student that is barely being optimized — a broken QAT path, not a bad format.

## Stage 1 — Make the grid real (no GPUs)

**1a. The MX quantizer is now committed.**
`fastvideo/layers/quantization/mlx_mx_qat.py` — the MX sibling of
`mlx_affine_qat.py`, covering mxfp4 and mxfp8. Block size 32, E8M0 shared
scale, E2M1/E4M3 element grids, round-half-to-even on the grid, and an
**unclipped** STE.

> One caveat stated plainly: `mlx_affine_qat` was transcribed from MLX's CPU
> kernel, so it agrees with MLX structurally. `mlx_mx_qat` is derived from the
> OCP Microscaling spec, so it agrees *only if MLX also follows the spec*. That
> is what 1b exists to prove.

**1b. Run the parity test on the Mac.**

```bash
pytest fastvideo/tests/mlx/test_mlx_mx_qat_parity.py -v
```

`fastvideo/tests/mlx/test_mlx_mx_qat_parity.py` pins the round-trip against
`mx.quantize(..., mode="mxfp4"/"mxfp8")` bitwise, covers all-zero blocks,
power-of-two straddles, and single-outlier blocks, and asserts the STE passes
gradients even at grid saturation. Likely divergence points, if it fails, are
listed in the module docstring — shared-exponent rule first, rounding mode
second.

**This is the gate. No GPU time until it is green.**

**1c. Teach the callback about `mode`.** `mlx_qat.py` lives on the MLX branch,
so apply this there rather than here:

```python
# fastvideo/train/callbacks/mlx_qat.py
from fastvideo.layers.quantization.mlx_mx_qat import MX_BLOCK_SIZE, fake_quantize_mlx_mx

# __init__: accept mode: str | None = None; when set, ignore bits/group_size
# and use MX_BLOCK_SIZE for the _is_target divisibility check.

def _fake_quantize_weight(weight, *, mode, group_size, bits, simulate_dtype):
    original_shape = weight.shape
    weight2d = weight.reshape(original_shape[0], -1) if weight.dim() > 2 else weight
    if mode is not None:
        fq = fake_quantize_mlx_mx(weight2d, mode=mode, simulate_dtype=simulate_dtype)
    else:
        fq = fake_quantize_mlx_affine(weight2d, group_size=group_size, bits=bits,
                                      simulate_dtype=simulate_dtype)
    return fq.reshape(original_shape).to(weight.dtype)
```

Keep `DEFAULT_EXCLUDE_PATTERNS = (r"norm", r"scale_shift_table")` — norms and
modulation tables must stay out of the quantized set on any grid.

**1d. Free diagnostics on the existing checkpoints.**

- Evaluate the old mxfp4 checkpoint on **raw, non-EMA** weights. Validation
  computes `quantize(EMA(w))` while training optimized `quantize(w_t)`; that
  gap is nearly linear at int8 and sharply nonlinear at 4-bit.
- Log per-layer **grid saturation** and **STE gradient passthrough** on both
  paths. If the old quantizer clipped, this shows it directly and confirms the
  diagnosis.

## Stage 2 — One controlled 1.3B run

```bash
NUM_GPUS=8 bash examples/train/run.sh \
  examples/train/configs/distribution_matching/wan/dmd2_t2v_1p3b_mlx_mxfp4.yaml
```

~16 h on 8 GPUs. `dmd2_t2v_1p3b_mlx_mxfp4.yaml` is `int8_v2` with **exactly one
change** — the QAT grid — so the int8 run is a controlled baseline and any
delta is attributable to the format.

Watch `grad_norm/student` against the int8 baseline of **1.332**. If it is
still near 0.6, the quantizer is not the whole story and Stage 1d's saturation
numbers are the next thread. If it tracks the baseline, the grid is working.

Iterate here, not at 14B: 5.3 GPU-days against 26.

## Stage 3 — Distill the launch models

Once 1.3B mxfp4 matches int8 quality, scale the same config to 14B.
`dmd2_t2v_14b_mlx_int8.yaml` is already written and is the template — swap the
`mlx_qat` block to `mode: mxfp4`.

| Run | GPUs | Wall-clock |
|---|---|---|
| 14B mxfp4 | 32 | ~20 h |
| 1.3B mxfp4 | 8 | ~16 h |
| int8 fallback, either size | — | one extra run each, for pre-M5 Macs |

With 36 GPUs available the 1.3B and 14B runs go concurrently.

## Stage 4 — Harden the runtime

- MX deploy path in `mlx_runtime/`: `MLXQuantizationSpec.from_name` already
  knows `mxfp4`/`mxfp8`; verify the checkpoint format in `checkpoint.py`
  round-trips MX scales, and make the loader **refuse a grid mismatch loudly**
  rather than silently dequantizing.
- **E1 throughput benchmark on M5.** Add mxfp4/mxfp8 arms to
  `benchmark_mlx_linear` and `benchmark_mlx_attention` at realistic shapes.
  mxfp4 should be *substantially* faster than int8; within noise means MLX is
  emulating rather than dispatching to the neural accelerators. Worth knowing
  before launch, though the memory win stands either way.
- `hardware_tier.py` gains a generation axis: memory band picks the model, chip
  generation picks the grid.
- Sequenced residency: encode → free encoder → denoise → decode, so peak is a
  max rather than a sum.

## Launch shape

| Tier | Model | Weights | Mac |
|---|---|---|---|
| Fast | 1.3B mxfp4 | ~0.8 GB | 16 GB+, M5+ |
| Quality | 14B mxfp4 | ~7.4 GB | 24 GB+, M5+ |
| Compatibility | either at int8 | 1.4 / 15 GB | pre-M5 |

## Rules

- **Every config branches from `dmd2_t2v_mlx_int8_v2.yaml`.** Never v1, never
  from scratch. Adapting an already-distilled model to a grid works; learning
  distillation *through* a quantizer is what failed as v1.
- **Change one variable per run.** The reason the current evidence is
  ambiguous is that the mxfp4 runs differed from the working int8 run in more
  than the grid.
- **No mxfp4 GPU time before the parity test passes.**
- **Debug at 1.3B, ship at 14B.**

## Assets

| Kind | ID |
|---|---|
| Dataset | `FastVideo/Wan-Syn_77x448x832_600k` (600k, 77×448×832, ~1 TB) |
| Validation | `examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json` |
| Teacher 1.3B / 14B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` / `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Student init 1.3B / 14B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` / `FastVideo/FastWan2.1-T2V-14B-480P-Diffusers` |

## Risks

- **The MX reference may not match MLX.** It is spec-derived, not transcribed.
  1b catches this; the fix is reading MLX's kernel and adjusting, which is
  hours, not a redesign.
- **mxfp4 may genuinely be too coarse for a video DiT**, even with a correct
  grid and healthy STE. The 1.3B run in Stage 2 is what settles it, at 5.3
  GPU-days. If it fails there with `grad_norm/student` tracking the int8
  baseline, that is a real answer and int8 is the launch grid.
- **1.3B is the hardest case for 4-bit** — least redundancy of the two sizes.
  A failure at 1.3B does not strictly rule out 14B, but it is the cheap
  experiment, so run it first and treat a 14B retry as a deliberate second bet.

## Not on this path

The H3 CUDA port is blocked on Hugging Face egress
(`tests/local_tests/minimax_h3/PORT_STATUS.md`, I001) and gates nothing here.
