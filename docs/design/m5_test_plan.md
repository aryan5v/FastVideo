# M5 MacBook Test Plan

Everything below runs on a borrowed M5 in an afternoon, needs no GPU cluster
and no training, and answers whether 4-bit is viable on Metal — and if so,
which 4-bit format to distill against.

## What the literature actually says

Researched rather than assumed, because the format choice was previously made
on first principles alone.

**NVFP4 beats MXFP4 on quality, structurally.** MXFP4 uses blocks of 32 with an
E8M0 (power-of-two, no mantissa) shared scale. NVFP4 uses blocks of 16 with an
E4M3 scale plus a per-tensor FP32 global scale. Finer blocks and a scale with
mantissa bits fit each block better; replacing E4M3 scales with E8M0 is reported
as a noticeable quality drop at equal element format. Cost is ~4.5 bits/param
against MXFP4's ~4.25 — **6% more memory for meaningfully better
reconstruction**. ([Spheron](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/),
[ai.rs](https://ai.rs/ai-developer/int4-qat-mxfp4-nvfp4-quantization))

**Both run on M5 through MLX.** Apple benchmarked GPT-OSS-20B in native MXFP4
on M5, and Ollama ships NVFP4 on Apple Silicon — an M5 Max running
Qwen3.5-35B-A3B in NVFP4 went from 1,154 to 1,810 tok/s prefill and 58 to 112
decode. Note also that **int8 gets roughly 2× over fp16** on the M5 neural
accelerator, so int8 is not the slow option it would be on M1–M4.
([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5),
[Ollama](https://ollama.com/blog/mlx),
[9to5Mac](https://9to5mac.com/2025/11/20/apple-shows-how-much-faster-the-m5-runs-local-llms-compared-to-the-m4/))

**4-bit video diffusion is proven — but not by QAT alone.** SVDQuant absorbs
activation outliers into a low-rank branch rather than smoothing them, and
holds visual fidelity at 4 bits. Follow-on work applies it specifically to
**Wan2.2-I2V at W4A4**, reporting 59.3% peak memory reduction for **0.9% VBench
degradation**. The techniques that make it work are low-rank outlier
absorption, GPTQ on the main branch, and **timestep-wise clipping-ratio
search** — activation distributions shift strongly across the denoising
trajectory, so a single calibration is wrong everywhere.
([SVDQuant](https://arxiv.org/abs/2411.05007),
[Timestep-Aware SVDQuant-GPTQ for Wan2.2-I2V](https://arxiv.org/html/2605.27003v1),
[S²Q-VDiT](https://arxiv.org/pdf/2508.04016))

**For MXFP4 specifically, block rotation closes most of the gap.** MR-GPTQ
(ICLR 2026) applies block-wise Hadamard rotation before quantizing, spreading
outliers across channels within each block.
([Block Rotation is All You Need for MXFP4](https://arxiv.org/pdf/2511.04214))

### What this changes

**Target NVFP4, not MXFP4.** Better quality per bit, real M5 support, and —
decisively — **FastVideo already has committed NVFP4 QAT machinery**:
`nvfp4_qat_train_config.py`, `nvfp4_config.py`, and the
`fp4linear._LinearFWD4BWD16Fn` straight-through estimator. That is the piece
whose absence caused the mxfp4 runs to be untrustworthy. MXFP4 stays a
secondary target, worth revisiting with rotation if NVFP4 disappoints.

And 4-bit for a video DiT is not a coin flip — it works, but naive QAT is not
the published recipe. Expect to need timestep-aware calibration at minimum.

## The tests

### T0 — Does the existing FastWan-QAD checkpoint already work on MLX? (1 h)

**Run this first. If it works, most of the distillation plan below is moot for
the 1.3B.**

FastWan-QAD-1.3B is already QAT'd against NVFP4 — it is the RTX 5090 flagship,
1.8 s for a 5 s 480p clip. The reason it may transfer to Metal unchanged:

**The checkpoint is bf16, not FP4.** `convert_model_to_nvfp4()` in
`nvfp4_config.py` converts at *load time*, from bf16 weights, and only then
optionally purges the bf16 copies from GPU memory. So the shipped artifact is a
bf16 master that was trained to survive NVFP4 quantization. The MLX runtime does
the same thing with a different backend —
`mlx_dit_from_diffusers_safetensors(..., quantization="nvfp4")` quantizes at
load. Same weights, same format, different quantizer.

Both sides implement the same NVFP4 definition: E2M1 elements, 16-element
blocks, E4M3 block scales. FastVideo's CUDA path also applies a per-tensor
global scale of `(448 * 6) / amax` (`nvfp4_config.py:494`); whether MLX's
`mode="nvfp4"` uses the same convention is the one real unknown, and it is a
scale convention rather than a grid-structure difference.

**The measurement.** Run the survey on *both* checkpoints and compare the nvfp4
column:

```bash
python -m fastvideo.benchmarks.mlx_quant_survey \
    --checkpoint ~/models/FastWan-QAD-1.3B/transformer \
    --modes int8 nvfp4 mxfp4 --json-out survey_qad.json

python -m fastvideo.benchmarks.mlx_quant_survey \
    --checkpoint ~/models/FastWan2.1-T2V-1.3B-Diffusers/transformer \
    --modes int8 nvfp4 mxfp4 --json-out survey_plain.json
```

QAT'd weights should quantize *better* on the grid they were trained for. If
the QAD checkpoint's nvfp4 reconstruction error is markedly lower than the plain
distilled checkpoint's, **the QAT transferred and MLX's grid matches NVIDIA's**.
Then generate from it in MLX at `nvfp4` and compare against fp16.

**What will not transfer:** the "FP4 + FP4" in that flagship row is NVFP4
*linears* plus FP4 *attention* via the `attn_qat_infer` kernel, which hard-gates
on sm_120. MLX has no FP4 attention and will run attention in fp16. That is
strictly *less* quantization than the model was trained for, so it is benign —
output should be as good or better. But the 1.8 s figure will not transfer
either; that is Blackwell silicon with FP4 attention, and a Mac will be slower.

**If T0 works**, the 1.3B ships with **zero new distillation**, and the 14B path
becomes "run the existing, committed NVFP4 QAD recipe at 14B on CUDA" rather
than inventing an MLX-specific quantizer. That is a far better position than the
mxfp4 track.

### T1 — Mode support, throughput, and reconstruction error (30 min)

One script does all three:

```bash
python -m fastvideo.benchmarks.mlx_quant_survey \
    --checkpoint ~/models/FastWan2.1-T2V-1.3B-Diffusers/transformer \
    --modes int8 int4 mxfp8 mxfp4 nvfp4 \
    --json-out survey.json
```

**T1a — which modes run at all.** `mxfp4`/`nvfp4` need recent MLX.

**T1b — throughput at DiT shapes.** Median ms for a quantized matmul at
(8192, 1536). What to look for: 4-bit clearly faster than int8 means the
neural accelerators are engaged. Within noise of int8 means MLX is emulating —
the memory win survives, the speed argument does not.

**T1c — reconstruction error on the real checkpoint.** The highest-value number
in this document. Relative L2 per mode over actual FastWan weights, no training,
no inference. **This ranks the formats before a single GPU-hour is spent**, and
it is the test whose absence let ~31 GPU-days go into an unvalidated format.

Expected ordering if theory holds: `int8 < mxfp8 < nvfp4 < mxfp4 < int4`. If
nvfp4 lands close to int8, it is the format to distill against. If every 4-bit
mode is several times worse than int8, that is a real answer and the launch
grid is int8.

### T2 — Grid parity for the QAT reference (15 min)

```bash
pytest fastvideo/tests/mlx/test_mlx_mx_qat_parity.py -v
```

Pins `fastvideo/layers/quantization/mlx_mx_qat.py` bitwise against
`mx.quantize`. Note this covers **MX only** — if T1c says nvfp4 wins, the
equivalent NVFP4 reference and test are the next thing to write, and the
existing CUDA `nvfp4_config.py` quantizer is the starting point.

**No GPU time on any format before its parity test is green.** That rule is
what the previous runs violated.

### T3 — End-to-end PTQ quality, per mode (1–2 h)

Generate the same prompts and seeds through the existing 1.3B distilled model
at fp16, int8, nvfp4, and mxfp4, using the runtime's existing quantization
path. Compare visually and by MS-SSIM against the fp16 reference —
`fastvideo/benchmarks/mlx_fastwan_bench.py` already computes MS-SSIM.

This is PTQ, so it is a *lower bound* on what QAT achieves. A format that looks
close to int8 here will look better after distillation. A format that produces
garbage here is unlikely to be rescued by QAT alone.

### T4 — Peak memory per mode (30 min)

Record actual peak unified-memory use per mode for the 1.3B, and extrapolate to
14B. Confirms the arithmetic that motivates 4-bit at all: 14B should be ~15 GB
at int8 versus ~8 GB at nvfp4 — the difference between needing a 32 GB Mac and
running on a 24 GB one.

### T5 — Raw vs EMA on the existing mxfp4 checkpoint (20 min)

If the old mxfp4 checkpoint is on hand, generate from **raw** weights rather
than EMA. Validation computed `quantize(EMA(w))` while training optimized
`quantize(w_t)`; that gap is nearly linear at int8 and sharply nonlinear at
4 bits. If raw looks materially better, EMA handling is implicated and the
earlier result is further discounted.

## Decision table

| T0 result | Do this |
|---|---|
| QAD nvfp4 error ≪ plain nvfp4 error, output good | **Ship the 1.3B as-is.** No new distillation. Run the committed NVFP4 QAD recipe at 14B on CUDA for the quality tier. |
| QAD nvfp4 error ≈ plain, output good anyway | nvfp4 is forgiving enough here that PTQ suffices; still ship, but QAD at 14B is worth doing properly. |
| Output bad | MLX's nvfp4 convention differs from NVIDIA's — most likely the per-tensor global scale. Diff the two quantizers before concluding anything about the format. |

| T1c result | T1b result | Do this |
|---|---|---|
| nvfp4 near int8 | 4-bit clearly faster | Distill NVFP4. Best case — quality and speed both. |
| nvfp4 near int8 | 4-bit ~= int8 | Distill NVFP4 anyway. Half the memory is the point; speed is a bonus. |
| all 4-bit ≫ int8 error | either | Launch on int8. Revisit 4-bit with SVDQuant-style outlier absorption, not plain QAT. |
| mxfp4 ≫ nvfp4 | either | Confirms the structural argument; drop mxfp4, keep nvfp4. |

## What to send back

`survey.json` plus the T3 sample videos. That is enough to pick the launch
format and to size the distillation runs.
