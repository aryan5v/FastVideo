---
date: 2026-07-08
experiment: M4 Phase A numerics gate (docs/design/apple_silicon_program_plan.md, step 1)
category: infrastructure
severity: important
---

# MLX affine-quant parity: bit-pin the CPU stream, tolerance-pin Metal

## What Happened
On Day-1 toolchain validation (M4 Max, MLX 0.31.2), 4 of the QAT numerics-gate
tests in `test_mlx_affine_qat_parity.py` failed with a `np.testing.assert_array_equal`
mismatch of ~1.2e-4. The tests bit-pin the torch fake-quant twin against
`mx.quantize`/`mx.dequantize`, which is *the* gate that must be green before any
GPU is spent on a QAT run. The test docstring claimed it "runs on any MLX
backend (Metal or mlx[cpu])" — that claim was false.

## Root Cause
MLX's CPU and Metal kernels do not agree bit-for-bit. The Metal kernel
accumulates the affine math — `(w - bias) / scale` on quantize and
`code * scale + bias` on dequantize — in **fp32**, while the CPU kernel (which
the twin transcribes) stays in the input **fp16**. Consequences on Metal for
fp16 inputs:
- a vanishing fraction of boundary weights round to a neighbouring code:
  measured **0.0147%** over 156k elements, all **±1 LSB**;
- the dequantized reconstruction drifts by **~1.2e-4** on code-agreeing
  elements.
fp32 inputs are exact to ~1e-8 on both backends. Probing the same math with
`mx.stream(mx.cpu)` gave **0** mismatches everywhere, confirming the twin is a
faithful transcription of MLX's *CPU* kernel, not the Metal one.

## Fix / Workaround
Split the gate into two assertions (the program plan already sanctioned
"pinned bitwise OR tolerance-pinned against Metal"):
- **(a) Decisions, bit-pinned on the CPU stream.** Codes/scales/biases compared
  under `mx.stream(mx.cpu)` — deterministic, machine-independent, still catches
  real bugs (rounding rule, zero-point anchoring, group reduction).
- **(b) Deploy reconstruction, tolerance-pinned on the Metal stream.** Code-flip
  rate and ±1-LSB bound, accumulation drift, and `mx.quantized_matmul` vs the
  twin (measured 1.1e-3 vs a 2e-2 deploy tolerance, 18×), all with recorded
  headroom. Metal-only tests are guarded with
  `skipif(not mx.metal.is_available())` so the Linux `mlx[cpu]` CI job stays
  green on the CPU-bitwise half.

## Prevention
- This is the **house pattern** for every deploy-time quantizer/kernel twin:
  bit-pin the decisions to a CPU/reference spec, tolerance-check the Metal
  deploy path with measured headroom. Track A's attention-QAT should reuse it.
- Never bit-pin a gate to a GPU/Metal kernel's floating-point accumulation —
  it is non-deterministic across hardware and MLX versions. Pin the spec on CPU.
- Mac install cannot use `uv pip install -e '.[dev,mlx]'` (core dep
  `fastvideo-kernel` → `triton` has no macOS/arm64 wheels); use the lightweight
  recipe in `.github/workflows/ci-macos-mlx.yml`. Keep MLX pinned to 0.31.2.
