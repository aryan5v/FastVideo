# Experiment: MLX block-scaled quant backends (MXFP8 / MXFP4 / NVFP4)

**Date:** 2026-07-10  
**Machine:** Apple Silicon (M4-class, no M5 Neural Accelerators required for this probe)  
**MLX:** 0.31.2  
**Code:** `fastvideo/mlx_runtime/quant_backends.py` + `fastvideo/tests/mlx/test_quant_backends.py`

## Motivation

FastVideo’s MLX DiT path currently quantizes weights as **INT8 affine, group size 64** at load.
Block-scaled low-precision formats (MXFP8, MXFP4, NVFP4) are candidates for large denoise
speedups once Apple’s M5 Neural Accelerators (Metal 4 TensorOps) can execute them natively.
This experiment only answers: *which formats does MLX 0.31.2 already expose, how many
bytes do they store per weight, and how much matmul reconstruction error do they add?*

## Method

- Fixed seed; random fp16 `w` (1024×1024) and `x` (8×1024).
- Reference: `x @ w.T` in fp16, error measured as relative Frobenius norm in fp32.
- Quantization / matmul: native `mx.quantize` + `mx.quantized_matmul` only.
- Bytes/weight: total `nbytes` of packed weight + scales (+ biases for affine) divided by
  number of original weight elements (measured, not hard-coded).

## Results (mlx 0.31.2, this machine)

| backend           | supported | bytes/weight | rel-error (Frobenius) |
|-------------------|-----------|--------------|------------------------|
| `affine_int8_g64` | **yes**   | 1.062500     | 5.39e-03               |
| `mxfp8`           | **yes**   | 1.031250     | 7.64e-02               |
| `mxfp4`           | **yes**   | 0.531250     | 1.22e-01               |
| `nvfp4`           | **yes**   | 0.562500     | 1.02e-01               |

All four backends probe-clean on MLX 0.31.2: both `mx.quantize` and
`mx.quantized_matmul` succeed for affine INT8 (g64) and for the three block-scaled modes
(MXFP8/MXFP4 defaults group size 32; NVFP4 defaults group size 16).

## Takeaway

On MLX **0.31.2 we can already use all four formats** (`affine_int8_g64`, `mxfp8`,
`mxfp4`, `nvfp4`) for correctness and memory probes — none require a newer MLX build on
this machine. Affine INT8 remains the quality baseline (~0.5% rel-error); MXFP8 is ~15×
noisier but nearly the same footprint as INT8; MXFP4/NVFP4 halve storage at ~10–12%
rel-error. Hardware acceleration of the block-scaled paths is an M5 / Metal 4 TensorOps
question separate from API availability; until that lands, treat MX/NV modes as
portable numerical experiments rather than assumed speedups on M4.
