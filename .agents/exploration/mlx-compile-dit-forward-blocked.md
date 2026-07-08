# Exploration Log: mx.compile on the FastWan DiT forward

## Status: under_review

## Context
Day-1 quick win from `docs/design/apple_silicon_program_plan.md` step 1: run the
`mx.compile` A/B (`--compile --assert-min-ssim 0.9`) and record the number in
`apple_silicon_benchmark_baseline.md`. The roadmap lists benchmarked `mx.compile`
of the DiT forward as one of the next runtime wins. Measured on M4 Max, MLX
0.31.2, 480×832×17 / 3-step DMD, TAEHV decode.

## Progress
- [x] Establish eager baseline with SSIM gate green (fp16 4.48 s/step, int8
      4.62 s/step, int8-vs-fp16 MS-SSIM 0.974).
- [x] Attempt `--compile` A/B.
- [ ] Root-cause and remove the eval/graph-break inside the traced `_forward`.
- [ ] Root-cause the Metal segfault (may be an MLX 0.31.2 bug independent of the
      eval break).
- [ ] Re-run the A/B and record the real speedup.

## Findings
`mx.compile` is currently **unusable** on `FastWanTransformer._forward`
(`fastvideo/mlx_runtime/fastwan.py:659`). Two failure modes observed across runs:

1. **Catchable:** `Attempting to eval an array during function transformations
   like compile or vmap is not allowed` — a `mx.eval` (or eval-forcing op like
   a Python `float(...)`/`.item()` on a traced array) executes inside the traced
   forward. The wrapper at `fastwan.py:673-686` catches this and falls back to
   eager, so there is **no speedup**, just a warning.
2. **Uncatchable:** the process **segfaults (exit 139)** during the compile
   attempt on the Metal backend, before any cell output. The `try/except
   Exception` cannot catch a native SIGSEGV, so `--compile` can take down the
   whole benchmark run.

Because of this, there is no positive `mx.compile` number to record; the eager
baseline above is what a working compile path must beat. Recorded in the
baseline doc's "Day-1 runtime measurements" section.

## Mistakes / Dead Ends
- The lightweight Mac install (per `ci-macos-mlx.yml`) omits `transformers`,
  `imageio-ffmpeg`, `pytorch-msssim`, and `av`, all of which the *benchmark*
  (not the smoke tests) needs for prompt encode, MP4 export, and the SSIM gate.
  Install them before running `mlx_fastwan_bench`.
- The benchmark must be run as a module from the repo root
  (`python -m fastvideo.benchmarks.mlx_fastwan_bench`, `PYTHONPATH=<repo>`); the
  bare `python fastvideo/benchmarks/...` invocation fails to import `examples`.

## Proposed Standardization
Once a working compile path exists: promote a short SOP for the `mx.compile`
A/B (env deps, module invocation, off/on run pair, `denoise_steady_step_s`
comparison, SSIM gate). Until then this stays a blocker note for review.
