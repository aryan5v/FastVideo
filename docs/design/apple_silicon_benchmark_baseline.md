# Apple Silicon MLX Benchmark Baseline

This page records the current Apple Silicon FastWan MLX baseline so runtime
changes can be compared against a stable reference. It is intentionally honest:
these numbers are early proof-of-concept measurements, not final product claims.

## Hardware and runtime

- Machine: Apple M4 Max
- Unified memory: 36 GB class (`memory_size` reported by MLX: 38.65 GB)
- MLX: 0.31.2
- Model shape tested: 480×832, 81 frames, 3-step DMD
- Runtime path: torch/MPS prompt encode → MLX DiT denoise → TAEHV or Wan VAE decode

## Current measured baseline

| Tier / config | Mode | Decoder | Result | Total | Denoise | Decode | MLX peak |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 36 GB class | INT8 | TAEHV | pass | 123.7s | 117.6s | 2.1s | 5.63 GiB |
| 16 GiB MLX cap | INT8 | TAEHV | pass | 132.1s | 126.0s | 1.9s | 4.69 GiB |
| 36 GB class | FP16 | Wan VAE | pass | 240.3s | 114.1s | 122.2s | 6.91 GiB |

The 16 GiB row uses MLX allocator caps and PyTorch MPS watermarks. This is a
useful stress test for the DiT/runtime path, but it is not a substitute for
testing on an actual 16 GB Mac because macOS does not expose a perfect
process-wide unified-memory simulator.

## Commands

The benchmark harness now exposes memory-tier flags and presets. A fast smoke
test for the 16 GB tier can be run with:

```bash
python fastvideo/benchmarks/mlx_fastwan_bench.py \
  --benchmark-preset mac-16gb \
  --prompt "A fox runs through a mossy forest." \
  --output-dir video_samples/mlx_fastwan_bench_mac16_smoke
```

The standard motion prompt sweep can be run with:

```bash
python fastvideo/benchmarks/mlx_fastwan_bench.py \
  --benchmark-preset mac-16gb \
  --prompt-set motion7 \
  --output-dir video_samples/mlx_fastwan_bench_motion7_mac16
```

Each run writes:

- `metrics.json`
- `metrics.md`
- `index.html` with side-by-side videos and synchronized play/pause controls
- generated MP4s grouped by prompt id

## What this proves

- The MLX DiT path is technically feasible at the 5-second 480p-class shape.
- INT8 + TAEHV is currently the practical fast path.
- Full Wan VAE decode works, but it is not yet the speed path on Mac.
- The next runtime wins should come from benchmarked `mx.compile`, fused norms,
  fewer host/device transfers, and eventually pre-quantized MLX checkpoint
  loading.

## What it does not prove yet

- It does not prove final visual quality.
- It does not prove true 16 GB Mac behavior until tested on stock 16 GB
  hardware.
- It does not prove long-video coherence or 60-second generation.
- It does not replace the Mac-targeted QAT/distillation track.

## Toolchain and the M4 Phase A numerics gate (2026-07)

Mac install does **not** use `uv pip install -e '.[dev,mlx]'`: the core package
pulls `fastvideo-kernel` → `triton`, which has no macOS/arm64 wheels. Follow the
lightweight recipe in `.github/workflows/ci-macos-mlx.yml` instead (CPU torch
wheels + a short dependency list + `mlx`), importing `fastvideo` from the source
tree. MLX is pinned to **0.31.2** per the program's "pin the MLX version"
decision.

The QAT numerics gate (`fastvideo/tests/mlx/test_mlx_affine_qat_parity.py`) is
now two assertions, because MLX's CPU and Metal kernels are not bit-identical —
the Metal kernel accumulates the affine math in fp32, the CPU kernel in the
input fp16:

- **Decisions, bit-pinned on the CPU stream.** Codes/scales/biases match MLX's
  CPU kernel exactly — the deterministic, machine-independent spec of the
  quantizer.
- **Deploy reconstruction, tolerance-pinned on the Metal stream** (with recorded
  headroom): code-flip rate **0.0147%** (all ±1 LSB), dequant accumulation drift
  **~1.2e-4**, `mx.quantized_matmul` vs the twin **1.1e-3** against a 2e-2 deploy
  tolerance (**18× headroom**). All fp32-input paths are exact to ~1e-8.

This is the house pattern for every deploy-time quantizer/kernel twin (Track A's
attention-QAT will reuse it): bit-pin the decisions to a CPU/reference spec,
tolerance-check the Metal deploy path.

## Day-1 runtime measurements (2026-07-08, M4 Max, MLX 0.31.2)

Two roadmap measurements, run at a fast 480×832×**17**-frame / 3-step DMD shape
(these are relative deltas and a status check, not the 81-frame product shape):

**Checkpoint-cache load-time delta** (`--mlx-checkpoint-cache`):

| Load path | `load_source` | `load_s` | `load_peak` |
| --- | --- | ---: | ---: |
| cold (convert Diffusers fp32 → fp16 → `mx.quantize` int8, save) | `diffusers_then_saved` | 4.63s | 1.45 GiB |
| warm (reload pre-quantized MLX checkpoint) | `mlx_checkpoint` | 0.006s | 0.011 GiB |

The warm path skips the safetensors read, fp16 cast, and int8 requantization,
saving **~4.62s per load**; it is shape-independent. (The 0.006s is MLX's lazy
load creating array handles — data materializes during denoise — so the real
saved work is the requantization, not a literal 800× I/O win.)

**`mx.compile` A/B (`--compile --assert-min-ssim 0.9`): currently BLOCKED.**
On the FastWan DiT forward with MLX 0.31.2 on Metal, `mx.compile` either raises
`Attempting to eval an array during function transformations like compile or
vmap is not allowed` and falls back to eager (`fastwan.py:673-686` handles this
— no speedup), or it **segfaults the process (exit 139)**, which is uncatchable.
So there is no compile speedup to record yet; unblocking it (removing the
eval/graph-break inside the traced `_forward`, and the Metal segfault) is a
runtime task on the roadmap's `mx.compile` line. Eager baseline the future work
must beat (steady step / MS-SSIM vs own fp16):

| mode | steady step | first step | MS-SSIM vs fp16 |
| --- | ---: | ---: | ---: |
| fp16 | 4.48s | 4.60s | 1.000 (ref) |
| int8 | 4.62s | 4.70s | 0.974 |

The SSIM gate (`--assert-min-ssim 0.9`) passes: int8 stays at 0.974 vs its own
fp16 at this shape.

## QAD-INT8 evaluation (M4 exit measurement, 2026-07)

Three runs on the motion7 prompt set (480×832×81, 3-step DMD, TAEHV decode,
shared seed; INT8 = affine group-64 quantized at MLX load). `MS-SSIM` scores
each run's INT8 cells against that run's own FP16 cells, i.e. it isolates
quantization damage per model:

| model | INT8 mean MS-SSIM vs own FP16 | INT8 steady step | INT8 peak GiB |
| --- | ---: | ---: | ---: |
| stock FastWan2.1-1.3B (PTQ) | 0.9069 | 34.2s | 5.62 |
| QAD raw student | 0.9487 | 35.4s | 5.63 |
| QAD EMA student | **0.9860** | 35.4s | 5.63 |

Per-prompt, QAD EMA beats stock PTQ on all seven prompts (stock ranges
0.821–0.974; EMA sits at 0.986 ± 0.0003). Runtime cost of QAT is zero, as
expected — identical architecture, so step time and peak memory match stock.

**Correction after visual review:** the EMA row is void. The EMA export's
outputs are noise — and noise quantizes uniformly, which is precisely why its
INT8-vs-FP16 SSIM was a suspiciously constant 0.986. Within-model SSIM
measures quantization *consistency*, not absolute quality; it cannot detect a
broken model. Root cause under investigation: the EMA checkpoint state is
stored as world-size-dependent local shards (`EMA_FSDP` `local_shard` mode)
from a 4-GPU run, while `dcp_to_diffusers --ema` forces a 1-GPU world.
Training-time validation clips (which swap EMA weights via `ema_context`)
looked good, so the EMA weights themselves were healthy — the export path is
the suspect.

The **raw student** row stands: 0.9487 vs stock PTQ's 0.9069 passes the
quantization-robustness criterion, but visual review found motion defects and
overall quality below stock — so the parity criterion is not met and the
run-2 levers (FastWan init, larger effective batch) are activated per the
roadmap's decision tree.
