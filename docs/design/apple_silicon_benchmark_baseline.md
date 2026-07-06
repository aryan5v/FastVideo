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
