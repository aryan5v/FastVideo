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

## Hardware tiers

FastVideo auto-selects a practical MLX model + quant + allocator cap from the
Mac's unified-memory size via `fastvideo/mlx_runtime/hardware_tier.py`. The
goal is the north-star product behaviour: a 16 GB Mac gets a small INT8 model
that fits; 32/64 GB Macs get higher fidelity without the user hand-picking
flags.

| Unified memory | Tier | Model (today) | Quant | Decoder | MLX cap | Benchmark preset |
| --- | --- | --- | --- | --- | ---: | --- |
| ≤ 18 GiB | `small` | FastWan 1.3B | int8 | TAEHV | 12 GiB | `mac-16gb` |
| ≤ 40 GiB | `medium` | 1.3B fp16 *(5B int8 when Track D lands)* | none / int8 | TAEHV | 24 GiB | `mac-32gb` |
| > 40 GiB | `large` | 1.3B fp16 *(5B fp16 when Track D lands)* | none | Wan-VAE | 48 GiB | `mac-64gb` |

Thresholds (`TIER_SMALL_MAX_GIB = 18`, `TIER_MEDIUM_MAX_GIB = 40`) and caps are
module constants so they are easy to retune. Detection order:

1. macOS `sysctl -n hw.memsize` (true unified memory)
2. MLX `device_info()["memory_size"]` when Metal is available
3. Linux `/proc/meminfo` MemTotal
4. Safe default **16 GiB** → `small` tier (non-Mac / undetectable)

**Track D / 5B:** `FIVE_B_MODEL_REPO` is `None` until the 5B MLX port is
parity-green. Medium/large tiers fall back to 1.3B fp16 while that constant is
unset. Set the repo id and keep `prefer_5b=True` (the default) to switch.

### `--auto-tier` on the benchmark

```bash
# Detect this Mac's memory and apply recommended modes + MLX cap.
PYTHONPATH=$PWD python -m fastvideo.benchmarks.mlx_fastwan_bench \
  --auto-tier \
  --prompt "A fox runs through a mossy forest." \
  --output-dir video_samples/mlx_fastwan_bench_auto_tier
```

What `--auto-tier` does:

- Calls `recommend_tier(prefer_5b=...)` (override with `--no-prefer-5b`).
- Seeds height/width/frames from the matching `mac-16gb` / `mac-32gb` /
  `mac-64gb` preset.
- Forces `modes`, `decoders`, `mlx_memory_limit_gib`, and `mlx_disable_cache`
  from the tier (printed as `[auto-tier] ...` and recorded in `metrics.json`).
- Does **not** rewrite `--model-root` (local path vs HF id); the recommended
  HF repo is in metrics as `auto_tier_model_repo`.

Static presets remain available without detection:

| Preset | Shape | Modes | Decoders | MLX cap |
| --- | --- | --- | --- | ---: |
| `mac-16gb` | 448×832×61 | int8 | taehv | 16 GiB (stress) |
| `mac-32gb` | 480×832×81 | int8,fp16 | taehv | 24 GiB |
| `mac-64gb` | 480×832×81 | int8,fp16 | taehv,wan-vae | 48 GiB |

Note the 16 GB *stress* preset still pins a 16 GiB allocator cap to exercise
tight memory; the auto-tier *product* recommendation for ≤18 GiB uses a 12 GiB
cap to leave headroom for the OS and torch/MPS encode/decode.

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

**`mx.compile` A/B (`--compile --assert-min-ssim 0.9`): ~1.4× denoise speedup.**
`mx.compile` on the DiT forward was initially broken — it raised `Attempting to
eval an array during function transformations like compile or vmap is not
allowed` (caught at `fastwan.py`, eager fallback, no speedup) or **segfaulted
(exit 139)**. Root cause: a **NumPy scalar multiplying a traced array** in
`gelu_tanh` (`np.sqrt(2/pi) * x`) dispatched through NumPy's `__mul__`, which
evals the traced array — illegal under compile, and the source of the segfault
too. Fixed by using a Python-`float` constant (`fastwan.py`), guarded by
`fastvideo/tests/mlx/test_mlx_compile_parity.py`. With the fix, compile traces
cleanly (no fallback) and is bit-identical to eager:

| mode | steady step (eager → compiled) | speedup | MS-SSIM vs fp16 |
| --- | --- | ---: | ---: |
| fp16 | 4.48s → 3.17s | 1.41× | 1.000 (ref) |
| int8 | 4.62s → 3.22s | 1.43× | 0.975 |

The SSIM gate (`--assert-min-ssim 0.9`) passes on the compiled path. Lesson: a
NumPy scalar times a traced `mx.array` silently breaks `mx.compile` — keep such
constants as Python floats.

## Causal streaming (Track C, 2026-07-08, M4 Max, MLX 0.31.2)

Block-autoregressive streaming of `wlsaidhi/SFWan2.1-T2V-1.3B` via
`MLXCausalWanDiT` + `stream_causal_latents` (4-step DMD per block,
`num_frames_per_block=1`, 480×832 → 1560 tokens/frame). Latency is denoise-side
(decode not included — VAE/TAEHV weights not required for these numbers):

| mode | load | time-to-first-frame | steady per-block | peak |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 5.2s | 2.70s | 3.30s | 9.34 GiB |
| INT8 | 2.0s | 2.20s | 3.34s | 8.19 GiB |

INT8 (affine group-64, via the same `quantize_matrix` path as the dense port)
cuts time-to-first-frame and peak memory; steady per-block latency is comparable
and grows slowly as the attention window fills (2.7s→3.8s over 6 blocks). These
are latency/plumbing numbers on the verified causal forward — visual quality vs
the CUDA SF reference is a separate eyeball check (a DGX ask). Reproduce with
`examples/inference/basic/mlx_wan_streaming.py`.

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
