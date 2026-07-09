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

## Wan2.2-TI2V-5B (Track D Rung 3)

Real-weight MLX port of `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`
(30 layers, 24×128 heads, hidden 3072, ffn 14336, VAE z_dim=48 / DiT
`in_channels=48`). Per-token timestep conditioning
(`expand_timesteps`: frame-0 tokens at t=0, remaining at the denoise level).

### Real-weight parity (M4 Max, MLX 0.31.2)

| Gate | Result |
| --- | --- |
| Tiny-config per-token (fp32) | max\|Δ\| within **2e-3** (`test_mlx_wan22_parity.py`) |
| CUDA tiny dump→Metal compare | max\|Δ\| **1.15e-3** (`wan22_cuda_reference.py`, L40S dump) |
| Local CPU dump→Metal compare | max\|Δ\| **1.6e-5** (same harness) |
| Full 5B fp16 MLX vs torch-fp16 | max\|Δ\| **9.8e-2**, mean **6.2e-3**, cosine **0.99995** |

Full 5B fp16 drifts more than the tiny gate because 30 layers of Metal vs
torch SDPA accumulate in half precision; cosine 0.99995 confirms the port is
structurally correct. Measured budgets are asserted in
`test_mlx_wan22_real_weights.py` (Metal + local weights).

### Memory / latency (denoise-only, 3-step DMD, flow_shift=5.0)

Shape: pixel 480×832×33 → latent `1×48×9×30×52`, random text embeds.

| Mode | Weight size | Load peak | Denoise peak | Steady step | Total denoise |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | **9.31 GiB** | 9.31 GiB | **10.94 GiB** | **3.79 s** | 11.4 s |
| int8 | **4.95 GiB** | 5.03 GiB | **6.60 GiB** | **3.97 s** | 11.9 s |

**Does 5B fit in 32 GB?** Yes. INT8 denoise peaks at ~6.6 GiB; fp16 at ~11 GiB
— both leave headroom for the OS, torch/MPS encode, and (later) decode on a
32 GB Mac. INT8 is the practical default for the medium hardware tier.

Steady step is ~4× a 1.3B INT8 denoise step at similar shapes (1.3B INT8 was
~1 s-class on denser paths; exact cross-model comparison depends on latent
token count — 5B uses 48 channels and 16× spatial VAE).

### Decode note

The 5B VAE is **z_dim=48** (Wan2.2). TAEHV `taew2_1.pth` is built for the
Wan2.1 VAE and is **not** a drop-in. Until a 2.2-compatible TAE lands upstream,
full Wan2.2 VAE decode on torch-MPS is the decode path; chunked/tiled decode
may be required on 32 GB once the decoder is wired.

### Harness

```bash
# Memory/latency (needs ~/models/fastwan22_5b or --model-root)
PYTHONPATH=$PWD python -m fastvideo.benchmarks.mlx_wan22_5b_bench \
  --model-root ~/models/fastwan22_5b --modes fp16,int8

# Real-weight parity (Metal + weights)
pytest fastvideo/tests/mlx/test_mlx_wan22_real_weights.py -q -s

# CUDA reference (dump on Modal L40S, compare on Mac)
modal run fastvideo/tests/modal/launch_l40s_job.py \
  --command "python fastvideo/tests/modal/wan22_cuda_reference.py dump --path /root/data/wan22_ref/ref.npz" \
  --gpu-type L40S --num-gpus 1 --install-extra dev --pr-number <PR#> \
  --env-vars "MASTER_ADDR=localhost,MASTER_PORT=29551,FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA" \
  --commit-volume
```

### I2V (Track D Rung 4)

TI2V-5B image conditioning needs **no DiT architecture change**: replace latent
frame 0 with the VAE-encoded image and set the per-token timestep so frame-0
tokens are `t=0` (clean) while the rest stay at the denoise level. Helpers live
in `fastvideo/mlx_runtime/wan22_i2v.py`; DiT-level parity (tiny config, no VAE)
is `test_mlx_wan22_i2v.py` (atol 2e-3). Token order is **frame-major** (first
`tokens_per_frame` entries = image frame).

End-to-end I2V still needs the Wan2.2 VAE (z_dim=48) on torch-MPS for encode +
decode — TAEHV `taew2_1.pth` is Wan2.1-only.

### 5B QAD arming (run-6 gate)

Cheap arming check (no training loop): load the 5B FullAttn transformer, apply
`MLXQuantizationAwareCallback`, assert ≥300 fake-quantized weights.

```bash
# Modal (CUDA) — preferred for HF download onto the volume
modal run fastvideo/tests/modal/launch_l40s_job.py \
  --command "python fastvideo/tests/modal/wan22_5b_qad_arming.py" \
  --gpu-type L40S --num-gpus 1 --install-extra dev --pr-number <PR#> \
  --env-vars "MASTER_ADDR=localhost,MASTER_PORT=29561,FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA" \
  --commit-volume
```

Look for: `mlx_qat: fake-quantizing N weights (int8, ...)` with N ≥ 300, no
`matched no weights`, no FSDP/DTensor/parametrizations Traceback.

**Measured (local, FullAttn 5B transformer, 2026-07-08):** `mlx_qat` armed
with **307** weights (int8, group_size=64); no Traceback.

### Hardware tiering (PR #6)

Once PR #6 (`hardware_tier.py`) and this 5B stack (#8/#9 + I2V) are merged, set:

```python
FIVE_B_MODEL_REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
```

Do **not** set this on #6 alone — the tier would recommend a model the runtime
cannot load until the port lands. INT8 5B denoise peak ~6.6 GiB fits 32 GB.

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
