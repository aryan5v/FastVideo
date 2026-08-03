# MLX Runtime — Next Wins Beyond RIFE Fast Mode

Status: design, 2026-08-03. Context: the 14B INT8 QAD run (GB200 job chain
1087→1088→1089) produces the checkpoint this runtime will ship. This note
inventories what the runtime already has, then ranks innovation candidates by
payoff / effort, easiest first.

## What already exists (do not rebuild)

| Piece | Where | Notes |
|---|---|---|
| On-device DMD sampling | `mlx_runtime/sampling.py` | No host round-trips; lazy graph intact |
| Fused attention | `fastwan.py` | `mx.fast.scaled_dot_product_attention` (MFA-class path) |
| Fused norms | `fastwan.py` | opt-in `mx.fast.layer_norm` / `rms_norm` |
| `mx.compile` | example CLI `--mlx-compile` (default on) | DiT-level |
| RIFE fast mode | `--fast --fast-factor N` | generate fewer frames, interpolate with rife-mlx |
| TAEHV fast decode | `--decode-backend taehv` | checksum-verified, vendored MIT source |
| Memory tiering | `mlx_runtime/memory.py` | MLX allocator caps + MPS watermarks; 16/24/32 GB presets |
| int8 checkpoint | `mlx_runtime/checkpoint.py` | pre-quantized save/load, the QAD deploy artifact |
| Windowed attention kernel | `mlx_runtime/windowed_attention.py` | chunked, sink-aware, O(S·w); probe only, not wired |
| Hardware probes | `accel_probe.py`, `attn_scaling_probe.py` | M5 Neural Accelerator + attention scaling measurements |
| Quant backend zoo | `quant_backends.py` | int8/int4/mxfp8/mxfp4/nvfp4 support matrix |

## Ranked candidates

### 1. Prompt-embedding cache (trivial, ship now)

`--prompt-encode-mode inline` re-encodes the prompt every run — for 14B the
umT5 encode is seconds per invocation and dominates short experiments. Cache
`prompt_embeds` keyed by `(prompt, max_sequence_length, encoder dtype)` under
the model root. Iterating on seeds/steps/RIFE factors becomes encode-free.

### 2. MetalFX spatial fast mode (medium, big UX)

The Turbo path from the design docs: generate at half resolution, upscale with
MetalFX (temporal anti-aliased upscaling, Apple-silicon native). Correction
from code inspection: `fastvideo/benchmarks/eval_metalfx_rife.py` is RIFE-only
despite the name — no MetalFX upscaler exists yet, so this win includes
building the upscaler (small MLX SR model or MPS-backed MetalFX binding) plus
the `--fast-spatial` CLI flag composing with `--fast` (RIFE). Spatial and
temporal shortcuts are orthogonal: 4× fewer pixels ≈ 4× less DiT work at
fixed frames/steps.

### 3. Full-step compile coverage (small)

`mx.compile` currently covers the DiT. The denoise step (scheduler math +
AdaLN table lookups + rope grid) is already on-device but not compiled as one
graph. Compiling the whole step function removes per-op dispatch overhead
across 50 transformer blocks × 3 steps. Watch the documented trap: arrays
evaluated mid-graph break `mx.compile` (see `fastwan.py:404`).

### 4. Windowed attention as opt-in draft mode (medium, needs a gate)

The kernel and scaling probe exist but are not wired into generation. At
S=32k, window 4096 + 64 sinks cuts attention work ~8×. The QAD recipe
deliberately keeps attention 100% dense for quality, so this ships as an
explicit `--draft-attention` mode with an SSIM gate vs dense in
`tests/mlx/`, never as default. Useful for interactive prompt exploration
before a final dense render.

### 5. Sequenced residency for 14B (medium, required for 24 GB)

The 1.3B runtime co-resides encoder + DiT + decoder. At 14B int8 (14.9 GB)
that breaks the 24 GB tier. Encode → free encoder → denoise → free DiT →
decode, with peak = max(stages) not sum. `memory.py` has the caps; the runtime
needs the stage lifecycle. This is a hard launch requirement for the 14B
checkpoint, not an optimization.

### 6. W8A8 fused INT8 GEMM for Metal (large, the ambitious bet)

The M5 survey showed weight-only quantization buys zero speed: every GEMM
runs fp16 arithmetic on dequantized weights and the integer matrix units sit
idle. The only route to quantization-derived speed on Metal is quantizing
activations too (W8A8) with a fused integer GEMM — Ideogram shipped exactly
this for consumer GPUs; nobody has done it for diffusion on Metal against M5
Neural Accelerators. Needs timestep-aware activation calibration (DiT
activation distributions shift across the denoise trajectory) and a custom
Metal kernel. Highest payoff, highest effort; do after 1–5.

### 7. Causal streaming mode (large, differentiator)

`causal*.py` in `mlx_runtime/` is the SelfForcing-style causal path —
chunk-wise autoregressive generation with KV reuse, enabling streaming /
interactive video. Already prototyped; maturing it is a product decision
(different app shape, not a speed win).

## Recommendation

Ship 1–3 with the 14B launch (they are cheap and compound), gate 4 behind
SSIM, treat 5 as launch-blocking for the 24 GB tier, and scope 6 as the
follow-up research item that would make the Mac story distinctive.
