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


### 2b. Local prompt enrichment, Context-IR-style (small, quality)

Borrowed from MiniMax-H3's Context-IR stage (its docs call it "critical to
the quality of the final output"). Wan's training captions are long and
cinematic; user prompts are short, and the gap costs visible quality. Ship an
optional `--enhance-prompt` pre-pass in the MLX runtime: a small local LLM via
mlx-lm expands the user prompt into Wan-style cinematography language. Reuse
the system prompts and operation contract from
`fastvideo/entrypoints/streaming/prompt/` (the streaming server's
enhance/rewrite orchestration) but with a local backend instead of the remote
providers. No training required; evaluate with the standard SSIM prompt set
before defaulting on.

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

### 6. W8A8 fused INT8 GEMM for Metal — **CLOSED, NO-GO (2026-08-04)**

We ran the full gate chain. Details and tables:
`docs/design/w8a8_int8_gemm_metal.md` (canonical receipt) and the launch blog
draft section *"We Tried to Make INT8 Fast"*.

| Gate | Result |
|---|---|
| 1. Act calib (5B randn + realctx, 14B) | **PASS** — per-token scales OK; ~7% layers (`attn2.to_q`) stay fp16 |
| 2. Fused int8×int8 `metal_kernel` on **M5 24 GB** | **NO-GO** — correct (3e-6 err) but **0.01–0.85× fp16**; DiT shapes ~0.02–0.04× |
| 3. W8A8 QAT on GB200 | **Cancelled** (no deploy kernel to train toward) |

Mechanism: `mx.fast.metal_kernel` is a scalar-MAC shader; MLX 0.32 does not
expose an M5 Neural Accelerator int8×int8 path. Weight-only
`mx.quantized_matmul` stays ~fp16 (dequant then fp16 arith) — confirms the
survey. **Do not reopen** unless MLX grows real int8 matmul or a lower-level
kernel clears the pre-registered **1.2× fp16** ship bar at DiT shapes.

INT8 **weight-only** QAD (1.3B/5B/14B launch) is unaffected: that is the
memory/quality grid, not this speed bet.

### 7. Causal streaming mode (large, differentiator)

`causal*.py` in `mlx_runtime/` is the SelfForcing-style causal path —
chunk-wise autoregressive generation with KV reuse, enabling streaming /
interactive video. Already prototyped; maturing it is a product decision
(different app shape, not a speed win).

## Recommendation

Ship 1–3 with the 14B launch (they are cheap and compound), gate 4 behind
SSIM, treat 5 as launch-blocking for the 24 GB tier. **Item 6 is closed
NO-GO** — keep the prototype as an oracle, put the M5 table in the launch
blog, and spend no further GB200 on W8A8 QAT. Mac speed continues to come
from pipeline levers (RIFE / fast-spatial / refine / step count), not from
GEMM format tricks under current MLX.
