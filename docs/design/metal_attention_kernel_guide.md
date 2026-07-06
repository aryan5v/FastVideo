# Guide: Metal Attention Kernel Program (Stage 0 → Track A → Track B)

For the Mac agent session. Objective: replace `mx.fast.scaled_dot_product_attention`
in the MLX FastWan runtime with our own Metal attention kernel, then extend it
with INT8 quantization (Track A) and VSA-style block sparsity (Track B).
Attention is the dominant denoise cost (dense, ~29k tokens at 480p×81f), so
this program is the main latency lever after `mx.compile`.

Researcher framing: the original FastWan-QAD's decisive move was quantized
*attention* (SageAttention3/Attn-QAT), not just linear layers. No equivalent
exists on Metal. Track A is the direct Apple analog and is publishable on its
own; Track B brings FastWan's sparse-distillation to the platform. Both are
features of one kernel, so they share Stage 0.

## Stage 0 — dense flash attention in Metal

**Deliverable:** `fastvideo/mlx_runtime/attention.py` exposing
`flash_attention(q, k, v, *, scale, causal=False, kv_start=0)` built on MLX's
custom-kernel API (`mx.fast.metal_kernel`; verify exact API against the
pinned MLX version's docs — it JIT-compiles a Metal source string with typed
input/output specs). Fall back to an MLX C++ extension only if the JIT path
can't express threadgroup memory the way the algorithm needs.

**Algorithm** (standard flash-attention-2 structure, adapted to Metal):

- One threadgroup per (batch, head, query-tile). Query tile in registers /
  threadgroup memory; iterate over KV tiles.
- Online softmax: track running max `m` and running sum `l` per query row;
  rescale the output accumulator when `m` updates. Accumulate in fp32,
  inputs fp16.
- Tile sizes are the central tuning knob on Apple GPUs (start Bq=32, Bk=64;
  sweep). M-series threadgroup memory is 32 KB — budget
  `Bq*d + Bk*d (K) + Bk*d (V)` in half precision, d=128.
- Design in from day 1 (even though Stage 0 ships dense/non-causal):
  a `causal` flag with `kv_start` offset (Track C's cached decoding needs
  "queries at global positions kv_start..kv_start+Lq attend to keys 0..";
  see `mac_streaming_causal_guide.md`), and a per-KV-tile *skip mask* input
  (Track B's sparsity is "skip this KV tile entirely" — sparsity lands as a
  mask argument, not a rewrite).

**Gates (in order):**

1. Correctness: new pytest `fastvideo/tests/mlx/test_metal_attention.py` —
   vs `mx.fast.scaled_dot_product_attention` on random shapes including the
   real FastWan shape (heads=12, d=128, L∈{1k, 8k, 29k}), fp16 in / fp32
   accumulate, tolerance pinned with measured headroom (expect ≲1e-2 fp16
   absolute; record actual). Include ragged L not divisible by tile sizes.
2. Integration: swap into `MLXWanTransformerBlock` behind an env flag
   (`FASTVIDEO_MLX_ATTENTION=custom|fast`), then the full-DiT parity test
   must pass with the custom kernel, and the benchmark's
   `--assert-min-ssim 0.98` vs the `fast` path must hold.
3. Performance: benchmark per-step time vs `mx.fast` SDPA at the real shape.
   **Stage 0 does not need to beat `mx.fast`** (Apple's kernel is good);
   within ~15% is a pass, because A and B only exist on our kernel. Record
   the gap in `apple_silicon_benchmark_baseline.md`.

## Track A — INT8 attention + attention-QAT

**Kernel side (Mac):** quantize the QKᵀ and PV matmuls. Follow the
SageAttention recipe as the reference design: per-block INT8 quantization of
Q and K (per-tile amax → scale), keep the softmax and accumulators in
fp16/fp32, optionally INT8 for V with per-channel scales. Smoothing (subtract
per-head K mean before quantizing, add back via a correction term) is what
makes INT8 K viable — implement it; it is the difference between "works on
toy shapes" and "works on video-length sequences".

**Numerics-twin side (any machine):** a pure-torch
`fake_quantize_attention(q, k, v)` in `fastvideo/layers/quantization/` that
reproduces the kernel's quantization decisions (scales, rounding, smoothing)
exactly — same discipline as `mlx_affine_qat.py`, and the same test shape:
pin it against the Metal kernel's intermediate outputs (add a kernel debug
mode that returns quantized q/k tiles for the test).

**Training side (DGX run 3):** extend `MLXQuantizationAwareCallback` with
`quantize_attention: true` — wrap the student's attention modules the same
forward-scoped way, calling the torch twin. Recipe = run-2 YAML + that flag.
Note the repo's CUDA `attn_qat_train` Triton backend exists as prior art for
where to hook attention fake-quant in training; ours must match *our* Metal
numerics, not theirs.

**Gates:** (1) twin-vs-kernel numerics test green before run 3 launches;
(2) PTQ sanity first — run the benchmark with INT8 attention on the run-2
model *without* retraining; expect visible degradation (that gap is the
budget QAT must close, and the honest baseline for the writeup);
(3) run-3 model: SSIM-vs-own-FP16 with INT8 attention ≥ the linear-only
number (~0.95), and grid review at parity.

## Track B — VSA-style block sparsity

**Reference implementation to mirror:**
`fastvideo/attention/backends/video_sparse_attn.py` — tiles the video token
grid into (ts×hs×ws) tiles (`get_tile_partition_indices`), computes per-tile
mean-pooled Q·K scores, keeps top-k KV tiles per query tile plus mandatory
tiles. The CUDA kernels live in `fastvideo-kernel/`; do not port them —
reimplement the *selection math* in MLX (it is plain matmul/topk on pooled
tensors) and feed the result to Stage 0's per-KV-tile skip mask.

**Critical alignment:** training (DGX, CUDA VSA) and deployment (Metal) must
select the *same tiles* given the same inputs. Gate: a cross-implementation
test that runs the CUDA selection (exported as reference tensors from the
DGX; the Mac cannot run it) against the MLX selection on identical inputs —
tile-index sets must match exactly at fp32, and the tolerance behavior at
fp16 must be characterized before run 4.

**Training side (DGX run 4):** the repo already trains with VSA
(`distill_dmd_VSA` recipes, `WanTransformerBlock_VSA`). Run 4 = run-2 recipe
with the VSA attention backend + the sparse-distill settings from
`docs/distillation/dmd.md`, plus linear INT8 QAT. Sparsity at 480p×81f
should cut attention cost several-fold at ~equal quality (that is FastWan's
published claim; we verify it on Metal).

**Sequencing rule:** B starts only after A's kernel is stable, and run 4
only after run 3's model ships or is abandoned for cause — one attention
change per training run.

## What to ask the DGX agent for

- Run 3 / run 4 launches (gated as above), standard runbook lifecycle
  (smoke → full → dual export → transformer upload).
- Reference tensors for the Track B selection-equivalence test (a small
  script's `.pt` outputs, shipped to the Mac).
- Nothing else — kernels never touch the DGX.

## Deliverables checklist

- [ ] `fastvideo/mlx_runtime/attention.py` + `test_metal_attention.py` (Stage 0)
- [ ] `FASTVIDEO_MLX_ATTENTION` integration + full-DiT parity green on custom kernel
- [ ] INT8 attention mode + torch numerics twin + twin gate (Track A)
- [ ] attn-QAT callback flag + run-3 handoff (Track A)
- [ ] MLX tile selection + skip-mask sparsity + selection-equivalence gate (Track B)
- [ ] run-4 handoff (Track B)
- [ ] benchmark rows for every stage in `apple_silicon_benchmark_baseline.md`
