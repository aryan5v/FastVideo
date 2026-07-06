# Guide: Streaming / Causal Generation on Mac (Track C)

For the Mac agent session. Objective: port FastVideo's Self-Forcing causal
Wan to the MLX runtime and ship a live-preview demo — frames appearing while
generation continues. This is the track with the highest product visibility
per unit of technical risk: it needs **no new kernels** (dense `mx.fast`
SDPA suffices; Stage 0's `causal`/`kv_start` mode slots in later) and the
model/training side already exists in the repo.

Researcher framing: bidirectional video DiTs make you wait for the whole
clip; causal (block-autoregressive) DiTs emit chunks of frames as they go.
On a laptop, perceived latency is the product. A 3-step-per-chunk causal
model streaming 480p on a MacBook is a demo nobody else has, and it
composes with everything already built (INT8 QAT, TAEHV, presets).

## The reference implementation, precisely

`fastvideo/models/dits/causal_wanvideo.py` — read fully before porting.
Key mechanics to reproduce:

- **Chunked self-attention with a KV cache.** `CausalWanSelfAttention.forward`
  takes `block_mask` (torch flex-attention `BlockMask`), `kv_cache` (a dict:
  `k`, `v`, `global_end_index`, `local_end_index`), `current_start`,
  `cache_start`. Tokens arrive one frame-block at a time; new K/V are
  written at `local_start_index:local_end_index`.
- **Rolling cache with sink tokens.** When the cache would overflow the
  local-attention window, the oldest tokens after the first `sink_tokens`
  are evicted by shifting content left (see lines ~140–175). This is what
  bounds memory for long/streaming generation.
- **Block-causal masking** via `create_block_mask` — chunk i attends to
  chunks ≤ i. Flex-attention is a *training* convenience; at MLX inference
  you do not need masks at all: attend the current chunk's queries over
  `cache[0:end] + current chunk` densely. That equivalence (mask-free
  cached decoding == block-causal masked full pass) is the core porting
  insight and the thing the parity test must prove.
- The Self-Forcing pipelines and recipes:
  `fastvideo/training/self_forcing_distillation_pipeline.py` (legacy,
  authoritative for shipped SF models),
  `fastvideo/train/methods/distribution_matching/self_forcing.py` (new
  stack), `examples/train/configs/distribution_matching/wan/self_forcing_causal_t2v.yaml`,
  and released checkpoints in the registry (`SFWan2.1-T2V-1.3B` family) —
  meaning Track C can start from **existing causal weights** and does not
  wait for any training run.

## Port ladder (same discipline as the original Wan port)

1. **Rung 1 — architecture diff.** Read `causal_wanvideo.py` against
   `wanvideo.py`; enumerate every difference (attention call signature,
   rotary handling under `current_start` offsets, timestep conditioning per
   chunk, any extra norms). Write the diff into this doc as a table before
   coding.
2. **Rung 2 — MLX causal attention wrapper.** In the MLX runtime, a cached
   attention step is: append new K/V (after rotary at global positions) to
   preallocated `mx.array` buffers; SDPA of the chunk's Q over
   `k[:end], v[:end]`. Implement eviction + sink tokens exactly as the
   torch code does. No flex-attention analog needed.
3. **Rung 3 — block parity.** Tiny-config causal block: torch reference
   (flex-attention path, CPU) vs MLX cached path, feeding chunks
   sequentially; assert the MLX chunked outputs match the torch full-sequence
   block-causal outputs within pinned tolerance. Extend
   `fastvideo/tests/mlx/tiny_wan.py` with a causal variant.
4. **Rung 4 — full-model parity + real weights.** `MLXCausalWanDiT` loading
   the released SF checkpoint; full-forward parity vs torch on CPU
   (the `distributed_setup`/CpuPlatform machinery from the existing parity
   tests carries over).
5. **Rung 5 — streaming sampler + demo.** Chunked DMD sampling (the SF
   models use few-step-per-chunk schedules — read the SF pipeline for the
   exact schedule), TAEHV decoding *per chunk* (TAEHV is causal-friendly;
   decode each frame block as its latents finalize), frames pushed to a
   minimal viewer (start with incremental MP4/image writing +
   auto-refreshing HTML; a Gradio streaming demo is the polished version —
   see `examples/inference/gradio/` for house patterns).
6. **Rung 6 — quantize.** INT8 the causal DiT through the existing
   `quantize_matrix` path (weights are still Wan-block-shaped) and add
   benchmark cells: time-to-first-frame, per-chunk steady latency, peak
   memory with rolling cache. These are the demo's headline numbers.

## Gates

- Rung 3/4 parity tests in `fastvideo/tests/mlx/`, running in both CI jobs.
- KV-cache unit tests: eviction correctness (contents after overflow ==
  torch reference), sink-token preservation, rotary at offset positions.
- Ship gate for the demo: time-to-first-frame and per-chunk latency recorded
  in `apple_silicon_benchmark_baseline.md`; quality eyeballed against the
  released SF model's CUDA outputs (ask the DGX agent for 2–3 reference
  clips, same prompts/seeds).

## DGX asks

- Near-term: nothing but reference clips from the released SF checkpoint
  (small script, minutes).
- Later (run 5): **SF + QAD combined** — the run-2-style INT8 QAT callback on
  top of the self-forcing recipe, giving a causal model that is *also*
  quantization-robust. Gate: rungs 1–4 green plus the standard QAT numerics
  gate (already green from run 2's machinery). Est. ~1–1.5 days on 4×B200.

## Deliverables checklist

- [ ] Architecture diff table (rung 1) appended to this doc
- [ ] `fastvideo/mlx_runtime/causal.py` (cache + chunked attention + sampler)
- [ ] Causal block + full-model parity tests, KV-cache unit tests
- [ ] Streaming demo script (`examples/inference/basic/mlx_wan_streaming.py`)
- [ ] Benchmark rows: time-to-first-frame, chunk latency, peak memory
- [ ] Run-5 handoff (SF+QAD) when gates are green
