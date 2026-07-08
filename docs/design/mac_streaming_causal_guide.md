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

## Rung 1 — architecture diff (completed 2026-07-08)

`causal_wanvideo.py` vs `wanvideo.py`, enumerating every difference the MLX
port (`fastvideo/mlx_runtime/causal.py`) must reproduce on top of the existing
dense port (`fastvideo/mlx_runtime/fastwan.py`). The weight layout is identical
(`CausalWanTransformer3DModel` reuses `WanVideoConfig().param_names_mapping`),
so **the loader is unchanged — only the forward/attention differs.**

| Aspect | Non-causal `wanvideo.py` | Causal `causal_wanvideo.py` | MLX port implication |
| --- | --- | --- | --- |
| Forward modes | single `forward` (full sequence) | `_forward_inference` (KV-cached, per-chunk) + `_forward_train` (block-causal flex mask); `forward` dispatches on `kv_cache` presence | Port only `_forward_inference`; the train mask path is CUDA-only and not needed |
| Attention masking | none (dense bidirectional over whole seq) | train: flex-attention `BlockMask` (`_prepare_blockwise_causal_attn_mask`); inference: **mask-free**, chunk Q attends dense over `cache[0:end]` | No mask at MLX inference. Core parity claim (Rung 3): mask-free cached decode == block-causal masked full pass |
| Self-attn op | `LocalAttention` dense SDPA, done inline in block | `CausalWanSelfAttention` dense SDPA (`causal=False`) over the cached K/V window | Dense `mx.fast` SDPA suffices — **no new kernel** |
| KV cache | none | dict `{k, v, global_end_index, local_end_index}`, preallocated buffer; new K/V written at `local_start:local_end` (`causal_wanvideo.py:143-193`) | Preallocated `mx.array` K/V buffers per block; write-at-index |
| Rolling eviction + sinks | n/a | on overflow, shift cache left after the first `sink_tokens`, evicting oldest (`:160-171`); `max_attention_size = local_attn_size*frame_seqlen`, or `21*frame_seqlen` when `local_attn_size==-1` (`GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES`) | Replicate shift/sink arithmetic exactly; KV-cache unit tests gate it |
| Rotary | `get_rotary_pos_embed((F,H,W), …)`, positions from 0 | same, plus `start_frame=start_frame` global offset; rope applied to Q/K **before** cache write, at global positions | Apply rope at global offset per chunk (Q/K roped, then cached) |
| Cross-attention | recomputed every pass | `crossattn_cache` `{is_init, k, v}` — context K/V computed once, reused across chunks (`wanvideo.py:167-176`) | Compute cross-attn K/V once per generation, cache |
| Timestep conditioning | one `temb` for whole sequence (`[B,6,dim]` or `[B,seq,6,dim]`) | per-chunk: `timestep_proj` → `(6, hidden)`; block reshapes norm by `temb_seq_len`/`tokens_per_temb` (`causal_wanvideo.py:285-304`) | Per-chunk temb; norm unflatten by tokens-per-temb-frame |
| Chunk loop | full sequence at once | `_forward_inference` runs per frame-block; `current_start`/`cache_start` advance each call (CausVid Alg. 2) | Streaming sampler drives the per-chunk loop |
| QK norm | `RMSNorm(dim)` inside attn | block-level `norm_q/norm_k` = `RMSNorm(dim)` (`rms_norm_across_heads`), `forward_native` | Same norm; reuse MLX rms |
| New config fields | — | `num_frames_per_block` (≤3), `local_attn_size`, `sink_size`, `independent_first_frame` | Thread through MLX config |

Reference SF weights exist (`SFWan2.1-T2V-1.3B` family), so Rungs 2–5 need no
training run. Next: Rung 2 — `causal.py` cached chunked-attention wrapper.

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

- [x] Architecture diff table (rung 1) appended to this doc
- [~] `fastvideo/mlx_runtime/causal.py` — rung 2 done: `MLXCausalKVCache`
  (preallocated rolling cache + sink tokens) and `causal_self_attention_step`
  (rotary at global offset, cache write/evict, windowed dense SDPA). Sampler
  (rung 5) pending.
- [~] Rung 3 parity + KV-cache tests done in
  `fastvideo/tests/mlx/test_mlx_causal_attention.py`: mask-free cached decode ==
  block-causal masked full pass (no-eviction), == sliding-window masked pass
  (with eviction), and sink-token preservation. Full-block/full-model parity
  (rung 4, vs torch) pending.
- [ ] Streaming demo script (`examples/inference/basic/mlx_wan_streaming.py`)
- [ ] Benchmark rows: time-to-first-frame, chunk latency, peak memory
- [ ] Run-5 handoff (SF+QAD) when gates are green
