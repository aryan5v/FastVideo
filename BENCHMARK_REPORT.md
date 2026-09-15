# MiniMax-H3 hybrid attention on GB200 — kernel/runtime optimization

**Verdict: the gate was not met.** The hybrid got 2.67× faster and remains
2.39× slower than VSA. This report gives the measured split that says *why*, and
what a continuation should attack. Nothing here is estimated.

- **PR:** https://github.com/aryan5v/FastVideo/pull/40
- **Branch:** `agent/h3-hybrid-gb200-kernels`
- **Start commit:** `7cf6ba159a60a34fb97989d94f9199b15493d9e7` — verified to *be* the
  `hybrid-frame-write-scale` tip, not merely to contain it
- **Commits:** `89831174`, `dfa4bc43` (perf) · `4b1becf7`, `3868b772`, `5eabcfa1`,
  `c9e50847`, `f217ce82` (report, receipt, instrumentation, scan revert)
- **Result root:** `/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-kernel-nightly/h3gb200-20260915-163123`

---

## 1. Headline

| path | denoise | E2E | s/NFE | vs VSA |
|---|---|---|---|---|
| VSA strict (P2P on) | **3.349 s** | 7.885 s | 0.837 | — |
| hybrid strict, before | 21.375 s | 25.501 s | 5.344 | 6.38× slower |
| hybrid strict, after | **7.992 s** | 12.706 s | 1.998 | **2.39× slower** |

Both measured in the same allocation, same node class, seed, prompt, shape, frame
count and DiT-evaluation count. **Hybrid improved 2.67×; it did not beat VSA.**
The gate wanted < 3.18 s.

## 2. Geometry (established, not assumed)

```
spatial compression 2·2·2·2·1·1 = 16     temporal 1·2·2·1·1·1 = 4
latent 48×84, 31 latent frames, patch (1,2,2) → 24×42 = 1008 tokens/frame
num_frames = 31, video tokens = 31,248, seq_len ≈ 31.3k
--num-gpus 4 → sp_size 4;  num_attention_heads 56, attention_head_dim 128
→ 14 heads per rank under Ulysses (inner_dim 7168 is independent of hidden 5376)
```

An earlier draft of this work used 124 frames / 4032 tokens-per-frame, which is
4× wrong; every FLOP figure below uses the geometry above.

## 3. The historical baseline was handicapped

The comparison scripts export `NCCL_P2P_DISABLE=1` unconditionally. `nvidia-smi
topo -m` reports **NV18 between all four GPUs** — full NVLink.

| path | P2P disabled (historical) | P2P enabled | gain |
|---|---|---|---|
| VSA strict | 9.097 s / 14.091 s | **3.324 s** / 7.407 s | **2.7×** |
| hybrid strict | 21.617 s / 26.373 s | 21.375 s / 25.501 s | 1.01× |

The historical figures reproduce (9.187→9.097, 21.723→21.617), so they were real
measurements — but of a handicapped configuration. **The fair baseline is
3.324 s.** The asymmetry is also diagnostic: the hybrid barely moved, so it was
never communication-bound, which is why the fix had to target duplicated
*compute* rather than collective volume.

## 4. What was changed

| change | commit | effect |
|---|---|---|
| Request-static window plan | `89831174` | Decomposition, index tensors and output rows built once per (geometry, window params) instead of 200× per request. Arithmetic unchanged. |
| Ulysses head sharding | `dfa4bc43` | QKV projected on the local shard; both branches on 14 heads/rank instead of 56 over the full sequence; both output projections on the local shard. |
| Linear-branch launch cuts | `3868b772` | 5-tap temporal conv as one grouped `conv1d` instead of a 5-iteration tap loop; the delta-rule identity cached per (head_dim, device). |
| Route receipt | `c9e50847` | Logs each distinct SP route once so a silent fall back is visible. |

### Numerics preserved
Per-layer/per-head `write_log_scale` and the runtime tokens-per-frame adjustment
(`gamma = head_dim/tokens_per_frame · exp(write_log_scale + token_logit)`); FP32
Gram, Cholesky and scan paths; text state; anchor frames; uneven final chunks;
the far branch, gates and window radius untouched; no `sana_scaled` revert.
Dense/VSA untouched — nothing here touches `fastvideo/attention/`,
`fastvideo/layers/`, or the dense DiT path.

### Fallbacks (receipted, logged once)
SP=1, head count not divisible by SP, and quantized parameters each fall back to
the replicated route.

## 5. Correctness

```
pytest test_minimax_h3_hybrid_head_shard.py test_minimax_h3_hybrid_window.py \
       test_minimax_h3_hybrid_linear.py test_minimax_h3_hybrid_param_mapping.py \
       test_minimax_h3_hybrid_attention.py test_fp8_hybrid_suffixes.py
-> 67 passed
```

The head-shard tests simulate a full rank set on one process with pure-tensor
stand-ins for the collective, and assert the route receipt so a silent fall back
cannot pass. Bugs the tests caught, all of which would have shipped silently:

1. **The quantization guard matched every layer.** `ReplicatedLinear` always
   carries a `quant_method`; unquantized ones carry `UnquantizedLinearMethod`.
   Testing `quant_method is not None` disabled head sharding *entirely* — an
   optimization that reports success while doing nothing.
2. **A rank-4 view assigned into a rank-3 slice** in the window rewrite;
   mis-broadcasts `head_dim` instead of raising.
3. **An unseeded RoPE and a `mean|want|`-normalised tolerance** made the head-shard
   test pass or fail on the draw.

## 6. The measurement that changed the plan

`torch.profiler` **cannot** attribute this model: FastVideo runs the DiT in
*spawned worker processes*, so parent-process `record_function` hooks never reach
them, and a `sitecustomize`+`atexit` variant lands after teardown when
`key_averages()` is empty. Both attempts returned zero kernels. The working
approach is CUDA-event timers inside the model behind an env var.

Per DiT evaluation (hybrid strict, 4×GB200):

| phase | ms/NFE | share |
|---|---|---|
| window group SDPA | **652** | 33% |
| separable conv | **284** | 15% |
| norm + RoPE | 135 | 7% |
| scan | 178 | 9% |
| collectives (5×) | 266 | 14% |
| frame statistics | 77 | 4% |
| text state | 62 | 3% |
| Cholesky/solve | 57 | 3% |
| readout matmul | 55 | 3% |
| qkv projection | 65 | 3% |
| to_out + to_out_linear | 46 | 2% |
| alpha / write-strength / gate / gather | ~105 | 5% |
| **hybrid attention total** | **1948** | |
| *unaccounted → backbone* | **176** | |

Caveat: the timer synchronises per phase, so its sum overstates the true total
(measured 8.495 s vs 7.992 s uninstrumented). Use it for *relative* attribution.

### The number that matters

`B ≈ 176 ms/NFE`, so **VSA's own attention costs 837 − 176 = 661 ms/NFE**, not
the ~150 previously assumed. Beating VSA needs hybrid attention **< 661 ms/NFE**
against a window-branch FLOP floor of ~192. That is *reachable in principle*, and
it **reverses an earlier conclusion in this work** that the target was provably
out of range — that claim rested on assuming the backbone dominated VSA.

It was not reached. The honest position: closing 1948 → 661 is a 3× cut, and the
largest item (window SDPA at 652 ms/NFE ≈ 333 TFLOPS, ~15% of BF16 peak) is
already the actual arithmetic, not overhead. The reachable cuts are elsewhere.

## 7. Before/after

| setting | GPUs | precision | frames | DiT calls | warm E2E | denoise | s/NFE | video |
|---|---|---|---|---|---|---|---|---|
| VSA strict, P2P off (historical env) | 4 | BF16 | 124 | 4 | 14.091 | 9.097 | 2.274 | `videos/vsa_strict_p2pdisabled/` |
| VSA strict, P2P on | 4 | BF16 | 124 | 4 | 7.885 | 3.349 | 0.837 | `videos/vsa_opt_strict/` |
| old hybrid strict, P2P off | 4 | BF16 | 124 | 4 | 26.373 | 21.617 | 5.404 | `videos/hybrid_strict_p2pdisabled/` |
| old hybrid strict, P2P on | 4 | BF16 | 124 | 4 | 25.501 | 21.375 | 5.344 | `videos/hybrid_strict_p2penabled/` |
| **optimized hybrid strict** | 4 | BF16 | 124 | 4 | **12.706** | **7.992** | **1.998** | `videos/hybrid_opt_strict/` |
| final strict + production re-run | | | | | *pending* | *pending* | | `videos/final_*/` |

Run-to-run stability: VSA measured 3.324 and 3.349 across two allocations (~0.8%),
so the 2.67× hybrid delta is far outside noise.

Not measured: FP8, 1/2/8-GPU scaling, 345-frame. Deprioritized below the 4-GPU gate.

## 8. A reverted optimization, recorded so it is not retried blindly

A Hillis–Steele doubling scan replaces the 58 sequential `baddbmm` calls
(178 ms/NFE, measured at **61 µs per launch** — pure launch latency on a
dependency chain) with `ceil(log₂F) ≈ 5` batched steps. It was implemented and it
**passed** a dedicated equivalence test against the eager recurrence for frame
counts 1,2,3,5,8,29,31, with and without text state.

It was still reverted: it forms `A = T₀·T₁···T₂₈` as a single composite and
applies the text state through that product. For these delta-rule transitions
(inverses of frame Gram matrices) the composite is far worse conditioned than the
incrementally-updated state — it broke head-shard equivalence (**0.95 relative
error**) on real model parameters while passing on random ones. The spec requires
the FP32 scan semantics be preserved. Any retry must address conditioning first
(e.g. chunked scans with periodic state resets), not just launch count.

## 9. Remaining bottlenecks, ranked

1. **Window group SDPA — 652 ms/NFE.** ~217 TFLOP/NFE at ~333 TFLOPS. This is
   real arithmetic at poor MFU for `B=1, H=14, d=128` shapes. Next moves: batch
   the 7 rectangles into one varlen flash call to improve K/V reuse, or a
   block-banded kernel with a chunk mask. Do **not** expect 3×.
2. **Separable conv — 284 ms/NFE** for ~5.6 GFLOP/layer (~1 TFLOP/s). Layout, not
   arithmetic: `[F,H,W,C] → [F,C,H,W]` permutes materialise ~112 MB copies either
   side of cuDNN's depthwise path. Keeping `channels_last` end-to-end should avoid
   both copies; this is the best-effort-to-payoff item left.
3. **Norm+RoPE — 135 ms/NFE.** Instrumentation showed `FASTVIDEO_MINIMAX_H3_FUSIONS=0`
   under the strict profile, so this is the *unfused* path. The production profile
   enables Sol-Engine fusions; the pending production run will quantify it.
4. **Collectives — 266 ms/NFE.** Four all-to-alls plus two all-gathers per layer.
   Fusing the K/V exchange into one call would cut two of them.

## 10. Paths, environment, confirmations

- Root: `/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-kernel-nightly/h3gb200-20260915-163123`
- Logs `logs/`, submitted scripts `scripts/submitted/`, phase timings
  `profilers/phases/*.jsonl`, videos `videos/`, baselines `results/baselines.csv`
- 4× NVIDIA GB200 (sm100), NV18, torch 2.12.0+cu130, CUDA 13.0, Python 3.12.14,
  container `fastvideo-dev-sm100-2f28adad2e05.sqsh`, nodes `hpc-rack-2-5`,
  `hpc-rack-2-6`, `hpc-rack-3-5`; all jobs `sbatch`, 4 GPUs, `--exclusive`
- **Strict matched conditions: hybrid did NOT beat VSA (7.992 vs 3.349).**
- Dense/VSA unmodified semantically; VSA moved only with the environment.
- **All GPU allocations released** (jobs 9080–9125 COMPLETED/CANCELLED). The jobs
  still running under this account belong to other experiments and were never
  touched.
