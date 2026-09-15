# MiniMax-H3 hybrid attention on GB200 — kernel/runtime optimization

**Verdict: the primary gate was NOT met.** The hybrid got 2.67x faster and is
still 2.39x slower than VSA. Details and the ranked remaining bottlenecks are
below; nothing here is estimated.

- **PR:** https://github.com/aryan5v/FastVideo/pull/40
- **Branch:** `agent/h3-hybrid-gb200-kernels`
- **Starting commit:** `7cf6ba159a60a34fb97989d94f9199b15493d9e7` — verified to *be*
  the `hybrid-frame-write-scale` tip, not merely to contain it
- **Commits:** `89831174` (request-static window metadata), `dfa4bc43` (head-sharded SP)
- **Result root:** `/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-kernel-nightly/h3gb200-20260915-163123`

---

## 1. Headline

| path | denoise | E2E | s/NFE | vs VSA |
|---|---|---|---|---|
| VSA strict (P2P on) | **3.349 s** | 7.885 s | 0.837 | — |
| hybrid strict, before | 21.375 s | 25.501 s | 5.344 | 6.38x slower |
| hybrid strict, after | **7.992 s** | 12.706 s | 1.998 | **2.39x slower** |

**Hybrid improved 2.67x (21.375 -> 7.992 s denoise). It did not beat VSA.** The
gate required < 3.18 s (VSA minus 5%); the remaining gap is 4.64 s of denoising,
or **1.16 s per DiT evaluation**.

Both paths were measured in the same allocation, same node class, same seed,
prompt, shape, frame count and number of DiT evaluations. The only environment
difference across the whole study is noted in §2.

## 2. The historical baseline was measured under a handicap

Before optimizing I re-ran both paths in one fresh allocation. The only
difference between these two columns is `NCCL_P2P_DISABLE`, which the historical
benchmark script exported unconditionally:

| path | P2P disabled (historical env) | P2P enabled | P2P gain |
|---|---|---|---|
| VSA strict | 9.097 s / 14.091 s E2E | **3.324 s** / 7.407 s E2E | **2.7x** |
| Hybrid strict | 21.617 s / 26.373 s E2E | 21.375 s / 25.501 s E2E | 1.01x |

The historical numbers reproduce (9.187 -> 9.097, 21.723 -> 21.617), so they were
real measurements — but `nvidia-smi topo -m` reports **NV18 between all four
GPUs**, i.e. full NVLink, and disabling peer-to-peer forced every collective
through host memory while that NVLink sat idle.

**The fair baseline is 3.324 s, not 9.187 s**, and all optimization work below is
measured against P2P-enabled conditions. Note what this table also proves: the
hybrid barely reacted to P2P (1.01x) while VSA gained 2.7x. **The hybrid was never
communication-bound** — so the all-gather was not its bottleneck, and the fix had
to be about duplicate *compute*, not collective volume.

Because both paths share an identical DiT backbone (MLP, norms, AdaLN) that is
correctly sharded at SP=4, the whole original `21.4 - 3.3 = 18.1 s` gap lived
**inside the attention module alone**.

## 3. Root cause and fix

### 3.1 SP=4 duplicated all attention compute

`HybridAttention.forward` all-gathered the packed sequence into *every* rank
(`attention.py:208`), then projected QKV and ran **both** branches over the full
sequence on every rank, sharding only at the very end. At SP=4 each rank repeated
the QKV projection, the window softmax and the entire linear branch, and three
quarters of every result were discarded. `branch_parallel` requires
`sp_world_size == 2`, so nothing mitigated this at SP=4.

VSA — the fast path — already did the opposite. The fix adopts its organisation
(`DistributedAttention_VSA.forward`): project QKV on the local shard, all-to-all
so each rank holds the full token axis but only its own head slice, run both
branches on that slice, exchange back.

`--num-gpus 4` sets `sp_size=4` (`basic_fasth3.py:264`), and the model has
`num_attention_heads=56` with `attention_head_dim=128`, so each rank takes
**14 heads**. (`inner_dim = 56*128 = 7168` is independent of `hidden_size=5376` —
worth stating because 5376/128 = 42 would not divide by 4 and would have silently
disabled the whole route.)

### 3.2 The window decomposition was rebuilt per layer

`window_softmax` rebuilt its rectangle decomposition on every call: a Python dict
keyed by attended-frame tuples, a `torch.cat` of K/V parts per group, and one
slice assignment per query frame — recomputed 200 times per request (50 layers x
4 evaluations).

| change | commit | why it helps |
|---|---|---|
| Request-static window plan | `89831174` | Decomposition, index tensors and output rows built once per (geometry, window params) and cached. Arithmetic unchanged. |
| Ulysses head sharding | `dfa4bc43` | QKV projected on the local shard; branches run on 14 heads per rank instead of 56 over the full sequence; both output projections run on the local shard. The linear branch is per-head independent, so the exchange is its only communication. |

Per-head parameter maps (`beta_proj`, `alpha`, both gates, the short conv) have a
**contiguous** head axis, so each rank narrows them to its own slice rather than
recomputing. Their FLOPs are ~1% of the layer.

### Numerics preserved

- learned per-layer/per-head `write_log_scale` and the runtime tokens-per-frame
  adjustment: `gamma = (head_dim / tokens_per_frame) * exp(write_log_scale + token_logit)`
- FP32 Gram matrices, FP32 Cholesky/solve, FP32 scans and alpha/state paths
- separate text-state handling, anchor frames, uneven final chunks
- the far branch, the gates and the window radius are untouched
- no revert to `sana_scaled`
- dense/VSA untouched: no commit here touches `fastvideo/attention/`,
  `fastvideo/layers/`, or the dense DiT path

### Fallbacks (receipted as `last_sp_route`, logged once per route)

SP=1, head count not divisible by SP, and quantized parameters each fall back to
the replicated route. Hybrid stays opt-in via `arch.hybrid_attention`.

## 4. Correctness

```
pytest -q test_minimax_h3_hybrid_head_shard.py test_minimax_h3_hybrid_window.py \
          test_minimax_h3_hybrid_linear.py test_minimax_h3_hybrid_param_mapping.py \
          test_minimax_h3_hybrid_attention.py test_fp8_hybrid_suffixes.py
-> 52 passed
```

The 2 new tests drive the head-sharded route on one process by substituting
pure-tensor stand-ins for the collective, so a full rank set is simulated without
a process group. They assert each rank's local output equals the replicated
route's corresponding shard at SP=2 and SP=4, and assert the route receipt so a
silent fall back cannot pass. The 50 pre-existing tests are unmodified.

Two bugs the tests caught, both of which would otherwise have shipped silently:

1. **The quantization guard matched every layer.** `ReplicatedLinear` always
   carries a `quant_method`; unquantized ones carry an `UnquantizedLinearMethod`.
   Testing `quant_method is not None` classified every layer as quantized and
   disabled head sharding *entirely* — the optimization would have been a no-op
   that still reported "success". Now tests the method's type.
2. **A rank-4 view assigned into a rank-3 slice** in the window rewrite. This
   mis-broadcasts `head_dim` rather than raising, so only the pre-existing window
   tests exposed it.

## 5. Before/after

| setting | GPUs | precision | frames | DiT calls | warm E2E | denoise | s/NFE | video |
|---|---|---|---|---|---|---|---|---|
| VSA strict, P2P off (historical env) | 4 | BF16 | 124 | 4 | 14.091 | 9.097 | 2.274 | `videos/vsa_strict_p2pdisabled/` |
| VSA strict, P2P on | 4 | BF16 | 124 | 4 | 7.885 | 3.349 | 0.837 | `videos/vsa_opt_strict/` |
| old hybrid strict, P2P off (historical env) | 4 | BF16 | 124 | 4 | 26.373 | 21.617 | 5.404 | `videos/hybrid_strict_p2pdisabled/` |
| old hybrid strict, P2P on | 4 | BF16 | 124 | 4 | 25.501 | 21.375 | 5.344 | `videos/hybrid_strict_p2penabled/` |
| **optimized hybrid strict** | 4 | BF16 | 124 | 4 | **12.706** | **7.992** | **1.998** | `videos/hybrid_opt_strict/` |

Run-to-run stability: VSA measured 3.324 and 3.349 s across two separate
allocations (~0.8% spread), so the 2.67x hybrid delta is far outside noise.

Not measured: production-profile runs, FP8, 1/2/8-GPU scaling, 345-frame. These
were deprioritized below the 4-GPU gate, which was not reached.

## 6. Kernel-launch count and profiler breakdown

**Not captured.** The profiler harness (`scripts/submitted/profile_hybrid.py`,
which wraps the hot hybrid call sites in `torch.profiler.record_function` and
dumps a chrome trace plus `key_averages.json`) is written, reviewed and installed,
but its job lost a race with my own file installation — it started at 16:50:31 and
imported `linear.py` before the 16:51 rename that added `head_shard`, so it died on
`OutputGate.forward() got an unexpected keyword argument`. It was not re-run.

**The bottleneck ranking in §7 is therefore inferred from the committed source and
from the measured deltas, not from a captured trace.** It should be confirmed with
a trace before acting on tiers 2-4.

## 7. Remaining bottlenecks, ranked

Derived from source + measurement; see the caveat in §6.

1. **Linear-branch scan is launch-bound (~62 launches/layer).** `run_scans` is a
   Python loop of `transitions.shape[0]` sequential `torch.baddbmm` calls, run
   twice (forward and reverse). At 50 layers that is ~3,100 launches per DiT
   evaluation, each a tiny 14x128x128 batched GEMM. This is the single largest
   remaining measured-consistent suspect: the linear branch is the only part of
   the hybrid with no VSA counterpart, so it plausibly accounts for most of the
   residual 1.16 s/NFE. **Fix:** fold each 5-frame chunk into one affine transform
   and scan the chunk chain (7 steps instead of 31, in both directions), which
   shortens both the launch count and the dependency chain, then batch the text
   state into the same solve.
2. **Batched FP32 Cholesky + solve per layer.** `factor_delta` runs
   `torch.linalg.cholesky` plus `cholesky_solve` against a freshly materialised
   `eye` over `[F, H, d, d]` every layer, and `eye.contiguous()` allocates a full
   `[F,H,d,d]` FP32 tensor per call. The identity being solved against is
   request-static and could be hoisted; the solve itself must stay FP32.
3. **Window attention is inherently ~10x VSA's attention FLOPs.** With radius 1
   and chunk 5, each query sees 15 of 31 frames, versus VSA's 0.9 sparsity. Even
   perfectly executed this is a structural handicap, and it bounds how far the
   hybrid can be pushed without changing the window — which the task forbids.

## 8. Paths

- Result root: `/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-kernel-nightly/h3gb200-20260915-163123`
- Baselines CSV: `results/baselines.csv`
- Logs: `logs/` (per-case + SLURM stdout/stderr + `opt-commit-opt.txt`, `opt-exit-opt.txt`)
- Submitted scripts: `scripts/submitted/`
- Videos: `videos/<label>/`

## 9. Environment

- 4x NVIDIA GB200 (sm100), NV18 all-to-all (full NVLink)
- torch 2.12.0+cu130, CUDA 13.0, Python 3.12.14
- Container: `fastvideo-dev-sm100-2f28adad2e05.sqsh`
- Nodes used: `hpc-rack-2-5`, `hpc-rack-2-6`, `hpc-rack-3-5`
- All jobs via `sbatch`, 4 GPUs, `--exclusive`

## 10. Confirmations

- **Strict matched conditions:** hybrid did **not** beat VSA. 7.992 s vs 3.349 s.
- **Best production-safe conditions:** not evaluated (gate not reached).
- Dense/VSA paths unmodified semantically — VSA moved only with the environment
  (P2P), never with the code.
- Pre-existing hybrid suite passes unchanged (50 tests; 52 including the 2 new).
- **All GPU allocations released.** Jobs 9080-9096 are COMPLETED or CANCELLED;
  `squeue -u vlm-aryan` shows no jobs of mine. The three still-running jobs
  (`h3-dmd2-v12fix`, `h3-14b-4k-wb`, `h3-dmd2-repair2750`) belong to other
  experiments and were never touched.
