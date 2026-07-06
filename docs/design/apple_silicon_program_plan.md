# Apple Silicon Program Plan: Tracks A/B/C + the 5B Model

The master plan for the next phase of FastVideo-on-Mac, written for the agent
sessions that will execute it. Three execution environments cooperate:

- **Mac session** (Claude Code on the M4 Max, repo clone, branch
  `aryan/apple-silicon-fastwan-mlx`): all Metal/MLX kernel and runtime work,
  all on-device benchmarks. The Mac is the only place kernels can be
  iterated — CI runners are too slow and the Linux sandbox has no Metal.
- **DGX session** (existing agent): all training runs, per the established
  runbook pattern. GPUs are shared — 4 max unless idle.
- **Coordinator session** (Linux sandbox): planning, review, cross-cutting
  code that needs no device, doc upkeep.

Read `apple_silicon_fastvideo.md` (roadmap + history) and
`apple_silicon_qad_runbook.md` (the run-1/run-2 QAD lifecycle) first. The
program below assumes run 2 of the 1.3B QAD is the active training job.

## The one law of this program

**No training run starts before its deploy-time numerics gate is green, and
no kernel ships before its parity gate is green.** This is not process
overhead; it is the lesson of run 1 (three integration failures found by
gates, one quality failure found only by eyes) and of linear-QAD's success
(bitwise-pinned fake-quant → 0.949 robustness vs 0.907 PTQ). Every stage
below names its gate.

## Track map and dependencies

```
Stage 0: dense Metal flash-attention kernel        [Mac]  ── gate: parity vs mx.fast SDPA
   ├── Track A: INT8 attention (kernel + attn-QAT) [Mac + DGX run 3]
   └── Track B: VSA-style sparse attention         [Mac + DGX run 4]
Track C: streaming/causal Wan on MLX               [Mac; independent of Stage 0]
   └── SF+QAD causal run                           [DGX run 5]
Track D: Wan2.2-TI2V-5B port (+ I2V)               [Mac; independent]
   └── QAD-5B run                                  [DGX run 6]
Run 2 (1.3B QAD v2)                                [DGX; in flight, independent]
```

Tracks C and D need **no new kernels** — they run on the existing dense
`mx.fast` SDPA and the proven QAD machinery. They are the low-risk,
high-visibility work. Stage 0 → A → B is the high-risk, high-payoff kernel
program. Do not serialize C/D behind the kernel program.

Detailed guides:

- Tracks A+B: `metal_attention_kernel_guide.md`
- Track C: `mac_streaming_causal_guide.md`
- Track D: `ti2v_5b_port_guide.md`

## Suggested order of work for the Mac session

1. **Day 1 setup + quick wins:** environment (`uv pip install -e '.[dev,mlx]'`,
   `pytest fastvideo/tests/mlx/ -q` must pass on Metal), then the two
   outstanding measurements the roadmap needs: the `mx.compile` A/B
   (`--compile --assert-min-ssim 0.9`) and the checkpoint-cache load-time
   delta. Both are single benchmark commands; both numbers go into
   `apple_silicon_benchmark_baseline.md`. This validates the whole toolchain
   before any new code.
2. **Start Track C** (streaming) — biggest product payoff per unit risk, and
   it exercises the runtime knowledge needed later for the kernel program.
3. **Start Stage 0** (kernel) as the second concurrent effort once C's port
   scaffolding is committed — kernel work has long compile-measure loops that
   interleave well with C's test-driven porting.
4. **Track D** begins when either C or Stage 0 reaches its first gate, or
   immediately if run 2 ships and the 5B run becomes the program's critical
   path.

## What the DGX agent is needed for, by track

| Run | Track | What | Prereq gate | Est. cost (4×B200) |
| --- | --- | --- | --- | --- |
| 2 (in flight) | — | 1.3B QAD v2 (FastWan init, batch 16) | done | ~30 h |
| 3 | A | attn-QAT: re-run v2 recipe + attention fake-quant | Metal INT8 attention parity + torch twin bitwise gate | ~30–35 h |
| 4 | B | sparse-distill: v2 recipe + VSA (CUDA VSA at train, Metal sparse at deploy) | Metal sparse kernel parity + tile-selection equivalence gate | ~20–30 h (VSA cuts attention cost) |
| 5 | C | Self-Forcing causal QAD (1.3B) | causal MLX runtime parity + KV-cache tests | ~1–1.5 days |
| 6 | D | QAD TI2V-5B (T2V + I2V) | 5B MLX parity ladder green | ~3.5–4.5 days (or 8 GPUs ~2 days) |

Between runs the DGX agent also: exports (raw + EMA — now via the portable
EMA state; verify full-shape EMA keys in the first checkpoint of every run),
uploads transformer dirs for the Mac, and pulls W&B decile tables on request.

## Program-wide quality gates (all tracks)

1. **Numerics gates** (before GPU spend): torch twin of any deploy-time
   quantizer/kernel pinned bitwise or tolerance-pinned against the Metal
   implementation, as a pytest in `fastvideo/tests/mlx/`, running in both CI
   jobs.
2. **Parity gates** (before a kernel/runtime path is used): new path vs
   reference path on the tiny-Wan fixtures (`fastvideo/tests/mlx/tiny_wan.py`)
   with pinned tolerances and measured headroom recorded in the test.
3. **Ship gates** (before any release): the three-column Mac benchmark
   (candidate INT8 vs candidate FP16 vs stock), motion7 prompts, shared
   seeds, plus eyeball review of the HTML grids. SSIM-vs-own-FP16 measures
   only quantization robustness — run 1's EMA incident is the standing
   reminder that a constant high score can be a broken model agreeing with
   itself. Absolute quality is judged only by humans on the grids.

## Risks and standing decisions

- **MLX custom-kernel API churn**: pin the MLX version in the Mac session and
  record it in every benchmark artifact.
- **Kernel program slips**: acceptable — C and D do not depend on it; the
  release train (run-2 model, streaming demo, 5B) keeps moving on dense SDPA.
- **GPU contention**: runs 3–6 are strictly gated, so queue order is
  flexible; prefer shipping-order (2 → 5 or 6 → 3 → 4) if GPUs are scarce.
- **Upstreaming**: the FSDP dtype fix, EMA portability, and QAT callback are
  upstream-worthy now; open PRs against hao-ai-lab main early — reviewer
  goodwill is a program asset when the Mac PRs arrive.
- **Naming**: released models follow the family convention
  (`FastWan-QAD-INT8-*`, `-MLX` suffix for pre-quantized MLX checkpoints).
