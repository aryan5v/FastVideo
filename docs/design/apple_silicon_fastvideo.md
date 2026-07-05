# FastVideo on Apple Silicon

## Statement of purpose

FastVideo should make fast, local video generation practical beyond high-end
NVIDIA systems. Apple Silicon is the next important target: millions of users
already have Macs with capable neural, GPU, and unified-memory hardware, but the
software stack needs to be designed around the constraints and strengths of that
platform.

Our goal is to bring the FastVideo experience to a wide range of Apple Silicon
Macs, from 16 GB unified-memory laptops to higher-memory MacBook Pro, Mac Studio,
and Mac Pro systems. More memory should unlock higher resolution, longer clips,
and better decode quality. Smaller machines should still receive the best
quality and speed that is physically realistic through careful model choice,
quantization, scheduling, caching, and decode strategy.

This is not a one-time port. It is an ongoing optimization track. As Apple
hardware, MLX, model architectures, distillation methods, and quantization
techniques improve, the Mac path should keep improving with them.

## Why now

FastWan-QAD showed what is possible when the model, quantization strategy, and
runtime are co-designed: the recent launch generated a 5-second 480p video in
1.8 seconds on a single RTX 5090 using a two-stage recipe — a
quantization-aware finetune that matches the target-precision matmul, followed
by quantization-aware DMD distillation down to 3 sampling steps.

Apple Silicon will not reach that result by simply copying the NVIDIA path.
Blackwell tensor cores, CUDA kernels, and NVFP4-specific execution do not map
directly to Macs. The opportunity is to build the Apple-native equivalent:

- an MLX-first DiT runtime,
- memory-aware prompt encoding and decode,
- Mac-friendly quantization targets,
- a distilled/QAT model that is trained with those targets in mind,
- and benchmarks that make the quality/speed tradeoffs visible.

## Where we are (July 2026) — honest inventory

The proof of concept is real and measurable. It lives as an active Apple
Silicon branch on top of upstream main and consists of:

**Mature and tested**

- An on-device MLX DMD sampler (`fastvideo/mlx_runtime/sampling.py`) that keeps
  every large tensor on the MLX device, with unit tests covering the schedule
  lookup, `pred_noise_to_pred_video`, re-noising, and step semantics.
- MPS as a first-class inference platform (`fastvideo/platforms/mps.py`):
  platform resolution tries MPS first, capability gates make CUDA-only FP4/FP8
  paths skip cleanly, and fp16/eager compatibility overrides are tested.
- Block-level parity harnesses (synthetic and real FastWan-1.3B blocks) that
  compare the MLX Wan transformer block against the PyTorch reference within
  explicit tolerances.

**Functional but experimental**

- A full MLX Wan T2V DiT forward (`fastvideo/mlx_runtime/fastwan.py`): patch
  embed, condition/time embedding, the transformer block stack with RMSNorm
  q/k, rotary embeddings and `mx.fast.scaled_dot_product_attention`, and
  unpatchify — loading Diffusers-format safetensors directly.
- An end-to-end hybrid pipeline
  (`examples/inference/basic/mlx_wan_prompt_to_video.py`): UMT5 prompt encoding
  on torch, MLX DiT denoising with 3-step DMD, torch VAE or TAEHV decode.
- Load-time weight quantization for the DiT linears: INT8 (affine, group size
  64), INT4, and MXFP8/MXFP4/NVFP4-style modes where the installed MLX supports
  them. INT8 is currently the most reliable quality/memory point.
- A benchmark harness (`fastvideo/benchmarks/mlx_fastwan_bench.py`) sweeping
  quantization mode × decoder × memory tier, measuring load/denoise/decode
  latency, MLX peak memory, MS-SSIM (optionally LPIPS) against an FP16
  reference, with an `--assert-min-ssim` regression gate,
  `mac-16gb`/`mac-32gb`/`mac-64gb` presets, `metrics.json`/`metrics.md`, and
  side-by-side HTML outputs.
- Opt-in `mx.compile` of the DiT forward and fused MLX norm kernels, both off
  by default and gated behind environment variables.

**16 GB memory work already done**

- Prompt-encoding isolation: encode, move embeds to CPU, free the text encoder
  before the DiT loads — inline, or in a separate subprocess so the OS reclaims
  everything, plus an on-disk embedding cache.
- Shared memory-tier controls for benchmarks and one-off generation:
  MLX allocator/cache/wired-memory caps plus PyTorch MPS watermark settings.
- TAEHV tiny-VAE decode as the low-memory alternative to the full Wan VAE.
- Only matrix weights are quantized; norms and modulation tables stay fp16.

**Fragile or thin**

- MXFP8/MXFP4/NVFP4 support still depends on the installed MLX build and Apple
  hardware, but unsupported modes now fail early with an actionable message
  instead of raising deep inside model load.
- TAEHV code is vendored, but the checkpoint download/cache path still needs
  the same fresh-machine polish as the rest of the installer story.
- CI coverage is moving from CPU-golden to on-device: a macOS arm64 MLX smoke
  workflow now runs the sampler, memory helpers, quantization capability probe,
  and tiny full-DiT parity tests for PRs touching the MLX runtime. The next
  step is watching its first upstream PR runs and tightening the test set only
  after it proves stable on hosted Apple runners.
- The public user surface is still example/benchmark driven; the production
  `fastvideo generate --preset mac-*` CLI should wait until the MLX path is
  wired through the normal pipeline abstractions.

**Explicitly missing**

- Quantization-aware training for any Mac-relevant precision. The repository's
  only QAT path (`fastvideo/layers/quantization/nvfp4_qat_config.py` and the
  `attn_qat_*` backends) is NVFP4, flashinfer, and Blackwell-only. There is no
  INT8 quantization method in the layer registry at all.
- Sparse attention on Mac. VSA and SLA are CUDA/Triton kernels in
  `fastvideo-kernel/`; the MPS platform hard-codes dense torch SDPA, and the
  MLX path uses dense `mx.fast` SDPA. Mac models must target FullAttn/dense
  variants.
- Pre-quantized checkpoint *distribution*. The runtime side now exists
  (`fastvideo/mlx_runtime/checkpoint.py` saves/loads a cast + quantized DiT so
  reloads skip requantization; `--save-mlx-checkpoint`/`--mlx-checkpoint` in
  the generation example), but no pre-quantized weights are published on
  Hugging Face yet, and load-time/download-size targets are unmeasured.
- Image-to-video, multi-device, and a real CLI surface (the Mac path is driven
  by example scripts).

The important result is not that the current videos are final quality. They are
not. The important result is that the pipeline is now real enough to measure,
compare, and improve.

## Product target

The Apple Silicon path should eventually expose memory-aware presets rather than
one brittle configuration:

- **16 GB Macs:** accessible local generation with smaller resolution/clip
  length, INT8 or better quantized DiT weights, prompt-cache/freeing, and TAEHV
  decode by default.
- **24-36 GB Macs:** better resolution, longer clips, optional higher-quality
  decode, and more room for FP16/INT8 comparisons.
- **64 GB+ Macs:** higher quality presets, longer clips, more model families,
  stronger decode options, and broader benchmarking coverage.

Every tier should aim for the same principle: use the available unified memory
intelligently, keep the runtime responsive, and avoid hiding quality regressions
behind raw speed numbers.

## Strategy pillars

1. **Co-design, not port.** FastWan-QAD worked because the model, quantization,
   and runtime were designed together for Blackwell. The Mac equivalent is a
   QAT model whose train-time fake-quantization bit-matches the deploy-time MLX
   quantizer (affine INT8, group size 64, matrix weights only). This numerics
   parity is a hard, tested requirement — if training simulates a different
   quantizer than the one MLX applies at load, the QAT gains evaporate at
   deploy time. No GPU spend on training until the parity test passes.
2. **Train on NVIDIA, deploy on Mac.** Distillation runs on rented NVIDIA GPUs
   using the existing modular trainer (`fastvideo/train/`: `DMD2Method`,
   `KDMethod`, and the working `examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml`
   config), exported via `fastvideo/train/entrypoint/dcp_to_diffusers.py` into
   Diffusers safetensors the MLX loader already reads. MPS/MLX training is a
   non-goal this cycle.
3. **Dense-first.** No VSA/SLA port to Metal in this window. The Mac targets
   are the FullAttn model variants. A Metal/MLX sparse-attention kernel is a
   parked stretch goal, revisited only after the QAT model ships.
4. **Benchmark-driven.** No optimization lands without a benchmark delta, and
   the SSIM regression gate is enforced on every runtime change. Speed that
   breaks quality does not count as progress.

### Two lanes: FastVideo-in-general vs the MLX fast lane

The goal is bringing FastVideo — not only FastWan-QAD — to Macs. That happens
on two lanes with different economics:

- **Compatibility lane (torch-MPS).** `MpsPlatform` is model-agnostic: any
  FastVideo pipeline whose components fit in unified memory can run through
  the normal pipeline abstractions with dense SDPA, fp16, and eager execution.
  This lane inherits every model family the repo supports (Wan 2.1/2.2,
  HunyuanVideo, LTX-2, …) at whatever speed torch-MPS delivers. Work here is
  general: memory-tier controls, decode strategies, prompt-encoder freeing,
  and CI all apply across families. Each family still needs a one-time
  verification pass (op coverage, dtype quirks, memory fit), which is why the
  benchmark suite tracks model coverage explicitly.
- **Fast lane (MLX).** The MLX DiT runtime is per-architecture by design —
  today it implements the Wan T2V transformer. Wan is the right first target
  because FastWan's 3-step DMD models make consumer-Mac latency plausible at
  all. Extending the fast lane to another family (Wan 2.2 TI2V-5B is next per
  M6; HunyuanVideo/LTX-2 later) is a bounded port: block parity harness →
  full-DiT parity test → benchmark cells, the same ladder Wan followed. The
  parity/benchmark/checkpoint infrastructure built for Wan is family-agnostic
  and is the reusable part.

Practically: FastWan-QAD-style models are the flagship demo of the fast lane,
while the compatibility lane is what makes this "FastVideo on Mac" rather than
"one model on Mac". Both lanes ship through the same presets, benchmarks, and
docs.

## Milestones and exit criteria (July–November 2026)

Each milestone is done only when every exit criterion is met. Criteria are
deliberately measurable so "done" is not a judgment call.

### M1 — Trustworthy baseline (Month 1)

Harden what exists so everything after it can be trusted.

Exit criteria:

- **[done]** A full-DiT MLX-vs-PyTorch parity test with pinned tolerances
  exists and is runnable in CI (CPU-golden variant) and on-device:
  `fastvideo/tests/mlx/test_mlx_dit_parity.py` (fp32 tolerance 2e-4, measured
  ~1.7e-6 on the MLX CPU backend; int8 gated at ≥20 dB SNR, measured ~43 dB).
  `CpuPlatform` now selects Torch SDPA so the reference model also runs on
  CPU-only machines via `mlx[cpu]`.
- **[measured under a cap; stock 16 GB machine still pending]** End-to-end
  prompt→video succeeds on a stock 16 GB M-series Mac at a pinned
  configuration (3-step DMD, INT8, TAEHV, 448×832×61), with peak memory and
  wall time recorded. First real-device numbers now exist (M4 Max, MLX
  allocator capped to 16 GiB: INT8+TAEHV 480×832×81 passes at 4.69 GiB MLX
  peak — see `apple_silicon_benchmark_baseline.md`); a run on an actual 16 GB
  machine remains the closing evidence.
- **[done]** MXFP/NVFP4-style modes detect MLX capability and fail with a
  clear message instead of raising deep inside `mx.quantize`
  (`ensure_quantization_supported`; both benchmark sweeps record
  `unsupported_by_mlx` rows and keep going).
- **[done]** TAEHV is vendored (`fastvideo/third_party/taehv`, MIT) — no
  downloading and executing remote code at runtime; the checkpoint download
  is sha256-pinned.
- **[done]** `mlx_wan_quant_benchmark.py` uses the on-device DMD sampler.
- **[blocked on fork sync]** The branch is rebased onto current upstream main
  — the fork's `main` currently predates this branch's base, so there is
  nothing newer to rebase onto until the fork syncs with hao-ai-lab main.

### M2 — Benchmark suite as a product surface (Months 1–2)

Exit criteria:

- One command sweeps quantization modes × decoders × memory tiers over a
  standard prompt set (prompts with visible motion and physics) and emits the
  MP4s, a side-by-side HTML grid, and `metrics.json`/`metrics.md`.
  Initial benchmark presets exist; the next step is checking in baseline
  reports from real 16 GB / 32 GB / 64 GB machines.
- Quality is measured two ways: fidelity to the FP16 reference (MS-SSIM/LPIPS)
  and a reference-free score (a VBench-style subset), because
  fidelity-to-reference alone cannot detect a bad reference.
- Latency is split into load/denoise/decode with cold-start vs warm-start, and
  sustained vs burst throughput is recorded on laptops (thermal throttling is
  real on fanless machines). Partially landed: the benchmark now records
  first-step vs steady-step denoise time (exposes `mx.compile` warm-up) and
  `load_source` (Diffusers conversion vs pre-quantized checkpoint reload via
  `--mlx-checkpoint-cache`); thermal/sustained measurement remains open.
- A macOS arm64 CI job runs an MLX-device smoke test on every PR touching
  `fastvideo/mlx_runtime/`. Initial workflow exists; first hosted-run results
  should be recorded before treating this as fully closed.
- A baseline report for at least one 16 GB and one 64 GB machine is checked
  into the repository. A first M4 Max baseline (including a 16 GiB-capped
  stress run) is in `apple_silicon_benchmark_baseline.md`.
- Model coverage beyond Wan: at least one additional FastVideo family (e.g.
  Wan 2.2 TI2V-5B or LTX-2) verified end to end on the torch-MPS
  compatibility lane, with its result recorded in the baseline report — this
  keeps "FastVideo on Mac" honest, not just "FastWan on Mac".

### M3 — Runtime hardening and UX (Months 2–3)

Exit criteria:

- `mx.compile` is on by default, with the SSIM gate proving no quality
  regression and the step-time improvement published in the benchmark report.
- Zero host/device transfers inside the denoise loop, asserted by test.
- **[partially done]** Pre-quantized MLX checkpoints can be saved and loaded:
  16 GB users download INT8 weights (roughly half the bytes) and skip
  requantization on every run; load-time and download-size targets recorded.
  Save/load with exact round-trip tests landed
  (`fastvideo/mlx_runtime/checkpoint.py`); remaining: publish pre-quantized
  weights and record on-device load-time/size numbers.
- A real CLI replaces the example scripts:
  `fastvideo generate --preset mac-16gb|mac-32gb|mac-64gb`, wired through the
  existing pipeline and platform abstractions.
- The decode story is decided per tier by benchmark: TAEHV vs chunked/tiled Wan
  VAE (torch-MPS) vs an MLX-native decoder, whichever wins the quality/memory
  point for that tier.
- Install docs verified on a fresh machine: under 15 minutes from clone to
  first video.

### M4 — Mac-targeted QAT distillation, 1.3B (Months 2–4, parallel track)

Runs on NVIDIA cloud GPUs in parallel with M3.

Phase A — numerics first:

- **[numerics gate passed]** A portable INT8 fake-quant module whose forward
  bit-matches MLX's quantizer (affine INT8, group size 64). Landed as
  `fastvideo/layers/quantization/mlx_affine_qat.py` — a pure-torch transcription
  of MLX's CPU `quantize` kernel (v0.31.2), including the negative-scale
  anchoring and integer zero-point re-fit — with
  `fastvideo/tests/mlx/test_mlx_affine_qat_parity.py` pinning codes, scales,
  biases, and dequantized values **bitwise** against `mx.quantize` /
  `mx.dequantize` for int8 and int4, fp32 and fp16, plus an STE gradient test
  and a quantized-matmul tolerance check. This was the gate before GPU spend.
- **[landed]** The training-side wrapper: `mlx_qat` is a builtin callback
  (`fastvideo/train/callbacks/mlx_qat.py`) that registers a weight
  parametrization on the student transformer — every forward sees the exact
  MLX deploy grid, gradients flow straight-through to master weights — and
  composes with any method via YAML. A ready-to-run recipe exists at
  `examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml`
  (Wan2.1-1.3B teacher → 3-step INT8 student, FastWan timesteps
  [1000, 757, 522], 4 GPUs). First multi-GPU smoke run must verify the
  parametrizations interact cleanly with HSDP sharding.

Phase B — the run:

- **[smoke passed on DGX B200, 2026-07]** The QAD smoke run (100 steps,
  4 GPUs, `dmd2_t2v_mlx_int8.yaml` per `apple_silicon_qad_runbook.md`)
  trained and validated end to end with the student computing every forward
  on the INT8 deploy grid; step-100 validation clips show coherent subjects
  emerging. Getting here surfaced and fixed three real integration bugs:
  a missing `torch.distributed.checkpoint` import, denoising stages sniffing
  fp32 FSDP master storage for their target dtype (broke base-recipe
  validation too), and the QAT weight parametrization mixing sharded
  DTensors with unsharded tensors (replaced with a forward-scoped weight
  swap that leaves modules vanilla outside their own forwards).
- **[training complete, 2026-07]** Quantization-aware DMD of
  Wan2.1-T2V-1.3B (dense attention) to 3 steps: 4,000 steps on 4×B200 in
  8h52m (~6.8 s/step steady), losses finite throughout, step-4000
  validation clips near teacher-class per frame. Raw and EMA students
  exported to Diffusers format (`dcp_to_diffusers`, incl. the new `--ema`
  path). **Mac evaluation passed the M4 criterion**: INT8-vs-own-FP16
  MS-SSIM 0.9860 (QAD EMA) / 0.9487 (raw) vs 0.9069 (stock PTQ), EMA
  ahead on all seven motion7 prompts at zero runtime cost — see
  `apple_silicon_benchmark_baseline.md`. Remaining for M4 close-out:
  visual parity sign-off vs stock FP16, then publish the EMA weights +
  pre-quantized MLX checkpoint to Hugging Face.
- Export: DCP checkpoint → `dcp_to_diffusers.py` → Diffusers safetensors →
  existing MLX loader, plus the M3 pre-quantized MLX format.

Exit criteria:

- The INT8-QAT model beats INT8 post-training quantization on the full M2
  benchmark suite and is within thresholds of FP16 quality (thresholds set
  from M2 baselines before training starts, not after).
- Weights published on Hugging Face under the FastVideo org.
- Only after INT8 proves out: evaluate INT4/MXFP4 QAT for the 16 GB tier.
  Guardrail: INT4 is not attempted before the INT8 exit criteria are met.

### M5 — 16 GB flagship preset (Month 4)

Exit criteria:

- A documented preset runs on stock 16 GB M2/M3/M4 machines (not just top-bin
  development hardware): peak unified memory at or below ~14 GB with no swap
  collapse, a 5-second 480p-class clip within a stated wall-time budget, using
  the INT8-QAT model from M4.
- Reproduced by someone outside the core team following the docs alone.

### M6 — Higher-memory tiers and TI2V-5B (Months 4–5)

Exit criteria:

- The M4 recipe repeated on FastWan2.2-TI2V-5B-FullAttn, adding image-to-video
  and anchoring the 32 GB+ quality tiers.
- A per-tier preset table (model, resolution, frame count, decode mode,
  expected time and peak memory) validated on real 16/32/64 GB hardware.
- Higher-resolution, longer-clip, and full-VAE-decode presets for 64 GB+.

### M7 — Public Mac support story (Month 5)

Upstreaming starts much earlier — platform and runtime pieces go up for review
from M1 onward, not as one giant drop at the end.

Exit criteria:

- The work is merged into hao-ai-lab main.
- A reproducible demo and a benchmark article with honest tradeoffs (the
  FastWan-QAD launch post is the template).
- Setup docs and a contributor guide for adding models, prompts, metrics, and
  Apple-specific optimizations.

## Staying on track

The failure mode for a project like this is months of interesting kernel work
with no shipped preset. The countermeasures are structural:

- **Weekly benchmark runs** on a fixed hardware pool, with deltas recorded in a
  running journal (`.agents/memory/experiment-journal/` exists for exactly
  this and is currently empty).
- **Every milestone ends with a demo artifact** — a video, a report, a preset a
  new user can run — not just merged code.
- **Decision gates are written down, not re-litigated:** INT8 before INT4;
  dense before sparse; no Mac-training track; no Apple Neural Engine/CoreML
  detour this cycle.
  Each non-goal carries a revisit date instead of an open-ended debate.
- **Risk register**, reviewed monthly:
  - MLX API churn, especially the MXFP/NVFP4-style quantization modes — pin
    MLX versions and gate by capability.
  - Hardware variance (M1 through M4, fanless vs active cooling) — benchmark on
    the low end, not only the machines we happen to own.
  - Upstream divergence — main is moving fast (Attn-QAT, FP4 linear paths);
    rebase on a fixed cadence and upstream early.
  - GPU spend wasted on wrong numerics — the parity test gates Phase B.
  - Quality-metric blind spots — SSIM against a reference misses "the reference
    is bad"; the VBench-style score and a small human evaluation at M4/M5
    catch it.

## Differentiation

What makes this stand out rather than being "Wan, but slower, on a Mac":

- The first 3-step DMD video generation co-designed for consumer Apple
  Silicon — quantization-aware training matched to the deploy-time MLX
  quantizer, the Apple-native analog of FastWan-QAD's NVFP4 co-design.
- Memory-tier presets as a product surface: a 16 GB MacBook Air owner and a
  128 GB Mac Studio owner both get a configuration that is honest about what
  their machine can do.
- Public, reproducible benchmarks with quality metrics attached to every speed
  claim.

Parked stretch ideas, with rationale recorded so they are deliberate choices
rather than forgotten ones: a Metal/MLX sparse-attention kernel (after the QAT
model ships), step-output caching (TeaCache-style), and packaging (a ComfyUI
node — the repository already carries `comfyui/` — or a small native app).

## Operating principle

We should optimize for honest progress. Fast videos that look broken are not the
goal. Beautiful videos that take too long are also not the goal. The work is to
find the best quality-speed-memory point for each Mac tier, make it reproducible,
and keep pushing that frontier forward.
