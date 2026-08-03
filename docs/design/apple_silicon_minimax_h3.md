# Apple Silicon Plan: MiniMax-H3 Local Video Generation

Status: design, pre-implementation. Depends on the `minimax_h3` CUDA port
(`tests/local_tests/minimax_h3/PORT_STATUS.md`).

This is the **local/Metal track**. It is deliberately separate from upstream
FastVideo H3 support, which is a plain CUDA model port. Nothing here should be
folded into that PR.

Goal: the highest-quality local video generation that runs at interactive speed
on Apple Silicon — not the largest model that technically loads.

## 1. Design thesis: decouple the quality axes

The instinct is to fit H3 on the Mac and turn everything down until it fits.
That spends the entire memory budget on axes the viewer barely scores.

Perceived quality of a 5–15s clip is dominated by motion coherence, prompt
adherence, and temporal stability. Spatial resolution and frame rate are
comparatively cheap to buy back after generation, and on Apple Silicon they are
*very* cheap — MetalFX is a fixed-function-adjacent spatial upscaler and RIFE
interpolation is a small convnet.

So the budget split is:

| Axis | Bought with | Cost |
|---|---|---|
| Motion, coherence, prompt adherence | denoiser capacity + step count | the entire weight budget |
| Spatial resolution | MetalFX upscale, skip in-context regeneration | near-free |
| Frame rate | RIFE interpolation, generate fewer frames | near-free |
| Audio | H3 audio VAE decode | small |

`fastvideo/benchmarks/eval_metalfx_rife.py` already measures the
fewer-frames-plus-interpolate trade against MS-SSIM. That harness is the
template for the resolution trade too.

H3 makes this unusually clean. In-context regeneration means the base model
*natively* produces a low-resolution result and then re-reads context to
regenerate at high resolution. Skipping the second pass is a supported mode of
the architecture, not a hack, and it roughly halves per-clip compute.

The second H3 property that matters: Contextual Omni Representation compresses
the multimodal context to roughly 4k tokens. Video models normally die on Metal
because of sequence length — attention is where MLX is weakest relative to
CUDA. At 4k tokens, attention is not the bottleneck. **Weight memory is the
only real constraint.** That is a favorable shape for Apple Silicon.

## 2. Quantization: two deploy grids, selected by chip generation

### Why MX looked worse — and what that does and does not predict

Observed in practice on the Wan track: mxfp8 and mxfp4 both produced visibly
worse output than affine int8. This is a property of the *numeric grid*, not of
MLX or of the hardware:

- **No zero point.** Affine stores a scale *and* a bias per group, so it fits an
  asymmetric group range exactly. MX formats carry only a shared power-of-two
  (E8M0) scale and cannot recenter, wasting code space whenever a group's weight
  distribution is off-center.
- **Power-of-two scales.** Restricting the group scale to a power of two gives
  up to a full bit of effective range against an arbitrary fp16 scale.
- **Logarithmic element spacing.** E4M3 has 3 mantissa bits, so ~6% relative
  step size. Affine int8 gives 256 *uniform* levels across the group range. For
  roughly Gaussian weights, uniform spacing wins in the bulk where the mass is.

The important consequence: **newer silicon makes MX fast, not accurate.** The
grid deficits above are arithmetic and do not improve when the format gets a
hardware path. Anyone expecting M5 to fix MX output quality on its own will be
disappointed.

What *does* fix it is targeting the MX grid during QAT instead of the affine
grid. Which grid you train against must match the grid you deploy on — a model
QAT'd onto the affine grid and then deployed as mxfp8 reintroduces exactly the
error QAT existed to remove, and will look worse than either path done
consistently. This is the single easiest way to get a confusing bad result.

### Generation-dependent target

| Silicon | Deploy grid | Rationale |
|---|---|---|
| M1–M4 | affine int8, `group_size=64`, mixed precision | No hardware path for microscaled formats; MX costs quality and buys no throughput. At `gs=64` affine int8 is 8.5 bits/param against mxfp8's 8.25 — better quality for a third of a bit. |
| M5 and newer | mxfp8 for attention + FFN, with **MX-grid QAT**; mxfp4 for FFN only if metrics hold | M5 adds per-core GPU Neural Accelerators for matrix work and Metal 4 exposes tensor primitives, so microscaled formats get a real throughput path. Worth the grid deficit once QAT is aimed correctly. |

This makes `hardware_tier.py` **generation-aware, not just memory-aware**. Today
it reads `hw.memsize` and nothing else. It needs to detect chip generation and
Metal 4 / neural-accelerator availability and select the deploy grid from that,
independently of the memory band that selects model size.

Concretely this means shipping two quantized artifacts per student — an affine
int8 build and an mxfp8 build — from two QAT runs, not one build requantized. The
checkpoint manifest in `mlx_runtime/checkpoint.py` already records the
quantization spec, so it can distinguish them; the loader needs to refuse a grid
mismatch loudly rather than silently dequantizing.

### Mixed precision is the highest-leverage unshipped change

Applies to both grids. Not all DiT weights deserve the same treatment. In a DiT,
the AdaLN/modulation projections emit scale and shift terms that multiply entire
activation tensors, so error there propagates multiplicatively; FFN weights only
contribute additively. Same story for patch embedding and the final output
projection, which sit at the boundaries where there is no downstream layer left
to absorb error.

Keep in fp16: modulation/AdaLN projections, time-step embedding MLP, patch
embed, final output projection, all norm affines. These are a low single-digit
percentage of parameters and they dominate quantization error.

Quantize: attention QKV/out projections and FFN matrices — the overwhelming
majority of the weights.

For layers that still show drift on the affine grid, drop to `group_size=32`
(9 bits/param) before considering fp16.

### Why QAT is load-bearing here and not for LLMs

An LLM samples a token, and quantization error is partly absorbed by the
sampling temperature. A diffusion denoiser feeds its own output back in for
every step, so weight error compounds across the schedule. Post-training
quantization of a video DiT looks fine at step 1 and drifts visibly by step N.

`fastvideo/layers/quantization/mlx_affine_qat.py` fake-quantizing onto the exact
MLX affine deploy grid during distillation is what makes deployed Metal weights
hold up. It needs a sibling that fake-quantizes onto the MX grid — shared
callback, swappable grid — for the M5 build. Step distillation helps twice over:
a 3-step schedule gives error three chances to compound instead of fifty.

### Mixed precision is the highest-leverage unshipped change

Not all DiT weights deserve the same treatment. In a DiT, the AdaLN/modulation
projections emit scale and shift terms that multiply entire activation tensors,
so error there propagates multiplicatively; FFN weights only contribute
additively. Same story for patch embedding and the final output projection,
which sit at the boundaries where there is no downstream layer left to absorb
error.

Keep in fp16: modulation/AdaLN projections, time-step embedding MLP, patch
embed, final output projection, all norm affines. These are a low single-digit
percentage of parameters and they dominate quantization error.

Quantize to affine int8: attention QKV/out projections and FFN matrices — the
overwhelming majority of the weights.

For layers that still show drift, drop to `group_size=32` (9 bits/param) before
considering fp16.

### Why QAT is load-bearing here and not for LLMs

An LLM samples a token, and quantization error is partly absorbed by the
sampling temperature. A diffusion denoiser feeds its own output back in for
every step, so weight error compounds across the schedule. Post-training
quantization of a video DiT looks fine at step 1 and drifts visibly by step N.

`fastvideo/layers/quantization/mlx_affine_qat.py` fake-quantizing onto the exact
MLX affine deploy grid during distillation is what makes deployed Metal weights
hold up. Step distillation helps twice over: a 3-step schedule gives error three
chances to compound instead of fifty.

## 3. The capacity decision

The one genuinely hard question, and it needs answering before GPU time is
spent.

Weight-memory ceilings on a Mac, affine int8 at 1.06 GB per billion params (int4
at 0.56, quality permitting):

| Unified memory | Realistic MLX cap | DiT weight budget | int8 ceiling |
|---|---|---|---|
| 24 GB | ~14 GiB | ~10 GiB | ~9B params |
| 32 GB | ~20 GiB | ~15 GiB | ~14B params |
| 64 GB | ~44 GiB | ~36 GiB | ~34B params |
| 128 GB | ~96 GiB | ~85 GiB | ~80B params |

If H3 is MoE — every recent MiniMax model is — then *total* parameters drive
memory, not active ones, and the ceilings above bite far harder than the
active-parameter count suggests. Resolve this from `config.json` before
committing to a track.

Three paths, in order of preference:

**A. H3 ships a small variant.** Step-distill and QAT it directly. Weeks of
work, reuses the existing Wan recipe almost unchanged. Check for this first.

**B. Two-tier ship (recommended default).** Do not compromise the flagship to
fit the smallest Mac. Run H3 itself at int8 on 64/128 GB Macs — it will be fast
there, because 4k context plus a distilled 3-step schedule is a small compute
job — and ship a distilled student for 24–32 GB. This answers "highest possible
quality" honestly: the highest quality is H3 itself on a machine that fits it.

**C. Capacity distillation into a small student.** Only if A is unavailable and
B is unacceptable. Prefer *depth* pruning (drop blocks on a schedule, then
distill to recover) over width pruning — DiTs tolerate depth reduction better,
and it preserves per-layer weight shapes so the existing conversion mapping
survives. Budget this as a multi-week GPU project with a genuinely uncertain
quality outcome, not as an engineering task.

Under B and C, run capacity reduction and step+precision distillation as
**separate stages** — prune/recover first at full step count in bf16, then DMD2
plus QAT on the recovered student. Folding all three objectives into one run
sounds efficient and in practice makes failures unattributable.

## 4. Tracks

### Track A — CUDA port (prerequisite)

Standard `add-model` work, tracked in `tests/local_tests/minimax_h3/`. Required
before anything below: the trainer loads the teacher through FastVideo model
classes, and MLX parity tests compare against the torch modules.

### Track B — Distillation on GPU

Recipe base: `examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml`,
teacher swapped to H3. Operator flow follows the existing QAD runbook.

New for H3:

- **Audio.** The student must keep the audio branch, which needs its own loss
  term. This is the piece with no precedent in the Wan work, and it is also the
  differentiator — local audio-video generation is something nothing else ships.
- **Multimodal conditioning.** Distillation data needs the image/video/audio
  reference paths exercised, not just text prompts, or I2V and editing quality
  will collapse in the student.
- **Regeneration pass.** Distill the base pass only. The student does not need
  to learn in-context regeneration if the Mac path never runs it.

### Track C — MLX runtime

New `fastvideo/mlx_runtime/minimax_h3.py`, roughly the scale of `fastwan.py`
plus `wan22.py`. Reused unchanged: `quantize_matrix`/`linear`, the
`mlx_dit.safetensors` checkpoint format, `memory.py`, `hardware_tier.py`, the
`mx.compile` harness with eager fallback.

New work: the H3-Omni block structure, the multimodal token packing and per-
modality embeddings, and an **MLX audio VAE decode path** — `taehv_decode.py`
and `wan_vae.py` are video-only.

`hardware_tier.py` gains an H3 band set **and a second selection axis** (§2):
memory band picks the model, chip generation picks the deploy grid. The current
thresholds (≤18 GiB → 1.3B int8 + TAEHV, 12 GiB cap) stay valid for the Wan
tiers and should not be retuned to accommodate H3.

Metal 4 also matters to the runtime beyond quantization. Its tensor primitives
and inline ML command encoding let the postprocess chain (§Track D) run in the
same command stream as decode instead of round-tripping through separate
dispatches. Treat that as a Track C follow-up, gated on measurement — it is a
latency win, not a quality one.

### Track D — On-device quality recovery

- Generate at base resolution; never run in-context regeneration on device.
- MetalFX spatial upscale to target resolution.
- Temporal upsampling from a reduced frame count. Two candidates: the existing
  MLX RIFE path, and **MetalFX frame interpolation** on Metal 4, which is
  hardware-assisted and should be materially cheaper. Bench both — RIFE is
  already integrated and known-good, so it stays the fallback for pre-Metal-4
  machines regardless of how the comparison lands.
- TAEHV-class decoder for interactive preview, full H3-VAE for final render.

Extend `eval_metalfx_rife.py` to sweep the resolution trade alongside the
existing frame-count trade, and to cover the MetalFX-interpolation arm. Pick the
operating point from measurement.

## 5. Quality gates

Measure the student against the **CUDA H3 reference**, not against an fp16 MLX
build, so distillation and quantization error are scored together. The existing
`fastvideo/eval/` framework covers most of this:

| Axis | Metrics |
|---|---|
| Motion / coherence | `vbench.subject_consistency`, `vbench.motion_smoothness`, `vbench.temporal_flickering`, `vbench.dynamic_degree` |
| Fidelity | `common.fvd`, `common.lpips`, `vbench.imaging_quality` |
| Prompt adherence | `vbench.overall_consistency`, `videoscore2` |
| Audio | `audio.clap_score`, `audio.frechet_distance` |
| AV sync | `audio.desync` |

`audio.desync` is the one that will catch a bad joint-AV distillation, and it
has no analogue in the video metrics. Gate on it explicitly.

SSIM against the teacher is the wrong headline metric for a distilled student —
the student is *supposed* to diverge. Use SSIM only as a regression tripwire for
the MLX runtime against its own torch reference, which is what
`fastvideo/tests/ssim/` and `test_mlx_dit_parity.py` are for.

## 6. Sequencing

1. Read `config.json`. Resolve dense-vs-MoE and total parameter count. Pick
   path A, B, or C from §3. **Everything downstream depends on this.**
2. Land the CUDA port (Track A).
3. Mixed-precision QAT sweep on the *existing* Wan student, before H3 weights
   are involved. Validates §2's layer-skip list on a model already known-good.
4. MX-grid QAT sweep on that same Wan student, benched on M5 hardware against
   the affine int8 build — quality from §5 metrics, throughput from
   `mlx_fastwan_bench.py`. This is what decides whether the two-grid split in §2
   earns its complexity.
5. Track B distillation, gated on §5 metrics.
6. Track C runtime, parity against Track A.
7. Track D operating-point sweep, including the MetalFX-interpolation arm.

Steps 3 and 4 are out of order on purpose. Both are cheap, both run on a model
that already works, and neither blocks on H3 weights being reachable. Step 4 in
particular should happen before any H3 GPU time is committed — if MX-grid QAT
does not close the quality gap on Wan, it will not close it on H3 either, and
the M5 path collapses back to affine int8 with no loss to the plan.

## 7. Open questions

| ID | Question | Blocks |
|---|---|---|
| M001 | Is H3 dense or MoE, and what is the total parameter count? | §3 path choice, everything |
| M002 | Does a small H3 variant exist or is one planned? | §3 path A |
| M003 | Is the audio branch a separate head on the omni transformer or a distinct module? | Track B audio loss, Track C decode |
| M004 | Can in-context regeneration be skipped cleanly, or does the base pass assume it? | Track D, §1 thesis |
| M005 | What does the text/omni encoder cost in memory, and can its output be precomputed and freed before denoising? | §3 budgets |
| M006 | Does the installed MLX build route mxfp8/mxfp4 through M5 neural accelerators, or still emulate? `quantization_support_error()` probes correctness, not whether a hardware path was taken — it will pass either way. | §2 M5 grid |
| M007 | How is chip generation / Metal 4 / neural-accelerator availability detected? `hw.memsize` is not enough; `hw.optional.*` sysctls or an MLX device-info field need checking. | `hardware_tier.py` |
| M008 | Does MetalFX frame interpolation accept the frame cadence a 3-step diffusion sampler emits, or does it assume game-engine motion vectors? | Track D |
