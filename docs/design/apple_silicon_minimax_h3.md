# Apple Silicon Plan: MiniMax-H3 Local Video Generation

Status: design, pre-implementation. Depends on the `minimax_h3` CUDA port
(`tests/local_tests/minimax_h3/PORT_STATUS.md`).

This is the **local/Metal track**. It is deliberately separate from upstream
FastVideo H3 support, which is a plain CUDA model port. Nothing here should be
folded into that PR.

Goal: the highest-quality local video generation that runs at interactive speed
on Apple Silicon — not the largest model that technically loads.

## 0. Decisions

| Question | Decision |
|---|---|
| Primary quantization | **mxfp4 with MX-grid QAT, M5 and newer.** This is the strategic bet, not the safe one (§2). |
| Fallback artifact | Affine int8, produced from the same student by one extra B2 run. Serves M1–M4 and covers the bet failing. Cheap; not the headline. |
| Memory target | **24 GB floor, 32–36 GB comfortable** (§3). Not 96 GB. |
| Student capacity | **~14B class** (§3). Chosen because mxfp4 puts 14B in ~7.4 GB — it fits 24 GB with room — and because quantization tolerance rises with size. |
| Release shape | **One student, two step-distillations**: Turbo (rapid) and Quality. Shared expensive stages, cheap divergence (§3). |
| Validation before commitment | Three experiments, days not months, on existing models (§6). The bet is gated on these, not on faith. |
| Expected H3 size | **Unknown (M001).** Ceilings in §3 say what fits, not what H3 is. |

Deliberate consequence: pre-M5 Macs get the int8 artifact and are not the design
target. That is an accepted cost of leading with mxfp4.

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

## 2. Quantization: lead with mxfp4 on M5, keep int8 as fallback

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

### What the existing measurements actually say

Two observations from the Wan track, and they point in the same direction:

- At **1.3B**, both mxfp8 and mxfp4 looked bad.
- At **14B** on a 36 GB machine, int8 was good and mxfp4 ran fine but looked
  bad. (Whether mxfp8 held at 14B needs confirming — M009. If it did, that
  sharpens the story considerably.)

Both were **post-training** quantization. Neither had QAT aimed at the MX grid,
because that code does not exist yet. So the honest read is not "MX formats are
bad" — it is "MX formats PTQ badly," which is exactly what the grid analysis
above predicts and exactly what QAT exists to fix.

The second signal is that quantization tolerance rises with model size. A 14B
has redundancy a 1.3B does not, and 4-bit formats lean on that redundancy hard.
This cuts against shrinking the model to fit, and is a large part of why §3
targets 14B rather than something smaller.

### The bet

**mxfp4 with MX-grid QAT, on a 14B-class student, targeting 24–32 GB.**

The arithmetic is what makes this worth doing. At 4.25 bits/param, 14B costs
~7.4 GB — against ~15 GB at int8 and ~28 GB at bf16. That is the difference
between a 14B-class model being a 36 GB-machine proposition and being a 24 GB
one. Running a model that size on median hardware is the actual game-changer;
int8 cannot get there.

Three things carry the quality:

1. **MX-grid QAT.** The core bet. `mlx_affine_qat.py` needs an MX-grid sibling —
   shared callback, swappable grid. Nothing existing PTQ'd to mxfp4 has had this.
2. **Mixed precision within the MX build.** Keep AdaLN/modulation, timestep
   embedding, patch embed, final projection, and norm affines at bf16 or mxfp8.
   Low single-digit percent of parameters, disproportionate share of the error.
3. **Few steps.** A 2–4 step schedule gives 4-bit error far fewer chances to
   compound than a 50-step one.

If the bet fails, the same B1 student takes one extra B2 run to int8 and ships
against the §3 ceilings at reduced reach. That fallback is days of GPU time, not
a restart — which is what makes leading with mxfp4 a reasonable risk rather than
a gamble.

### Runtime consequences

`hardware_tier.py` becomes **generation-aware, not just memory-aware**. Today it
reads `hw.memsize` and nothing else; it needs chip generation and Metal 4 /
neural-accelerator detection, selecting the deploy grid independently of the
memory band that selects model size.

Ship two artifacts per student — mxfp4 and affine int8 — from two QAT runs, never
one build requantized. `mlx_runtime/checkpoint.py` already records the
quantization spec in its manifest, so it can distinguish them; the loader must
refuse a grid mismatch loudly rather than silently dequantizing.

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

## 3. The capacity decision

The one genuinely hard question, and it needs answering before GPU time is
spent.

### Peak residency is a max, not a sum

The runtime must sequence and free, not co-resident everything. Encode the
prompt and multimodal references, materialize the conditioning tensors, then
release the omni encoder before denoising starts. Release the DiT before the
full H3-VAE decode if the decode is the larger peak.

Done properly, peak memory is
`max(encoder, DiT + activations, VAE decode)` rather than their sum. On an omni
model whose encoder may itself be several billion parameters, that roughly
doubles the usable DiT budget. This is a hard runtime requirement, not an
optimization — the budgets below assume it.

At ~4k tokens, activations and attention workspace are small (order 1–2 GiB),
which is why the DiT weight budget tracks the MLX cap so closely.

### Ceilings

bf16 at 2.0 GB per billion params; affine int8 at 1.06; mxfp4 at 0.53.

| Unified memory | Realistic MLX cap | DiT weight budget | bf16 | int8 | **mxfp4** |
|---|---|---|---|---|---|
| 16 GB | ~10 GiB | ~8 GiB | ~4B | ~7B | **~15B** |
| 24 GB | ~15 GiB | ~13 GiB | ~6B | ~12B | **~24B** |
| 32–36 GB | ~21 GiB | ~19 GiB | ~9B | ~17B | **~35B** |
| 64 GB | ~44 GiB | ~40 GiB | ~20B | ~38B | ~75B |
| 128 GB | ~96 GiB | ~90 GiB | ~45B | ~85B | ~170B |

These are ceilings, not targets. The mxfp4 column is the whole argument: it moves
a 14B-class model from "needs 36 GB" to "comfortable on 24 GB, roomy on 32."

### Why 14B and not smaller

Not carried over from the Wan tiering — that assumed 1.3B and 14B, and 5B was
never a validated point on this stack.

The size falls out of two constraints meeting. Downward pressure is gone:
mxfp4 puts 14B at ~7.4 GB, well inside a 24 GB budget, so there is no memory
reason to shrink. Upward pressure is real: 4-bit quantization leans on model
redundancy, and the existing evidence is that 1.3B does not have enough of it
while 14B does. Shrinking the student to fit smaller Macs would actively
undermine the mxfp4 bet.

So 14B-class is where the two arguments meet — big enough to survive 4-bit,
small enough that 4-bit gets it onto median hardware. If H3's own architecture
suggests a nearby natural size, prefer that over the round number.

### Two releases, one student

Both target the same memory band and the same weights. They differ in
step schedule and render path, which is where the expensive stages get shared:

| | **Turbo** | **Quality** |
|---|---|---|
| Goal | rapid generation | maximum local quality |
| Steps | 2 | 6–8 |
| Base resolution | low, MetalFX upscale | high, minimal upscale |
| Decoder | TAEHV-class | full H3-VAE |
| Audio | optional | on |
| Memory | 24 GB | 24 GB, comfortable at 32 |

This structure matters for cost. B0 (corpus) and B1 (capacity reduction) are the
GPU-weeks, and both releases share them entirely. Only B2 diverges — one step
distillation run per release, days each. Two products for roughly one product's
training budget.

It also means neither release is a compromise of the other. A 2-step Turbo is
not a degraded Quality build; it is a separately distilled model that happens to
share a parent.

Explicitly not shipping first: a 96 GB+ native-H3 tier. It would be the highest
quality available, but it reaches almost nobody, and the point of leading with
mxfp4 is reach. Revisit after Turbo and Quality land.

### How the students get built

If H3 is MoE — every recent MiniMax model is — then *total* parameters drive
memory, not active ones, and the ceilings above bite far harder than the
active-parameter count suggests. Resolve this from `config.json` first; it
decides which path below applies.

**A. H3 ships a small variant.** Step-distill and QAT it directly (Track B2).
Weeks of work, reuses the existing Wan recipe almost unchanged, no B0 or B1.
Check for this before anything else — it is an order of magnitude cheaper than
the alternatives.

**B. Native at the top, distilled below (recommended default).** Do not
compromise the flagship to fit the smallest Mac. Tier 4 runs H3 itself, and it
will be fast there — 4k context plus a few-step schedule is a small compute job
even for a large model. Tiers 1–2 get students. This answers "highest possible
quality" honestly: the highest quality is H3 itself on a machine that fits it,
and the students exist to extend reach downward, not to define the ceiling.

**C. Full capacity distillation.** Only if A is unavailable and B's tier-4-only
flagship is unacceptable. This is Track B0 + B1 in full, and it is a multi-week
GPU project with a genuinely uncertain quality outcome rather than an
engineering task. Scope it deliberately.

## 4. Tracks

### Track A — CUDA port (prerequisite)

Standard `add-model` work, tracked in `tests/local_tests/minimax_h3/`. Required
before anything below: the trainer loads the teacher through FastVideo model
classes, and MLX parity tests compare against the torch modules.

### Track B — Distillation on GPU

Recipe base: `examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml`,
teacher swapped to H3. Operator flow follows the existing QAD runbook. Full-weight
training throughout — LoRA and adapter methods are not applicable, because both
capacity and precision are changing.

#### B0 — Synthetic corpus

The most underestimated line item. FastWan distilled against
`FastVideo/Wan-Syn_77x448x832_600k` (600k samples, order 1 TB); H3 needs an
equivalent generated by the H3 teacher on your GPUs.

Coverage matters more than raw count. The corpus must exercise every conditioning
path the student is expected to keep — text-only, image reference, video
reference, audio reference, and editing prompts. Whatever is absent from the
corpus is absent from the student, and I2V and editing quality collapse silently
rather than loudly.

Budget real GPU-weeks here before any distillation starts.

#### B1 — Capacity reduction (skip if path A applies)

Full step count, bf16, no quantization. Keep the variables separate — folding
capacity reduction into the DMD run makes failures unattributable.

- **Dense teacher:** depth pruning over width pruning. DiTs tolerate block
  removal better, and it preserves per-layer weight shapes so the Track A
  conversion mapping survives unchanged. Choose blocks by sensitivity sweep —
  ablate each, measure output drift, drop the least sensitive. Do not drop
  uniformly; middle blocks are typically more redundant than the first and last.
- **MoE teacher:** depth pruning is the wrong lever. Distill MoE → dense
  directly, which is well-trodden for LLMs and collapses the total-parameter
  memory problem that makes MoE hostile to unified memory in the first place.

Recover with layer-wise feature distillation — match retained blocks' hidden
states against their teacher counterparts — plus an output-space loss. Feature
matching converges substantially faster than output-only supervision and is
what makes this affordable at all.

#### B2 — Step distillation + QAT, jointly

DMD2 on the recovered student with the QAT callback active throughout.

These two are deliberately *not* separated, unlike B1. They interact: QAT must
see the actual few-step inference distribution to place its grid usefully, and
DMD must learn around quantization error rather than inherit it afterward.
Step-distilling first and quantizing after gives up most of QAT's benefit. This
is already how `dmd2_t2v_mlx_int8.yaml` is structured.

Derive the step schedule from H3's own sigma schedule; the Wan `[1000, 757, 522]`
values do not transfer. Target 3–4 steps.

Two runs, identical except for the QAT grid: one affine int8, one MX. See §2.

#### B3 — Audio branch

No precedent in the Wan work, and the actual differentiator — local
audio-video generation is not something else ships.

The audio branch needs its own DMD objective on audio latents plus an explicit
AV-sync term. `audio.desync` is the gate; it is the only metric that catches a
student whose audio and video are individually fine and jointly wrong.

Whether this trains jointly with B2 or as a separate stage depends on M003
(dedicated head vs distinct module).

#### B4 — What not to distill

Distill the base pass only. The student never needs to learn in-context
regeneration, because the Mac path never runs it (§1).

#### Compute expectations

Order-of-magnitude, for planning:

| Stage | Scale |
|---|---|
| B0 corpus generation | GPU-weeks |
| B1 capacity reduction | GPU-weeks, and the least predictable line |
| B2 step + QAT | days on 8 GPUs, per grid |
| B3 audio | days, if separable |

For reference, the existing Wan-1.3B QAD run is 4–8 hours on 4×B200 — that is
B2 alone, on a model that needed no B0 or B1. B0 and B1 are the budget.

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

## 6. Proving mxfp4 on an M5 / 24 GB machine

The bet needs validating before H3 GPU time is committed. All three experiments
below run on **existing Wan weights**, need no H3 access, and answer separable
questions. Ordered by cost.

### E1 — Does MLX actually use the hardware? (hours, no model)

Pure microbenchmark. `benchmark_mlx_linear` and `benchmark_mlx_attention` in
`mlx_runtime/fastwan.py` already have the right shape; add mxfp4 and mxfp8 arms
alongside int8 and bf16, and run at H3-realistic shapes (~4k tokens).

What to look for: mxfp4 should be *substantially* faster than int8, not
marginally. A result within noise of int8 means MLX is emulating the format
rather than dispatching to the neural accelerators, and the entire throughput
half of the bet evaporates.

This resolves M006, and `quantization_support_error()` will not tell you — it
probes numerical correctness and passes either way, emulated or not.

Do this first. It is a few hours and it can kill the bet before anything
expensive starts.

### E2 — How much of mxfp4 is recoverable without retraining? (a day to a week)

**The already-trained 14B does not need a new GPU job to get better mxfp4
output.** The existing result is *naive* PTQ, and most of the gap between naive
PTQ and QAT is closable by calibration-grade methods that never touch the
training loop. Work down this ladder and stop when quality is acceptable.

Prerequisite for everything below: a **bit-exact mxfp4 grid simulator in
PyTorch**, matching `mx.quantize(..., mode="mxfp4")` output exactly. Build it
once and it unlocks both this ladder and E3.
`fastvideo/tests/mlx/test_mlx_affine_qat_parity.py` already establishes the
pattern for proving grid equivalence — write the MX sibling beside it.

**L1 — Layer targeting. Free, no data.** Confirm the failing run did not
quantize modulation/AdaLN, timestep embedding, patch embed, and the final
projection. `nvfp4_qat_config.DEFAULT_FP4_LAYERS` already encodes the right
target set for the CUDA path — attention and FFN projections only. If the MLX
run quantized uniformly, this alone may account for much of the gap.

**L2 — Shared-exponent search. Free, no data, minutes.** The lever most specific
to MX and most likely to be untouched. MXFP4's block scale is E8M0 —
power-of-two only — and the default absmax choice takes the smallest exponent
that avoids clipping outright. With E2M1 elements carrying so few magnitude
levels, deliberately choosing `E-1` and accepting slight clipping often halves
reconstruction error across the bulk of the distribution. Search `{E, E-1, E-2}`
per block against MSE. Power-of-two-only scaling is exactly why MX wastes range
on the default choice, so this recovers something affine int8 never had to give
up in the first place.

**L3 — GPTQ-style error compensation. Hours, small calibration set, no
backprop.** Quantize column by column, updating the remaining unquantized
weights to absorb the error already introduced, via the layer-input Hessian. At
4 bits this is routinely the difference between unusable and near-lossless, and
it applies to any deterministic round-to-grid function, MX included. AWQ-style
salient-channel scaling composes with it cheaply.

**L4 — Layer-wise output reconstruction (AdaRound / BRECQ). About a day.**
Optimize rounding decisions per layer against calibration activations. This is
optimization, but *layer-local* — no full-model backprop, no teacher, no
distillation corpus. The closest thing to QAT that is not QAT.

**Timestep-aware calibration is mandatory for L3 and L4.** DiT activation
statistics vary enormously across the noise schedule, and calibrating at one
timestep yields a quantizer that is wrong everywhere else — the failure mode
Q-Diffusion and PTQ4DM exist to address. Favorable twist: a model already
distilled to a few steps only needs calibration covering *those* steps. Few-step
models are markedly easier to PTQ well than 50-step ones, which compounds nicely
with the rest of the plan.

Run this on the 24 GB M5 machine directly — 14B at mxfp4 is ~7.4 GB. Precompute
prompt embeddings offline to sidestep text-encoder residency during the test.

**Expected outcome.** L1 and L2 are free and worth doing regardless. L3 is where
the large recovery usually lives. Realistic expectation is that this ladder
closes most of the distance to QAT but not all of it — weight error still
compounds across steps, and only QAT trains the model to absorb that. "Most of
the way, in a day, without a GPU job" is the right frame. "PTQ makes QAT
unnecessary" is not.

### E3 — Does MX-grid QAT close the rest? (~a week, small GPU spend)

Cheaper than it first looks, because the scaffolding already exists.
`nvfp4_qat_train_config.py` is a wired STE path — trainable bf16 master weight,
fake-quantized to a microscaled 4-bit grid every forward, full-precision
backward through `fp4linear._LinearFWD4BWD16Fn`. NVFP4 and MXFP4 are close
cousins: both E2M1 elements, differing in block size (16 vs 32) and scale type
(FP8 E4M3 vs power-of-two E8M0). **MX-grid QAT is a quantizer swap inside an
existing path, not a new training stack.**

Write the MX-grid sibling to `mlx_affine_qat.py` reusing that STE, then run the
existing `dmd2_t2v_mlx_int8.yaml` recipe on the **1.3B** — deliberately the
hardest case, since it has the least redundancy and the worst observed MX
behavior. Compare QAT-mxfp4 against the best E2 build at equal steps.

If QAT visibly rescues mxfp4 at 1.3B, it will do better at 14B, and the bet is
proven end to end. Use 1.3B rather than 14B because it is cheap and because a
win there is a strictly stronger result.

### Gate

Commit to mxfp4-first only if E1 shows a real hardware path and E2 or E3 shows a
real quality recovery. If E1 fails, fall back to int8 and keep the §3 ceilings.
If E1 passes but E2 and E3 both fail, mxfp8 becomes the M5 target instead of
mxfp4 — still a win over int8 on throughput, at a much smaller memory advantage.

## 7. Sequencing

**Now, unblocked:** E1 → E2 → E3 (§6). None of these need H3 weights, network
access to Hugging Face, or the CUDA port. They run entirely on existing Wan
checkpoints and they decide the deploy grid.

**In parallel:** land the CUDA port (Track A), and read `config.json` the moment
weights are reachable — dense-vs-MoE and total parameter count gate the
capacity path in §3.

**After the §6 gate passes:**

1. Track B0 — synthetic corpus generation.
2. Track B1 — capacity reduction to the 14B-class student.
3. Track B2 ×2 — Turbo and Quality step distillations, each with QAT on the
   chosen grid. Plus one int8 run per release for the fallback artifact.
4. Track B3 — audio branch.
5. Track C — MLX runtime, parity against Track A.
6. Track D — operating-point sweep per release, including the
   MetalFX-interpolation arm.

The ordering point worth holding onto: §6 costs days and gates months. Running
E1 before anything else is the single highest-leverage scheduling decision here,
because a failed E1 changes the entire plan and costs an afternoon to discover.

## 8. Open questions

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
| M009 | Did mxfp8 hold up at 14B, or was it only mxfp4 that degraded? The 1.3B run showed both failing; the 14B run is reported as mxfp4-bad. If mxfp8 was fine at 14B it strongly supports the size-tolerance argument in §2 and gives the bet a safer intermediate landing spot. | §2, E2 scope |
| M010 | Does `mx.quantize` mxfp4 accept a per-layer grid override, or does the mixed-precision split in E2 need weights held in separate arrays by dtype? | E2 implementation |
