# Apple Silicon Plan: MiniMax-H3 Local Video Generation

Status: design, pre-implementation.

Release 0 (§4) depends on nothing and can start immediately. The H3 releases
(§5) depend on the `minimax_h3` CUDA port
(`tests/local_tests/minimax_h3/PORT_STATUS.md`).

This is the **local/Metal track**. It is deliberately separate from upstream
FastVideo H3 support, which is a plain CUDA model port. Nothing here should be
folded into that PR.

Goal: the highest-quality local video generation that runs at interactive speed
on Apple Silicon — not the largest model that technically loads.

## 0. Decisions

| Question | Decision |
|---|---|
| Primary quantization | **Affine int8**, mixed precision. It is what works (§2). |
| mxfp4 | **Provisionally dropped.** Has failed at 1.3B and 14B, with and without QAT. Four confounds are worth eliminating first (§2); if they come back clean, abandon it. |
| Memory target | **24 GB floor, 32–36 GB comfortable** (§3). Not 96 GB. |
| Student capacity | **~14B class** (§3). mxfp4 puts 14B in ~7.4 GB, and it is the size already in hand for validation. Not justified by quantization tolerance — see §2. |
| First ship | **Release 0: FastWan-14B on Metal** (§4), ahead of and independent of H3. |
| Release shape | **One student, two step-distillations**: Turbo (rapid) and Quality. Shared expensive stages, cheap divergence (§5). |
| Validation before commitment | Three experiments, days not months, on existing models (§8). The bet is gated on these, not on faith. |
| Expected H3 size | **Unknown (M001).** Ceilings in §3 say what fits, not what H3 is. |

Deliberate consequence: pre-M5 Macs get the int8 artifact and are not the design
target. That is an accepted cost of leading with mxfp4.

## 0b. Confidence

Stated plainly, because the plan reads more confident than the evidence
supports.

| Claim | Confidence | Basis |
|---|---|---|
| Affine int8 works at 14B | high | measured |
| mxfp4 is a large memory win over int8 | certain | arithmetic |
| MX PTQs worse than affine int8, for the stated grid reasons | high | numerics; matches observed runs |
| **mxfp4 is viable for a video DiT at all** | **unresolved — not yet fairly tested** | failed at both sizes, but the matched control shows the mxfp4 run had *lower* loss and *half* the student gradient norm, which points at a broken QAT path rather than a bad format (§2). |
| MLX routes mxfp4 to M5 neural accelerators | **unknown** | untested — E1 |
| A 14B student can be distilled from H3 at acceptable quality on 8×B200 | **low** | depends on H3's size and whether it is MoE (M001) |

An earlier revision of this table rated MX-grid QAT "low–medium" on the belief
that it was untried. It has been tried, at both sizes, and did not work. The
correct posture now is **diagnose the existing run, then most likely drop
mxfp4** — not "run the experiment we have not run yet."

The honest summary: **affine int8 is the plan; mxfp4 is a diagnosis away from
being abandoned.** Release 0a (§4) already assumes this and depends on none of
it.

One expectation to set precisely, because it is easy to get backwards: **mxfp4's
upside is memory and throughput, not fidelity.** Affine int8 currently produces
better output and will likely keep doing so at equal effort. The goal of MX-grid
QAT is to reach *near-int8 quality at half the weight memory and, if E1 lands,
higher throughput on M5*. Anyone expecting mxfp4 to look better than int8 will
read a successful result as a failure.

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

The second H3 property that matters is sequence length, but it needs stating
carefully, because there are **two different token counts** and conflating them
overstates the case:

- **Conditioning context**: Contextual Omni Representation compresses
  multimodal inputs from ~100k tokens to ~4k. Genuinely small.
- **Denoised latent sequence**: separate, and driven by H3-VAE compression,
  clip length, and resolution — not by the 4k figure at all.

Order of magnitude on the second: 5s at 24fps and 720p is ~120 frames; a
conventional 8×8 spatial / 4× temporal VAE with 2×2 patching puts that near
100k tokens, and H3-VAE's claimed ~4× compression gain brings it to roughly 25k.
That is much better than typical for video, but it is not 4k, and attention is
not free at that length.

So the accurate claim is narrower than "attention doesn't matter": weight memory
is the **dominant** constraint and attention is a real but secondary cost, which
is why base resolution and clip length are the levers Turbo pulls hardest (§5).
Confirm the actual latent token count from H3-VAE's strides before sizing
activation budgets (M011).

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

**MX-grid QAT has already been run, and it did not fix mxfp4.** Two earlier
revisions of this document asserted the opposite — that both MX results were
post-training quantization and that QAT aimed at the MX grid was untried. That
was wrong, and it was the load-bearing assumption under the whole bet.

The actual state of evidence:

| Model | Grid | Method | Result |
|---|---|---|---|
| 1.3B | affine int8 | QAD | **good** — the prior plan of record |
| 1.3B | mxfp8, mxfp4 | PTQ | bad |
| 1.3B | mxfp4 | QAD | bad |
| 14B | mxfp4 | QAD — `dmd2_t2v_14b_mlx_mxfp4.yaml`, `simulate_dtype: fp16`, DMD2 3-step `[1000, 757, 522]`, 4000 iterations, 4×B200, student+critic from `fastwan14b_distilled`, frozen Wan2.1-14B teacher | bad |
| **14B** | **affine int8** | — | **never tried** |

Two things follow, and the second corrects an error repeated across several
earlier revisions of this document.

**mxfp4 has failed at two sizes, with and without QAD.** The evidence is against
it — but see the independence caveat immediately below, which materially weakens
how much that evidence proves.

### The two mxfp4 failures are not independent

Both QAD runs are from the same commit (`411cfa7e`) and share every component
that could plausibly be at fault:

| | 1.3B | 14B |
|---|---|---|
| Config | `dmd2_t2v_1p3b_mlx_mxfp4.yaml` | `dmd2_t2v_14b_mlx_mxfp4.yaml` |
| Quantizer | `mode: mxfp4`, `simulate_dtype: fp16` | identical |
| Schedule | DMD2 3-step `[1000, 757, 522]` | identical |
| `generator_update_interval` | 5 | 5 |
| EMA | `decay: 0.98`, `start_iter: 0` | identical |
| `rollout_mode` | simulate | simulate |
| Runtime | 1 d 8 h 34 m | 6 d 12 h 30 m |
| Logged `step_time_sec` | 51.2 | 239.8 |

They are two samples of *one* pipeline, not two independent tests of the mxfp4
grid. If the PyTorch fake-quantizer behind `mode: mxfp4` does not match MLX's
deploy grid, **both results are void for the same single reason** — and roughly
**31 GPU-days** of B200 time is void with them.

This is why E0 (§8) is the first thing to do and why it is worth more than
another training run. Two correlated failures through one unverified code path
is weak evidence about the format and strong evidence that the code path needs
verifying.

One supporting detail: the logged `step_time_sec` implies ~2300 iterations in
both runs, against 4000 configured — but the logged-to-implied ratio is 1.75 and
1.70 respectively, near-identical across two very differently sized jobs. A
consistent ratio points to metric semantics rather than two coincidentally
truncated runs; most likely `step_time_sec` samples only the costlier
student-update iterations. If so both runs did complete 4000 iterations, giving
the student **800 updates**, and true mean iteration time was ~29 s (1.3B) and
~141 s (14B) on 4×B200. Worth confirming, since it decides whether the
undertraining hypothesis is about 470 updates or 800.

**Nothing at 14B has been shown to work on Metal.** Earlier revisions asserted
that affine int8 was good at 14B and used that to present Release 0a as a
zero-risk ship. That run was never done — int8 is validated at 1.3B only. The
14B/int8 cell is the most important empty box in this table, and filling it is
now the highest-value experiment in the plan (§8/E0b).

### The matched control points at an optimization bug, not a bad format

A near-matched pair exists at 1.3B — same commit `411cfa7e`, same student and
critic init, same DMD2 3-step schedule, same `generator_update_interval: 5`,
same EMA, same `rollout_mode`, runtimes within 3%. The only material difference
is the quantization block. **This is the cleanest evidence available and it does
not say "mxfp4 is a bad format."**

| Metric | int8 (liked) | mxfp4 (bad) | ratio |
|---|---|---|---|
| Config | `bits: 8`, `group_size: 64` | `mode: mxfp4` | *different code paths* |
| Runtime | 1 d 7 h 34 m | 1 d 8 h 34 m | 1.03 |
| `step_time_sec` | 49.2 | 51.2 | 1.04 |
| `total_loss` | 0.622 | 0.575 | **0.92** |
| `generator_loss` | 0.419 | 0.390 | 0.93 |
| `grad_norm/student` | 1.332 | 0.623 | **0.47** |
| `grad_norm/critic` | 0.260 | 0.215 | 0.82 |

Two readings jump out:

**The mxfp4 run has lower loss and worse output.** Loss is not tracking quality,
which means the objective is not measuring what it should — a degraded critic
signal, or a student that is not actually moving.

**The student's gradient norm is less than half.** That is the diagnostic
number. A model converged to a worse optimum would show comparable gradient
magnitudes and *higher* loss. Low loss plus halved gradients reads as a student
that is barely being optimized at all.

**Check the recipe before the format.** The working int8 config is
`dmd2_t2v_mlx_int8_v2.yaml`, and its header documents why v1 failed: global
batch 4 left "the critic noisy and the student under-trained (weights moved only
~0.2% from init)", fixed by `gradient_accumulation_steps: 4` plus initializing
student and critic from the already-distilled FastWan rather than base Wan.
**That is the same signature the mxfp4 runs show**, and it failed that way on
int8, for reasons unrelated to the numeric format. The mxfp4 configs are not in
the committed tree at `411cfa7e`, so whether they inherited v2's fixes is
unverified — confirm before spending anything else.

If the recipe checks out, the next hypothesis is that **the straight-through
estimator is attenuating gradients on the mxfp4 path.** mxfp4's representable
range is far narrower than affine int8's, so many more weights sit at grid
saturation — and a clipped STE zeroes gradients for saturated values. That would
produce the same signature, and it is a bug in the QAT path rather than a
property of 4-bit weights.

Supporting this: the two configs are not the same code. int8 goes through the
affine path (`bits` + `group_size`); mxfp4 goes through a separate `mode`-keyed
path with no group size specified. The mxfp4 path is the far less exercised of
the two.

**Cheap diagnostic, no training run:** instrument one forward/backward on each
path and log, per layer, the fraction of weights at grid saturation and the
fraction of gradients passing through the STE non-zero. If mxfp4's passthrough
rate is materially lower, that is the bug, and it is fixable — a non-clipped or
soft-clipped STE, or per-group scale search (§8/E2 L2) to reduce saturation in
the first place.

### Other confounds worth eliminating

The run does not cleanly kill mxfp4 either, because four things could each
produce this outcome independently of the grid being unworkable. All are cheap
to check relative to another 4000-iteration run.

1. **Grid fidelity — check this first.** `simulate_dtype: fp16` fake-quantizes
   in PyTorch. If that simulation is not *bit-exact* with MLX's
   `mx.quantize(..., mode="mxfp4")` — block size, E8M0 exponent rounding,
   element rounding mode — then QAT optimized against a grid that is not the
   deploy grid, which is precisely the train/deploy mismatch this document warns
   about elsewhere. `fastvideo/tests/mlx/test_mlx_affine_qat_parity.py` bounds
   the affine path to one source-dtype epsilon; there is no MX equivalent.
   Without that test, the run's grid is unverified.

2. **Undertraining — now demoted.** The matched int8 run converged to liked
   quality on the *same* iteration and student-update budget, so the budget is
   evidently sufficient for this recipe. mxfp4 needing materially more would
   itself be a symptom of the weak-gradient problem above, not an independent
   cause. Do not spend a longer run on this until the STE is checked.

3. **No mixed precision.** The config shows no layer-skip list. If mxfp4 was
   applied to every targeted linear including modulation/AdaLN, timestep
   embedding, patch embed, and final projection, the highest-leverage cheap fix
   was never in play (L1 in §8/E2).

4. **Matched control — resolved.** One exists (see above), and it is what turned
   this from "the format is bad" into "the training path looks broken."

Order: **STE gradient/saturation diagnostic** (above) → **E0 grid parity** → **(3)
mixed precision** → then, only if all three come back clean, a corrected run.

If a verified-exact grid with a healthy STE and mixed precision still produces
bad output, mxfp4 is dead for video DiTs at this scale and affine int8 is the
answer. Nothing measured so far establishes that yet.

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

### Why 14B — honestly

Not carried over from the Wan tiering, and **not** justified by demonstrated
quantization tolerance at that size (see §2 — mxfp4 failed at 14B too). The
actual reasons are weaker than that would have been:

1. **mxfp4 removes the downward pressure.** 14B at ~7.4 GB sits comfortably in a
   24 GB budget, so there is no memory argument for shrinking further.
2. **It is a size already worked with.** The existing 14B Wan gives a same-size
   testbed for E1–E3, so the quantization work is validated at the target scale
   before H3 weights exist.
3. **Distillation cost rises steeply with student size**, and B1 is the schedule
   risk (§Track B). 14B is near the upper bound of what 8×B200 can plausibly
   reach in weeks rather than months.

Reason 3 is the binding one. If B1 proves harder than expected, the right
response is a smaller student, not a longer schedule — and a smaller student
makes the mxfp4 bet harder, since 1.3B is where MX behaved worst. Those two
pressures point in opposite directions and the tension is unresolved until §8
reports.

Also not carried over from the Wan tiering — that assumed 1.3B and 14B, and 5B
was never a validated point on this stack.

If H3's architecture suggests a nearby natural size, prefer that over the round
number. 14B is a target, not a constraint.

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

## 4. Release 0 — FastWan-14B on Metal, ahead of H3

Ships **before** any H3 work lands, and depends on none of it.

The reasoning: the entire Apple Silicon stack — mxfp4, MX-grid QAT, MetalFX and
RIFE postprocess, sequenced residency, generation-aware tiering — is
model-agnostic. The existing 14B Wan already exercises it at exactly the target
size, and the MLX Wan runtime already exists. So the stack can be validated and
shipped without waiting on the H3 port, the corpus, or capacity distillation.

### Ship it in two stages, not one

The prior plan of record was **FastWan-1.3B at int8**. Against that baseline,
the largest available quality win is not the grid — it is the 10× capacity jump
to 14B, on a grid that already works.

| | **0a — ships first** | **0b — the upgrade** |
|---|---|---|
| Model | FastWan 14B | FastWan 14B |
| Grid | affine int8, mixed precision | mxfp4 |
| Weights | ~15 GB | ~7.4 GB |
| Target | 32 GB+ Macs, any generation | 24 GB Macs, M5+ |
| Blocked by | **E0b** — 14B at int8 has never been run (§2) | E0, then E1–E3 (§8) |
| Risk | low, but not zero — validate before promising a date | the bet |

0a is *low* risk rather than *no* risk. int8 is proven at 1.3B, and 8-bit is far
more forgiving than 4-bit, so it is very likely to hold at 14B — but "very
likely" is not "measured," and an earlier revision of this document wrongly
claimed it was. Run E0b before committing to a ship date.

This is the honest framing of what mxfp4 buys: **it moves 14B from "needs
32 GB" to "comfortable on 24 GB."** At int8, 14B is ~15 GB of weights, which is
marginal on a 24 GB machine even with sequenced residency and a raised wired
limit, and comfortable on 32. That reach difference is the return on the bet —
not better output.

Staging this way means Release 0 is not gated on the mxfp4 experiments at all.
0a ships on the grid you already trust, delivers the capacity jump users will
actually notice, and 0b follows as a same-model upgrade if §8 lands.

| Property | Value |
|---|---|
| Model | existing FastWan 14B, no capacity reduction |
| Training needed | 0a: none beyond existing QAD. 0b: E2 ladder, then MX-grid QAD if E2 falls short |
| Blocked by | nothing for 0a — no H3 port, no HF access, no corpus |

What this buys, in order of value:

1. **A shippable Mac product in weeks rather than months**, with the launch date
   independent of B0 and B1 — the two line items most likely to slip.
2. **The stack validated at 14B before H3 exists.** Every unknown in the
   confidence table except M001 gets answered here, on a model already in hand.
3. **The headline.** First 4-bit video DiT on Metal is a claim that does not
   require H3 to make.

Once H3 lands, Turbo and Quality below swap the parent model into a proven
runtime instead of debugging quantization, distillation, and Metal numerics
simultaneously.

Sequencing consequence: §8's experiments stop being pre-work for H3 and become
**Release 0's critical path**. Start them now.

## 5. The two H3 releases

Both are MLX builds of the same distilled H3 student. Everything marked
*derived* follows from a decision elsewhere in this doc; everything marked
*unknown* waits on H3's config.

### Shared across both

| Property | Value |
|---|---|
| Parent | one ~14B-class student distilled from H3 (§3) |
| Deploy grid | mxfp4, M5 and newer — gated on §8 |
| Fallback artifact | affine int8, same student, one extra B2 run — serves M1–M4 |
| Mixed precision | attention + FFN projections quantized; AdaLN/modulation, timestep embedding, patch embed, final projection, norm affines held bf16 (§2) |
| DiT weights | ~7.3 GB quantized + ~0.6 GB bf16 ≈ **~8 GB** |
| Memory floor | 24 GB unified |
| Residency | encode → free encoder → denoise → decode; peak is a max, not a sum (§3) |
| Audio | H3 audio VAE, native stereo |
| Runtime | `fastvideo/mlx_runtime/minimax_h3.py` (Track C) |

The int8 fallback is ~15 GB of weights, which still fits 24 GB but with no
headroom. Pre-M5 machines are a compatibility target, not a design target.

### Turbo — rapid generation

| Property | Value |
|---|---|
| Positioning | fastest usable local video; interactive iteration |
| Steps | **2** (DMD2) |
| Base resolution | low — 480p class |
| Output resolution | MetalFX upscale to ~1080p |
| Frame generation | half cadence, RIFE or MetalFX interpolation to 24 fps |
| Clip length | 5s |
| Decoder | TAEHV-class tiny decoder |
| In-context regeneration | never |
| Audio | optional, off by default |
| Peak memory | ~10 GB |

Turbo pulls hardest on the two levers that cut the *latent sequence* rather than
the weights — base resolution and frame cadence — because at 2 steps the
per-step attention cost is the largest remaining term.

### Quality — maximum local fidelity

| Property | Value |
|---|---|
| Positioning | best local output that still runs on median hardware |
| Steps | **8** (DMD2) |
| Base resolution | 720p class |
| Output resolution | MetalFX upscale toward 1440p/2K |
| Frame generation | native 24 fps, no interpolation |
| Clip length | 5s, 10s where memory allows |
| Decoder | full H3-VAE, tiled |
| In-context regeneration | never on device — MetalFX substitutes |
| Audio | on |
| Peak memory | ~12–14 GB; comfortable at 32 GB |

### Why one parent and two distillations

B0 (corpus) and B1 (capacity reduction) are the GPU-weeks and are shared
entirely. Only B2 diverges — one step-distillation per release, days each, plus
one int8 run per release for the fallback. Two products for roughly one
product's training budget.

Neither is a degraded build of the other. A 2-step Turbo is separately
distilled, not Quality with steps truncated at inference — truncating a
schedule the model was not trained for is exactly the failure mode DMD exists
to avoid.

### Ship criteria

Per release, from §7: video gates on `vbench.subject_consistency`,
`vbench.motion_smoothness`, `vbench.dynamic_degree`, `common.fvd`; audio on
`audio.clap_score` and `audio.frechet_distance`; joint on `audio.desync`. All
scored against the CUDA H3 reference, not against an fp16 MLX build, so
distillation and quantization error are measured together.

Turbo additionally gates on wall-clock — it has no reason to exist if it is not
decisively faster than Quality on the same machine.

### What is still unknown

Resolution and clip-length rows are **targets, not commitments**. They depend on
H3-VAE's actual strides and the resulting latent token count (M011), on whether
the base pass can run standalone without in-context regeneration (M004), and on
the omni encoder's memory cost (M005). Expect these numbers to move once
`config.json` is readable; the structure of the two releases should not.

## 6. Tracks

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

Anchored on the one real datapoint available: the Wan-1.3B QAD run is **4–8
hours on 4×B200** for DMD2 to 3 steps. That is B2 alone, on a model that needed
neither B0 nor B1.

Scaling that to a 14B student on 8×B200 — ~10× the parameters, 2× the GPUs, and
DMD2's three resident networks (frozen teacher, trainable student, critic):

| Stage | 8×B200 estimate | Confidence |
|---|---|---|
| B0 corpus generation | 1–3 weeks | low — scales with teacher inference speed, unknown until H3 runs |
| B1 capacity reduction | **weeks to months, or infeasible** | very low — see below |
| B2 step + QAT, per release per grid | **~26 GPU-days** | high — measured |
| B3 audio | days, if separable | low |

The B2 figure is measured: the 14B mxfp4 QAD run took **6 d 12 h 30 m on
4×B200** for 4000 iterations — **26 GPU-days**, or ~141 s/iteration on 4 GPUs.
Earlier revisions of this document mis-stated this twice, first by extrapolating
from the 1.3B run and then by assuming the run used 8 GPUs.

| GPUs | Wall-clock for one B2 |
|---|---|
| 4 | 6.5 days |
| 8 | 3.3 days |
| 16 | **1.6 days** |
| 32 | 0.8 days |

This is much better news than the previous revision implied, and it changes two
decisions:

- **Running Turbo and Quality concurrently at 16 GPUs each finishes both in
  ~1.6 days.** With 32 available that is close to free.
- **The undertraining hypothesis is cheap to test.** 4× the iterations (16,000)
  at 16 GPUs is ~6.5 days — the same wall-clock the original run cost, for four
  times the student updates. Given the student saw only ~800 updates at
  `generator_update_interval: 5`, this is an affordable and well-targeted
  retry — but only *after* E0 confirms the grid (§8).

Note the run resumed from `checkpoint-40`, so earlier segments exist and
cumulative training may exceed 4000 iterations. Confirm before concluding
anything about training budget.

**B1 is the schedule risk and it is not a small one.** Memory alone is
manageable — a 14B student with Adam states plus a frozen teacher and critic is
on the order of 250 GB of optimizer and weight state, which shards across
8×B200's ~1.4 TB. The problem is that capacity distillation from a frontier
model is not a fine-tune. If H3 turns out to be a large MoE, the compression
ratio to a 14B dense student is severe enough that 8 GPUs may simply not be
enough to recover acceptable quality in any reasonable wall time. MiniMax
trained H3 on orders of magnitude more compute than this.

Treat B1 as the line item most likely to force a plan change, and resolve M001
before committing to it.

#### Do NVIDIA targets need their own training run?

No — not a separate distillation. A shared parent plus short per-grid
adaptation.

QAT bakes the *deploy grid* into the weights, so strictly, one grid means one
QAT. But the expensive thing being learned in B2 is **how to denoise in 2 or 8
steps**, and that is entirely grid-independent. Absorbing a particular grid's
quantization noise is the cheap part.

Grids actually in play:

| Target | Grid | Shape |
|---|---|---|
| Mac, M5+ | mxfp4 | E2M1, block 32, E8M0 power-of-two scale |
| Mac, M1–M4 | affine int8 | block 64, fp16 scale + bias |
| NVIDIA Blackwell (5090, B200) | nvfp4 | E2M1, block 16, E4M3 scale — already supported in-tree via `nvfp4_config.py` |

**mxfp4 is the most constrained 4-bit grid of these** — coarser blocks and a
coarser (power-of-two) scale than nvfp4. That asymmetry is useful: a model
QAT'd for mxfp4 is trained to tolerate the worst-case 4-bit noise, so deploying
it on nvfp4 lands on a strictly finer grid and should transfer gracefully. The
reverse does not hold. **Doing the Metal MX work first makes the Blackwell
artifact nearly free; doing it in the other order does not.**

Recommended structure, per release:

1. One full B2 on **mxfp4** — the constrained grid. Days.
2. Short **grid-adaptation fine-tunes** from that checkpoint for nvfp4 and
   affine int8. Hours, not days — the step schedule is already learned and only
   the quantization noise profile changes.

That yields 2 releases × 3 grids = 6 artifacts from 2 expensive runs and 4 cheap
adaptations, rather than 6 full distillations.

Worth noting the 5090 wants this regardless of Metal: 14B at bf16 is ~28 GB
against 32 GB of VRAM, so 4-bit buys real headroom for longer clips and larger
batches there too. The NVIDIA artifact is not charity for a second platform — it
is a better 5090 product than the bf16 build.

#### GPU allocation

Working ceiling: **36 B200s.** They are not all equally useful at every stage,
and the recommended split is not "all 36 on the training job."

| Stage | Useful count | Scaling |
|---|---|---|
| B0 corpus | **all 36** | near-linear — embarrassingly parallel inference. This is where extra GPUs convert most directly to wall-clock. |
| B1 capacity reduction | **32** | good to ~32 with 2D HSDP; the longest job and the one that most justifies the cluster. |
| B2 step + QAT | ~16 each | saturates earlier — DMD's global batch grows with GPU count, and past ~16 the extra batch buys convergence little. |
| B3 audio | ~8–16 | similar to B2. |

**Recommended standing split: 32 training + 4 held back.** Two reasons. First,
32 factors cleanly (4×8 or 8×4) for HSDP meshes and collective kernels; 36 does
not, and 6×6 or 4×9 meshes are a needless source of poor collective performance.
Second, a dedicated 4-GPU eval box means VBench and FVD runs never contend with
training, which matters when every B-stage gates on §7 metrics.

**Run the two B2 distillations concurrently, not serially.** Turbo and Quality
at 16 GPUs each finish both in roughly the wall time of one. That is the single
best use of having more than 16 GPUs, and it directly compresses time-to-launch.

Two caveats on the 36:

- **The frozen teacher may be large.** If H3 is a 200B+ MoE, the teacher alone
  is 400 GB+ at bf16 — three or more B200s per replica before the student,
  critic, or optimizer states are counted. That changes every row above. M001
  decides whether 36 is comfortable or tight.
- **36 GPUs does not make B1 feasible if the compression ratio is too severe.**
  Going 8 → 32 is ~4×, which turns months into weeks; it does not turn
  infeasible into feasible. It substantially de-risks B1 without eliminating the
  risk.

Also size the storage path, not just the GPUs. B0 at 36-way parallelism will
generate on the order of a terabyte of latents, and the write bandwidth becomes
the bottleneck well before the GPUs do.

The current runbook pins `hsdp_shard_dim` to the GPU count, which is correct for
a 1.3B on 4 GPUs (effectively pure FSDP) but wrong at this scale. A 14B student
plus frozen teacher and critic needs a shard dim large enough to fit the state,
then replication across the rest — a genuine 2D mesh. Budget config work for
this before the first large run.

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

## 7. Quality gates

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

## 8. Diagnosing mxfp4 before abandoning it

Reframed. This section previously read as "validate a promising untried bet."
mxfp4 QAD has since been run at both sizes and failed (§2), so the question is
no longer whether to try it — it is whether the failure has a cheap explanation
or is intrinsic.

**Do not spend another full run on mxfp4 before E0 completes.** The last one was
6.5 days of 8×B200; repeating it against an unverified grid would spend that
twice for the same ambiguity.

Everything below runs on existing Wan weights and needs no H3 access.

### E0 — Is the QAT grid actually the deploy grid? (a day, no GPU)

**The highest-priority item in this document**, and it should have been written
before the 14B run rather than after.

Take the PyTorch fake-quantizer behind `mode: mxfp4` with
`simulate_dtype: fp16`, and compare its output tensor-for-tensor against MLX's
`mx.quantize(..., mode="mxfp4")` on identical inputs. Check block size, E8M0
exponent selection and rounding, element rounding mode, and the dequantized
reconstruction. Bound the difference the way
`fastvideo/tests/mlx/test_mlx_affine_qat_parity.py` bounds the affine path.

If they diverge, **both** QAD runs trained against a grid they would never deploy
on, the "mxfp4 failed" result is void, ~31 GPU-days is written off, and one
corrected run is clearly worth doing. If they match bit-for-bit, the failure is
real and E2/E3 below become the last cheap things to try before dropping mxfp4.

Zero GPU cost. Do it first among the mxfp4 items.

### E0b — Does 14B work at affine int8? (a day for PTQ; ~1.6 days on 16 GPUs if QAT is needed)

**The highest-value experiment in the plan**, because it gates Release 0a — the
one thing here with a near-term ship date — and because the 14B/int8 cell in §2
is simply empty.

Two rungs:

1. **PTQ first, same day, no training.** Quantize the existing
   `fastwan14b_distilled` weights to affine int8 with the §2 mixed-precision
   split and look at the output. 8-bit PTQ is far more forgiving than 4-bit, and
   larger models generally PTQ better, so this may just work. If it does,
   Release 0a ships with **no training at all**.
2. **QAD if PTQ falls short.** The shipping 1.3B int8 was QAD'd, not PTQ'd, so
   14B may need the same. That is a `dmd2_t2v_mlx_int8.yaml` run at 14B —
   ~1.6 days on 16 GPUs (§Track B).

Either way this is cheap, and unlike everything mxfp4-related it is on the
critical path to a shipped product rather than to a research question.

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

## 9. Sequencing

**Now, unblocked — and now the critical path for Release 0, not pre-work:**
E1 → E2 → E3 (§8). None of these need H3 weights, Hugging Face access, or the
CUDA port. They run on existing Wan checkpoints and they decide the deploy grid.

**Then Release 0 (§4):** FastWan-14B on Metal. Ships independently of everything
below.

**In parallel throughout:** land the CUDA port (Track A), and read `config.json`
the moment weights are reachable — dense-vs-MoE and total parameter count gate
the capacity path in §3, and M001 can be answered long before the port finishes.

**After the §8 gate passes and the H3 port lands:**

1. Track B0 — synthetic corpus generation.
2. Track B1 — capacity reduction to the 14B-class student.
3. Track B2 ×2 — Turbo and Quality step distillations, each with QAT on the
   chosen grid. Plus one int8 run per release for the fallback artifact.
4. Track B3 — audio branch.
5. Track C — MLX runtime, parity against Track A.
6. Track D — operating-point sweep per release, including the
   MetalFX-interpolation arm.

The ordering point worth holding onto: §8 costs days and gates months. Running
E1 before anything else is the single highest-leverage scheduling decision here,
because a failed E1 changes the entire plan and costs an afternoon to discover.

## 10. Open questions

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
| M011 | What is the actual denoised latent token count at each target resolution and clip length? Follows from H3-VAE's spatial/temporal strides and patch size. Distinct from the ~4k conditioning context. | §1 thesis, §5 resolution rows, activation budgets |
