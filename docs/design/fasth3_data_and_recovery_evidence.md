# FastH3 recovery: data and practical research findings

September 6, 2026. This supplements the current consumer model decision;
it does not replace the authorized longer hard-pruning recovery experiment.

## Data decision

The 5,585 prepared training prompts are the 124-frame subset of the 58,669
training prompts. They are the first geometry-compatible curriculum, not a
permanent cap or evidence that the remaining prompts are less valuable.
Aspect ratios still need matching buckets. Add the remaining durations with
correct video/audio latent lengths and realistic token budgets. Do not crop
15-second dialogue or multiple timed shots into a five-second target.

Prompt-only teacher rollouts can use new captions without clean media. A
separate real-data denoising loss requires aligned video, stereo audio and
matching descriptions; merely adding prompt files cannot supply those targets.

Priority data candidates are existing authorized paired audiovisual corpora;
quality-screened Base H3 synthetic clips generated for targeted failures;
and a curated set of real speech, visible sound-producing actions, camera
motion and human/object interaction. Retain quiet and static examples too.
Broader prompt-only training is the cheapest first comparison. Introduce each
data/loss change separately from the unchanged-data continuation so its value
can be measured. Do not assert that a larger corpus cures a capacity deficit.

Proposed first additional-data experiment: 1,000–2,000 curated paired clips,
after auditing existing sources, mixed with the teacher-rollout objective in
a separate correctly noised forward. This size is an engineering pilot budget,
not a paper-derived optimum. Record whether each clip is real or synthetic,
source rights, prompt/latent alignment, speech transcript where present, shape,
and held-out exclusion. Avoid asynchronous soundtracks, irrelevant narration,
watermarks and scene cuts not represented by the caption. Avoid selecting
only attractive still frames when motion and audio are the failures.

## BVD assessment

[LAION-BVD](https://projects.laion.ai/bvd/) contains broad web-video sources
with synthetic video/audio captions. Its reported validation centers on
multimodal representation benchmarks, not recovery of a pruned joint generator.
Its site explicitly restricts release to research and excludes commercial use.
The [download page](https://projects.laion.ai/bvd/download.html) distinguishes
URL-only metadata releases from other subsets. A URL inventory is not an
immediately usable set of aligned H3 latents. Consider a separately tracked
research subset, but exclude it from the intended consumer-release lineage
unless suitable permission is established. No BVD media was downloaded.

## Research findings and their limits

| Evidence | Practical consequence |
|---|---|
| [TinyFusion paper](https://arxiv.org/abs/2412.01199) and [implementation](https://github.com/VainF/TinyFusion) optimize pruning for post-adaptation recoverability and use hidden-state KD that masks extreme activations. | If longer recovery stalls, test token-aligned, outlier-resistant representation transfer against our mean/RMS summaries. Choose masks by recovered quality, not immediate activation scores. Their image results do not establish H3 audio recovery. |
| [FastLightGen](https://arxiv.org/html/2603.01685v3) combines block adaptation with distribution matching; its preferred tradeoff removes about 30% of parameters, with larger cuts degrading motion and quality. | Our approximately half-size target is more aggressive. Keep an approximately 20B fallback and judge motion separately. Later distribution matching may improve a coherent model; do not demand final release quality from bootstrap alone. |
| [VDN-H3 training code](https://github.com/OpenVDN/vdn-minimax-h3) specifies 200 per-layer alignment updates, 500 end-to-end branch updates, 2,000 LoRA/branch co-adaptation updates and 250 DMD updates. | This is direct H3 evidence for staged adaptation and distinct media-latent versus prompt-only training. It modifies attention while retaining the backbone; it is not evidence for a half-size H3 or a transferable wall-time budget. |
| [VDN-H3 architecture](https://openvdn.github.io/) retains local softmax attention and adds long-range linear memory, calibrating new computation before joint adaptation. | Borrow the alignment principle if needed: supervise what a retained student interval must replace, then jointly recover. Do not add a new attention architecture to the current pruning control. |
| [LarryVRH Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) documents differing static-detail and fast-motion behavior, with 6–8 steps preferred for some cases. | Add static and high-motion comparisons to the held-out panel. Use as a separate correctly configured baseline; never merge its adapter into our different pruned block layout without compatibility work. |
| [fal H3 Max](https://fal.ai/learn/devs/introducing-h3-max-by-fal) reports substantial post-training data and separate human assessments of adherence, aesthetics and overall preference. | Add those dimensions alongside speech, synchronization and motion. The public report does not disclose enough to reproduce its data/objective or infer parameter pruning. Hosted speed is not a consumer-GPU measurement. |

## Ordered experiments

1. Finish compact prediction parity and decoded step-200 inspection. Job 6627
   remains pending resources at this research check; longer recovery is not yet
   running. Retain the completed numerical comparison and all checkpoints.
2. Resume unchanged hard recovery to 500. Prepare cached text and native-shape
   buckets from all supplied prompts while this control runs.
3. Compare broader prompt coverage with the control on a fixed held-out panel.
   A changed manifest needs explicit dataloader-state handling on resume.
4. If needed, compare paired-data supervision or improved intermediate feature
   transfer one at a time. These are new treatments, not proven fixes. Use the
   same parent checkpoint, prompts, seeds and measured compute budget.
5. Continue the best supported recipe to 1,000/2,000, and up to 4,000 if progress
   warrants it. If capacity or mask choice is the limiting factor, change the
   pruning target or alignment method instead of only increasing repetitions.
6. Proceed to call distillation and target-precision QAT/QAD after coherent
   joint generation is recovered. Baseline, bootstrap and final quality gates
   are distinct; improvements during later training remain possible.

No reviewed result guarantees recovery of this particular mask. Success must
be established through decoded joint media, held-out generalization and actual
consumer runtime measurements, with an explicit viable fallback in size.

## Completed pilot check: September 7, 2026 UTC

Both 200-update pilots completed successfully and retain their checkpoints.
Hard job 6533 took 1:29:58 (about 6.00 GPU-hours on four GPUs); annealed job
6535 took 1:49:57 (about 7.33 GPU-hours), excluding its separate 25:40
numerical retry 6534. These are equal-update, not equal-compute comparisons.

| Candidate | Video endpoint error | Audio endpoint error | Reduction from initial video / audio |
| --- | ---: | ---: | --- |
| Hard, step 200 | 2.038985 | 4.335886 | 2.008% / 11.899% |
| Annealed, step 200 | 2.064647 | 4.402529 | 0.775% / 10.545% |

These are two fixed held-out canaries, not perceptual quality percentages.
Earlier FP32 update checks passed; this check reconfirmed successful SLURM exits.

Static audit 6627 completed in 17:16. Compact 24-block export and masked
checkpoint predictions were bit-identical for video and audio on the tested
64x64, five-frame input, with 5,216 sampled parameter values matching. This
establishes parity only for the tested configuration. Four BF16 sample MP4s
were generated and passed media-format checks; perceptual and audio review
remain outstanding. Next: review those clips against matched parent samples,
then decide whether to execute the separately prepared longer recovery.
No additional training was launched during this capped pilot follow-up.

All 58,669 training prompts finished uploading with SHA256
42369c2fe0ba2356a861ef83453bcb6303a48e2e5b5bbb1ee7b25548080816f5.
Cached-embedding audit 6631 failed with host-memory exhaustion after 52 seconds;
encoding readiness is unverified. The next data task is a bounded higher-memory
metadata audit before reuse or encoding. This failure did not affect checkpoints.

### Decoded inspection and diagnostic continuation

Sampled frames 0, 60 and 120 from all four hard step-200 clips show multicolored
noise without recognizable prompted subjects. Audio quality remains unassessed.
The numerical gains have not yet produced demonstrated visual recovery.
Training and inference use matching four-call sigma schedules (video shift 12,
audio shift 3). Submitted one-prompt FP32 inference diagnostic 6633, capped at
one hour on four GPUs, using immutable audit-193ac259 code and the existing
compact export. This checks a precision contribution before longer training.
CPU cache-audit retry 6632 requests 32 GB and 20 minutes: the previous failure
occurred during container image import, before dataset inspection. Both jobs
were pending resources at submission. Checkpoints and optimizer remain intact.

### September 7 restart

No full-prompt training was submitted overnight. Local loader/resume changes
pass eight focused tests and pre-commit checks. Full-corpus cache audit job
6649 is submitted from immutable code/full-prompt-audit-v1; submission 6648
was cancelled after a SLURM environment-retrieval failure, before execution.
The cache READY manifest advertises 49,688 records for the 49,700-row source,
so full coverage must be measured and missing embeddings encoded explicitly.
The prepared recovery script now also requires encoder provenance verification.
Do not treat a cached shape or caption match as encoder equivalence.
