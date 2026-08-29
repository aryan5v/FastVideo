# FastH3 14B two-step QAD sprint

This document is the execution prompt for the GPU agent. Treat it as the source
of truth for the sprint. Work on the branch named below, keep the user updated at
each decision gate, and continue until the release artifacts pass the stated
quality gates or a stop condition applies.

## Mission

Produce a consumer-focused FastH3 preview in at most 96 hours. Start from the
released FastH3 Preview v1 four-step checkpoints. Build a depth-pruned student,
distill it to two transformer calls, and specialize it for two deployment
targets:

1. Apple MLX with INT8, INT6, and INT4 artifacts. INT6 is the flagship.
2. NVIDIA Blackwell with NVFP4 linear layers. Retain VSA-H3 attention for the
   primary sparse CUDA release. Treat dense FP4 attention as a separate stretch
   path.

The primary deliverable is a roughly 14.5B-parameter, two-call T2VA student that
preserves synchronized video and stereo audio. The release must include quality,
latency, memory, and exact-format evidence. A smaller file without a measured
generation improvement does not complete the mission.

The user has confirmed that the required MiniMax authorization is in place.
Keep the inherited MiniMax H3 license, notices, use restrictions, and derivative
model disclosures with every weight artifact.

## Branch and ownership

- Repository: `https://github.com/aryan5v/FastVideo.git`
- Branch: `aryan/fasth3-14b-2step-qad-sprint`
- Upstream base at manifesto creation: `hao-ai-lab/FastVideo` commit
  `a159b63c67a1a283ce55813b694524909ea67b15`
- Manifest:
  `docs/design/fasth3_14b_2step_qad_sprint_agent_prompt.md`

Clone the branch into a clean checkout. Do not work in an existing dirty
checkout.

```bash
git clone --branch aryan/fasth3-14b-2step-qad-sprint \
  https://github.com/aryan5v/FastVideo.git FastVideo-h3-14b-qad
cd FastVideo-h3-14b-qad
git status --short --branch
git rev-parse HEAD
```

You own this branch for the sprint. Commit focused code and configuration
changes. Push after each gate that leaves the repository in a useful state. Do
not commit checkpoints, downloaded models, generated videos, W&B state, or large
benchmark outputs. Store them on persistent model or experiment storage.

Do not rebase during a running training job. Record the source commit for every
run so later branch updates cannot make a checkpoint ambiguous.

## Current ground truth

Read these sources before editing:

- `AGENTS.md`
- `fastvideo/train/AGENTS.md`
- `fastvideo/configs/models/dits/minimax_h3.py`
- `fastvideo/models/dits/minimax_h3.py`
- `fastvideo/train/models/minimax_h3/minimax_h3.py`
- `fastvideo/train/methods/distribution_matching/dmd2.py`
- `docs/training/attn_qat.md`
- `fastvideo/train/attn_qat/README.md`
- `examples/train/scenario/qad_wan2_1_mixkit/`
- `examples/train/configs/distribution_matching/kandinsky5/dmd2_t2v_480p_qat.yaml`
- `scripts/checkpoint_conversion/convert_minimax_h3_mlx.py`
- `fastvideo/layers/quantization/nvfp4_qat_train_config.py`
- `examples/inference/basic/basic_fasth3.py`
- `examples/inference/basic/mlx_fasth3.py`
- `docs/getting_started/installation/mps.md`

The following facts were verified when this manifesto was written:

- Base H3 has 50 main blocks, hidden size 5376, 56 attention heads, head
  dimension 128, FFN dimension 14336, two refiners, 24 video latent channels,
  and 32 audio latent channels.
- The released FastH3 V1 VSA/Data-Free checkpoint uses four DiT calls, 90%
  VSA-H3 sparsity, tile size 64, and was published at training step 1300.
- VSA/Data-Free requires the VSA-H3 attention path. Dense attention is not a
  valid drop-in replacement for those weights.
- The dense/data-free ablation is a separate four-call checkpoint published at
  training step 1000.
- The current modular H3 training wrapper supports joint T2VA SFT, uses video
  scheduler shift 12 and audio scheduler shift 3, and rejects non-Torch-SDPA
  training backends.
- Generic modular DMD2 assumes one primary latent state. It does not yet provide
  the complete joint H3 video/audio contract required by this sprint.
- FastVideo already has an NVFP4 straight-through training configuration and an
  Attn-QAT training backend. They require H3 integration and shape validation.
- MLX conversion already supports `int8`, `int6`, and `int4` H3 checkpoints.
  It assumes the existing H3 architecture and four-call AdaLN cache until the
  sprint changes it.

Use these released checkpoints:

| Role | Hugging Face repository |
|---|---|
| Base quality teacher | `MiniMaxAI/MiniMax-H3` |
| Sparse four-call initializer | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` |
| Dense four-call initializer | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree` |
| Optional compact adapters | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` |

Download `checkpoint_metadata.json`, model configs, and adapter manifests with
the weights. At manifesto creation, the VSA/Data-Free HEAD was
`b65818d41939b5085451074fe8ca8b799f8d4921` and the Dense/Data-Free HEAD was
`f624f08c6c279ab43534c003e556fc5b295b6558`. Resolve and record the revision
actually downloaded rather than assuming these values remain current. Search
the mounted cluster storage and available git branches for the exact FastH3 V1
training configuration before recreating it. Limit that search to two hours.
The public repository says that the full training recipe is still pending, so
the sprint must remain able to proceed from checkpoint metadata and the modular
trainer.

## Required outcome

Aim to publish these artifacts:

| Artifact | Required behavior |
|---|---|
| `FastH3-14B-2Step-BF16-Preview` | Dense reference, two DiT calls, T2VA |
| `FastH3-14B-2Step-MLX-INT8` | Apple quality tier |
| `FastH3-14B-2Step-MLX-INT6` | Apple default tier |
| `FastH3-14B-2Step-MLX-INT4` | Apple memory tier with precise coverage disclosure |
| `FastH3-14B-2Step-VSA-NVFP4` | CUDA default, NVFP4 linears plus trained VSA-H3 attention |

A dense full-FP4-attention checkpoint is a stretch artifact. Do not block the
VSA plus NVFP4-linear release on it.

The Monday checkpoint at hour 48 may carry an `Alpha` suffix. The 96-hour
checkpoint may carry a `Preview` suffix after it passes the full gate.

## Non-negotiable constraints

1. Preserve joint video and audio generation. A video-only student does not
   count.
2. Preserve the shared base noise clock and the modality-specific shifts. Do
   not force video and audio onto one shifted sigma schedule.
3. Keep text, audio, and cross-modal paths dense when using VSA-H3. Only the
   eligible video-to-video region may use the trained sparse map.
4. Apply fake quantization to the student only during QAD. Keep the teacher and
   critic high precision.
5. Use real target arithmetic during validation. A BF16 model with a quantized
   filename does not count.
6. Do not silently fall back from Attn-QAT or NVFP4. Print and save the resolved
   attention backend, quantization method, GPU architecture, and fallback
   reason.
7. Preserve the full 50-block and four-call checkpoints. Write every student to
   a new output directory.
8. Do not change the video VAE, audio VAE, text encoder, or token packing during
   the sprint unless evidence shows a correctness defect.
9. Train at 832 by 480 and 124 frames first. Validate other shapes only after
   the main gate passes.
10. Use one fixed prompt and seed set across teacher, student, and quantized
    comparisons.

## Executive training decision

Use QAT followed by QAD as the main path. This preserves the successful
FastWan-QAD ordering:

1. Teach the student its deployment arithmetic.
2. Distill the quantized student to the target number of calls.
3. Keep the real-score teacher and fake-score critic high precision.

The sprint adds a short consistency bootstrap before DMD2 because H3 generates
two coupled latent streams and the target is an aggressive two-call schedule.
The bootstrap stabilizes the trajectory. QAD remains the final quality recovery
stage.

Use this objective order:

```text
released four-call FastH3 V1
  -> depth-pruned four-call BF16 recovery
  -> target-specific QAT recovery
  -> two-call discrete consistency bootstrap
  -> two-call QAD with DMD2 distribution refinement
```

Do not replace QAD with plain trajectory regression for the release candidate.
Offline trajectory matching is useful for initialization, schedule selection,
and a fallback checkpoint if DMD2 remains unstable.

## Student architecture

The short sprint uses depth pruning and retains all tensor widths. Width pruning
would require reshaping nearly every projection and needs a longer recovery
run.

The local parameter inventory produced these estimates. Verify them against the
downloaded checkpoint before publishing them:

| Candidate | Main blocks | Estimated parameters | Relative DiT work per call |
|---|---:|---:|---:|
| Teacher | 50 | 35,049,751,296 | 1.00 |
| Primary student | 20 | 14,526,541,056 | about 0.40 |
| Quality fallback | 24 | 17,262,969,088 | about 0.48 |

The primary student keeps:

- hidden size 5376;
- 56 heads with head dimension 128;
- FFN dimension 14336;
- both token refiners;
- video and audio input/output projections;
- time embeddings and modality-specific timestep handling;
- the trained VSA gate tensors for selected VSA blocks.

The theoretical DiT-work ratio for a 20-block, two-call student against the
50-block, four-call release is `20 / 50 * 2 / 4 = 0.20`. Report this only as an
arithmetic upper bound. Measure end-to-end latency because the text encoder,
VAE decoders, communication, and kernel efficiency do not scale by that ratio.

### Select the blocks

Build two 20-block candidates:

1. An activation-selected candidate.
2. A uniform-depth control candidate.

Always retain the first and final main blocks. Score the remaining blocks on at
least 256 representative packed T2VA examples. The score must include:

- output change when a block is skipped;
- video-row reconstruction change;
- audio-row reconstruction change;
- cross-modal attention or representation change;
- timestep coverage across high, middle, and low noise;
- prompt categories with speech, sound effects, music, motion, and multiple
  shots.

Use the PARE paper as guidance for structure-aware importance scoring, but do
not implement its adaptive router during this sprint. H3 has a different packed
audio-video layout, and a router adds training and runtime risk.

Remap selected teacher block indices to contiguous student indices. Save the
mapping in checkpoint metadata. Teach every loader, exporter, MLX converter,
and inference config to derive block count from the checkpoint rather than
assuming 50.

If neither 20-block candidate produces at least 10 usable results in the first
12-prompt gate after 400 recovery steps, switch to the 24-block fallback. Do
not spend the remaining sprint trying to rescue a collapsed 20-block model.

## Implement joint H3 distillation

Add the narrowest reusable H3 support to the modular training stack. Do not fork
the legacy trainer.

The H3 model wrapper must expose a joint latent state and predictions for both
modalities. The distillation method must support:

- video and audio noisy inputs;
- video and audio predictions;
- video scheduler shift 12;
- audio scheduler shift 3;
- a shared base noise amount before each shift;
- conditional-only H3 inference with guidance scale 1;
- packed text, video, and audio row metadata;
- dense teacher and critic attention;
- dense or VSA student attention;
- one data-parallel sample per packed document;
- sequence-parallel synchronization of the sampled base noise amount.

Normalize video and audio losses independently before combining them. Start
with equal modality weights after normalization:

```text
video_loss = mse(video_prediction, video_target) / detached_video_target_energy
audio_loss = mse(audio_prediction, audio_target) / detached_audio_target_energy
total_loss = video_loss + audio_loss
```

Clamp the energy denominator to a measured safe floor. Log raw and normalized
losses separately. Do not let the larger video tensor hide audio collapse.

Add focused CPU or small-CUDA tests for:

- shared base noise and distinct shifted sigmas;
- joint output shapes;
- loss normalization;
- two-call rollout for both modalities;
- student-only quantization selection;
- block-map loading and export;
- dense versus VSA configuration rejection;
- checkpoint resume and export metadata.

## Phase A: establish baselines

Before training, generate a locked baseline matrix at 832 by 480, 124 frames,
and the release four-call schedule.

Run at least these checkpoints:

- base H3 dense teacher;
- FastH3 V1 VSA/Data-Free;
- FastH3 V1 Dense/Data-Free.

Use 12 quick-gate prompts and one seed. Include:

- two exact spoken sentences;
- one cat or dog with an expected non-speech vocalization;
- one visible mechanical sound source;
- one musical scene;
- one silent or nearly silent scene;
- two human close-ups;
- two large-motion scenes;
- one two-shot scene transition;
- one prompt with on-screen text.

Save the prompt text, negative prompt if any, seed, schedule points, scheduler
shifts, output dimensions, frame count, audio sample rate, backend receipts,
wall time, DiT time, decode time, peak memory, and artifact paths.

The baseline is a gate. Do not begin a long run until the agent can reproduce a
published four-call output with synchronized audio.

## Phase B: recover the pruned four-call student

Initialize each student by copying the selected FastH3 V1 blocks and all shared
modules. Use the dense V1 checkpoint for the MLX track and the VSA V1 checkpoint
for the CUDA sparse track. Do not run VSA weights through dense attention.

Use a short high-precision recovery before target quantization:

- 200-step smoke checkpoint;
- 400-step candidate-selection checkpoint;
- continue the winner to 600 to 1,000 steps only if validation improves;
- learning-rate search centered on `1e-6` and `2e-6`;
- BF16 model execution with FP32 optimizer state;
- EMA checkpoint selection;
- gradient clipping at 1.0 unless measured gradients justify a change.

Use teacher velocity matching, selected hidden-state matching, and the normal H3
T2VA denoising target. Keep feature loss small enough that it cannot dominate
video and audio prediction losses. Record the exact weights in the config.

At 400 steps, compare activation-selected and uniform students on the locked 12
prompts. Select one block map for both deployment tracks if possible. If VSA and
dense tracks require different maps, document the evidence and keep the maps
separate.

## Phase C: run target-specific QAT

Fork the recovered four-call student into two main branches.

### MLX INT6 branch

Implement weight fake quantization that matches the actual MLX affine INT6
export. Match group size, scale calculation, packing range, rounding, and
higher-precision exceptions. Do not train against an approximate INT6 format
and export a different one.

The current MLX target is affine group-64 weight quantization. Verify that from
the converter and runtime manifest before coding. Keep activations in BF16.

Start with these possible higher-precision modules and remove exceptions only
after sensitivity tests:

- norms;
- timestep embeddings;
- AdaLN-sensitive tensors;
- video and audio input/output projections;
- Q/K normalization;
- first and final main blocks.

Run 300 to 600 four-call QAT recovery steps. Save a real INT6 export at each
validation checkpoint and run at least one forward with the exported artifact.

Convert INT8 and INT4 from the final BF16 or INT6-robust checkpoint. INT8 and
INT4 do not need independent full QAD runs in this sprint. If INT4 quality is
poor, use a measured mixed-precision allowlist and disclose exact coverage.

### NVIDIA NVFP4 plus VSA branch

Use `nvfp4_qat_train` for eligible attention and FFN linear layers. Retain the
VSA-H3 attention algorithm and train its existing sparse gates. This artifact
uses NVFP4 linears plus sparse attention. It is not a full-FP4-attention model.

Run on SM100 GB200 or B200 first. Verify real FP4 forward execution and the
straight-through backward. Save the receipt. Run 300 to 600 four-call QAT
recovery steps.

Keep boundary modules in BF16 until sensitivity evidence supports quantizing
them. Record the exact module coverage in a machine-readable manifest.

### Dense full-FP4 stretch branch

Only start this branch after the primary NVFP4 plus VSA run is healthy. Start
from the dense/data-free student, use `ATTN_QAT_TRAIN` for the student, and keep
the teacher and critic on dense high-precision attention.

Validate on SM120 RTX 5090 because `ATTN_QAT_INFER` is an SM120 inference path.
DGX Spark is SM121. Treat Spark support as a separate kernel-dispatch task. A
silent dense fallback is not full FP4.

Do not combine the VSA checkpoint with dense `ATTN_QAT_INFER`. They are
different trained attention paths.

## Phase D: bootstrap the two-call trajectory

Use the released four-call student as the immediate trajectory teacher and Base
H3 as the quality teacher.

Cache video and audio states for the four-call schedule. Every cache record must
include:

- prompt and encoder revision;
- seed;
- packed layout metadata;
- scheduler points;
- video and audio shifted sigmas;
- noisy and intermediate video latents;
- noisy and intermediate audio latents;
- final video and audio latents;
- teacher checkpoint revision and attention backend.

Run a small two-call schedule search. Do not copy Wan timesteps. Score candidate
schedules against teacher trajectories for both modalities. Select the schedule
using the locked validation prompts and a held-out cache split.

Run a discrete consistency warm-up for 300 to 600 steps while the student still
sees target fake quantization. Use per-modality normalized losses. The high-noise
call must preserve layout and diversity. The low-noise call must recover detail,
audio semantics, and synchronization.

TurboT2VA reports that a progressive consistency curriculum with per-modality
normalization stabilizes large joint video-audio distillation. Use that result
to set the order of training, not as a reason to port its complete LTX-specific
runtime.

## Phase E: finish with two-call QAD

QAD is the release path.

Build three roles:

| Role | Precision and architecture |
|---|---|
| Student | 20-block or fallback 24-block, target fake quantization, two-call rollout |
| Teacher | Base H3, 50 blocks, dense high precision, frozen |
| Critic | Student-sized dense high-precision model, trainable |

Initialize the critic from the recovered high-precision student unless a short
ablation shows that another initialization converges faster. The critic models
the generated distribution and does not need VSA or target quantization.

Adapt modular DMD2 to the joint H3 state. Keep the successful FastWan-QAD
cadence as the starting point:

- one student update for every five critic updates;
- student learning-rate search centered on `2e-6`;
- critic learning-rate search centered on `2e-6`;
- score timestep ratio range `0.02` to `0.98`;
- guidance scale 1 for H3 conditional-only training;
- 600 to 1,200 QAD steps;
- checkpoint and validate every 100 steps;
- no unverified unconditional branch.

The Wan three-call timestep list is not valid for H3. Use the schedule selected
in Phase D. Preserve the separate video and audio scheduler shifts throughout
student rollout, teacher scoring, critic noising, and target conversion.

Run the INT6 and NVFP4 QAD branches in parallel when 24 or more GPUs are
available. With 16 GPUs, prioritize the NVFP4 plus VSA branch through a stable
checkpoint, then resume the INT6 branch from the shared BF16 and consistency
checkpoints.

Watch for mode collapse. Every 100 steps, generate at least four seeds for two
fixed prompts. Stop and roll back if motion or composition diversity contracts
while single-sample sharpness rises.

## Data plan

Do not wait for a large new corpus before starting. Reuse the FastH3 V1 prompt
distribution and any mounted, authorized, preprocessed T2VA shards first. The
first long job must be able to start from prompts and cached latents already on
the cluster.

Build a 10,000 to 50,000-example quality pilot in parallel. Use this initial
mixture as a sampling target, then change it only when validation supports the
change:

| Share | Data |
|---:|---|
| 50% | Existing FastH3 data-free prompt distribution and hard validation prompts |
| 25% | Rights-cleared real multi-shot audiovisual clips with accurate integrated captions |
| 15% | Speech, dialogue, lip motion, multilingual speech, and speaker-turn cases |
| 10% | Visible sound events, music, ambience, and intentional silence |

Candidate public sources:

- [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) has
  43,751 CC-BY videos, time-coded speech, scene metadata, and an audiovisual
  correlation field. Use only the latest permitted revision and retain source
  attribution and deletion handling.
- [Full Modality Video Caption](https://huggingface.co/datasets/ngqtrung/full-modality-video-caption)
  has 55,940 ten-second clips with visual, audio, and integrated captions.
  Audit the underlying media provenance before using it beyond a pilot.
- [VGGSound](https://robots.ox.ac.uk/~vgg/data/vggsound/) provides over 200,000
  visible-source sound clips under its stated CC-BY 4.0 terms. Retain original
  ownership and attribution metadata.
- [VALID](https://huggingface.co/datasets/ontocord/VALID) is a large
  audiovisual preview. Do not make it a sprint dependency. Audit availability,
  provenance, and per-item rights first.
- [AudioCaps](https://audiocaps.github.io/) is useful for sound-caption
  evaluation and prompt enrichment. It is not the core video corpus.

For every ingested clip:

- cut a temporally coherent 5 to 10 second segment;
- retain source URI, creator, license, and attribution;
- reject hard cuts that do not match the caption;
- reject clipped, desynchronized, or inaudible audio;
- compute video perceptual hashes and audio fingerprints;
- remove cross-source duplicates and evaluation overlap;
- produce an integrated prompt with visual action, camera, speech transcript,
  speaker turns, sound effects, ambience, and music;
- precompute text embeddings, video latents, and audio latents with pinned
  component revisions.

Use quality scoring for selection, not as an unchecked training reward. A good
candidate selector combines prompt alignment, video quality, audio-caption
alignment, speech transcription accuracy, and A/V synchronization. Keep a
diversity quota so the selector does not reduce the dataset to static close-ups.

S2Q-VDiT reports that quantized video models are sensitive to calibration data
and benefits from salient sample selection and token-weighted distillation. Use
this as a contained experiment:

1. Rank calibration samples by target-format output sensitivity.
2. Keep a diverse top subset rather than random calibration alone.
3. Weight the quantization-recovery loss toward rows or tokens with large
   teacher influence.
4. Compare against random calibration with the same sample count.

Do not merge the technique unless it wins the locked validation gate.

### Calibration-driven mixed precision

The linked [Tencent Hunyuan post](https://x.com/TencentHunyuan/status/2093572224342954019)
describes compressing Hy4-preview from roughly 1.5 TB to about 200 GiB by using
calibration data to assign different low-bit formats to different layers under
one storage budget. The reported language-model formats and bit rates are not
directly transferable to H3, but the allocation principle is useful.

Run one bounded H3 sensitivity sweep after the uniform INT8, INT6, and INT4
exports exist:

1. Use the locked calibration set and measure the output change caused by
   quantizing one module group at a time.
2. Score video, audio, speech, and synchronization separately. Do not optimize
   only latent MSE or average video quality.
3. Build one mixed INT4 artifact under the same byte budget as the uniform INT4
   artifact. Spend higher precision on the most sensitive layers and reclaim
   the budget from insensitive layers.
4. Keep the runtime format set small. Prefer existing INT4, INT6, and INT8
   kernels over introducing a one-off sub-two-bit format during this sprint.
5. Compare the mixed artifact against uniform INT4 using the same prompts,
   seeds, two-call schedule, runtime, and model-size accounting.

Limit the sweep and export work to six hours. Promote it only if it materially
improves INT4 quality at equal or lower on-disk size without adding a fallback
or a large latency penalty. This is an INT4 rescue experiment. It must not
replace the fixed-format INT6 flagship, the NVFP4 CUDA branch, or QAD.

## Research experiments

The experiments below may improve quality, but none may delay the primary QAD
checkpoint.

### Progressive T2VA consistency

The [TurboT2VA paper](https://arxiv.org/abs/2608.24674) uses per-modality
normalization, a discrete consistency warm-up, continuous consistency, and
joint consistency plus distribution matching. The sprint adopts the first and
last parts. Add continuous-time consistency only if the two-call discrete
bootstrap is stable by hour 48.

### Two-call objective specialization

The [DUET paper](https://arxiv.org/abs/2608.09637) assigns the high-noise call
to a consistency expert and the low-noise call to a DMD expert to preserve
diversity and quality. Do not ship two full 14B experts in a consumer release.
If the main QAD run becomes mode-seeking, test step-weighted losses or small
step-specific adapters on the shared student. Keep the experiment behind an
explicit flag and compare total model size.

### Parallel Decoding Distillation

[FastGen PDD](https://research.nvidia.com/labs/genair/pdd/) predicts multiple
teacher sub-interval updates in one model call and reports stronger diversity
than adversarial few-step methods. Its public training code was not available
when this manifesto was written. If an authorized internal implementation is
mounted on the cluster, run one bounded compatibility investigation after the
QAD baseline. Otherwise record `Not run: training code unavailable` and move
on.

### Pruning and routing

The [PARE paper](https://arxiv.org/abs/2605.27336) motivates structure-aware
depth scoring and preserves motion-sensitive paths. Use its scoring ideas for
static block selection. Defer adaptive routing and width pruning to the longer
14B program.

### Reward feedback

Do not start reinforcement learning during the main sprint. Use reward models
for data selection and evaluation. If QAD passes early, one small reward-weighted
fine-tuning ablation may target known failures. It must retain a teacher or
reference regularizer and pass the diversity gate.

## Hardware plan

At startup, record:

```bash
nvidia-smi -L
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("devices", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, props.total_memory, props.major, props.minor)
PY
```

Record the scheduler, node count, GPUs per node, filesystem used for
checkpoints, and whether the jobs survive terminal disconnects. A detached
launcher is not evidence that a job is still running. Confirm logs and
persistent checkpoints.

Suggested allocation with 32 GPUs:

| GPUs | Work |
|---:|---|
| 8 | Base and four-call teacher cache generation |
| 8 | Activation-selected dense student |
| 8 | Activation-selected VSA student |
| 8 | Uniform control, validation, or 24-block fallback |

After block-map selection:

| GPUs | Work |
|---:|---|
| 12 to 16 | NVFP4 plus VSA consistency and QAD |
| 12 to 16 | MLX INT6 consistency and QAD |
| Remaining | Validation and cache generation |

With 16 GPUs, do not run two underpowered DMD2 jobs. Finish one stable NVFP4
plus VSA QAD checkpoint, then resume INT6 from the shared recovery checkpoint.

Use sequence parallelism because one packed H3 document has a long sequence.
Choose an SP size that divides 56 heads and confirm the packed audio/video
layout remains correct. Derive HSDP dimensions from the actual node layout.
Run a one-batch forward, backward, optimizer step, checkpoint, resume, and
validation before a job longer than 100 steps.

## Four-day schedule

### Hours 0 to 6

- Verify repository, hardware, storage, checkpoints, and published baseline.
- Inspect checkpoint metadata and locate any internal V1 training config.
- Create the locked prompt set and baseline run manifest.
- Add block-count and block-map configuration without changing default H3.

Gate: a published four-call checkpoint generates valid video and audio on the
available hardware.

### Hours 6 to 18

- Score blocks on at least 256 examples.
- Build activation-selected and uniform 20-block students.
- Add conversion and loading tests for variable block counts.
- Start teacher trajectory caching.
- Run one-step distributed training and resume tests.

Gate: both candidates load copied weights and complete a finite optimizer step.

### Hours 18 to 36

- Run four-call BF16 recovery.
- Compare both candidates at steps 200 and 400.
- Select the block map or switch to 24 blocks.
- Start INT6 and NVFP4 QAT recovery on the winner.

Gate: at least 10 of 12 quick prompts remain usable. Video and audio losses are
both improving.

### Hours 36 to 48

- Complete the short QAT recovery.
- Search two-call schedules from cached trajectories.
- Start discrete consistency warm-up.
- Export preliminary INT6 and NVFP4 artifacts.

Hour-48 handoff: provide the BF16 pruned checkpoint, first two-call samples,
preliminary target-format samples, and the full status table. Mark the artifact
`Alpha` unless it already passes the release gate.

### Hours 48 to 72

- Finish the consistency bootstrap.
- Start two-call QAD on the target branches.
- Validate every 100 steps.
- Monitor diversity and audio intelligibility.
- Ingest the quality pilot only after its rights and deduplication checks pass.

Gate: two-call QAD improves or matches the consistency checkpoint without mode
collapse.

### Hours 72 to 88

- Select EMA checkpoints.
- Convert MLX INT8, INT6, and INT4.
- Export the NVFP4 plus VSA checkpoint and coverage manifest.
- Run the locked 32-prompt, multi-seed release evaluation.
- Benchmark target hardware.

Gate: both flagship formats meet the release criteria below.

### Hours 88 to 96

- Fix packaging and reproducibility defects only.
- Re-run failed release checks.
- Write model cards, exact commands, license files, and benchmark tables.
- Push the final code and configuration commits.
- Upload weights to the user-approved model repositories.

Do not start a new training idea after hour 88.

## Evaluation and release gates

Use one locked 32-prompt suite. Keep it outside the training and calibration
data. Use the same seeds and component revisions for every comparison.

The suite must cover:

- exact dialogue and speech intelligibility;
- multi-speaker turns;
- non-speech animal sounds;
- visible mechanical and impact sounds;
- music, ambience, and silence;
- close faces and hands;
- rapid motion and camera motion;
- multiple shots;
- text rendering;
- landscape, portrait, and square shapes in the final extended gate.

Report these groups separately:

| Group | Measurements |
|---|---|
| Video quality | Human pairwise preference, DOVER or available perceptual score, artifacts |
| Prompt alignment | ViCLIP or available video-text score, category-specific human review |
| Audio | CLAP or available audio-text score, clipping, loudness, semantic correctness |
| Speech | WER and CER against exact prompted dialogue |
| A/V synchronization | SyncNet or available A/V sync score plus manual visible-source checks |
| Diversity | Four seeds on eight prompts, pairwise feature distance, human collapse review |
| Runtime | Cold and warm wall time, DiT time, VAE time, peak active memory, reserved memory, swap or host offload |

Release criteria:

1. The pruned four-call BF16 student produces at least 28 acceptable results in
   the 32-prompt suite.
2. The two-call BF16 student produces at least 26 acceptable results and does
   not show category-wide speech, audio, face, or motion collapse.
3. INT6 and NVFP4 each preserve at least 28 of 32 BF16-student results without a
   material new defect.
4. Known-speech WER may not regress by more than two absolute percentage points
   against the two-call BF16 student without explicit user approval.
5. The diversity check may not show repeated composition or motion across seeds
   that is absent from the four-call student.
6. No artifact may contain NaN or Inf, broken muxing, missing stereo audio,
   persistent line noise, or an unreported backend fallback.
7. Make a quality-improvement claim only when the new model wins a locked
   pairwise comparison against FastH3 V1 with a confidence interval that clears
   a tie. Otherwise state that the release targets quality preservation.

INT4 may ship as an experimental memory tier if it remains coherent but misses
the flagship quantization gate. Document its exact limitations.

## Target-hardware validation

### Apple MLX

The GPU agent may prepare MLX artifacts on Linux, but final MLX conversion or
validation may need the user's M4 Max. Produce a self-contained Diffusers
student checkpoint and exact conversion command:

```bash
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root /path/to/FastH3-14B-2Step/transformer \
  --out /path/to/FastH3-14B-2Step-MLX \
  --formats "int8 int6 int4"
```

Update the converter so the manifest carries the student block map, block
count, two-call schedule, modality shifts, quantization coverage, and AdaLN
cache timesteps. Do not hardcode the old four-call schedule.

The final Mac gate uses an M4 Max with 36 GB unified memory. Load the Qwen text
encoder, cache conditioning, unload it, then load the DiT. Do not keep the text
encoder and DiT resident together.

### NVIDIA Blackwell

Benchmark BF16, NVFP4 plus VSA, and the dense full-FP4 stretch artifact when it
exists. Report GPU model and compute capability. Verify the resolved kernel in
logs.

- GB200 or B200 validates the SM100 training and sparse inference path.
- RTX 5090 validates SM120 dense `ATTN_QAT_INFER` when available.
- DGX Spark validates SM121 NVFP4 linear execution. Do not claim native FP4
  attention unless an SM121 kernel actually binds.

## Stop and fallback rules

- If 20 blocks fail the 400-step recovery gate, use 24 blocks.
- If two-call consistency fails but three calls are stable, continue QAD on the
  two-call branch while preserving the three-call checkpoint. Ask the user
  before changing the public target.
- If DMD2 diverges, restore the last stable consistency or QAD checkpoint,
  lower the learning rate, and run one bounded retry. Do not restart from zero.
- If QAD sharpens frames while collapsing diversity, restore the consistency
  checkpoint and reduce DMD weight or duration.
- If audio loss improves but decoded speech remains unintelligible, check the
  audio target sign, scheduler shift, packing, and decoder revision before
  adding more data.
- If VSA weights run through dense attention, stop. Correct the attention path
  rather than interpreting the degraded output as a pruning failure.
- If full FP4 attention fails, ship NVFP4 linears plus VSA. State the boundary.
- If a public dataset cannot pass provenance review during the sprint, omit it.
  Existing authorized data and data-free QAD remain the main path.

## Validation before each push

Run the smallest relevant checks after every code change. Before the final
push, run:

```bash
pre-commit run --files <changed paths>
pytest <focused H3 training and conversion tests> -q
```

Follow repository excludes. Do not invoke formatter, linter, or type checker
directly when pre-commit intentionally excludes a path.

For GPU code, a syntax or CPU test is not enough. Record at least one real
forward, backward, optimizer step, checkpoint, resume, export, and inference
run on the target architecture.

## Progress reports

Send the user a report at hours 6, 18, 36, 48, 72, 88, and completion. Report a
material failure immediately. Use this table:

| Field | Value |
|---|---|
| Time and phase | |
| Repository commit | |
| Run ID and output path | |
| Hardware and GPU count | |
| Student architecture and block map | |
| Teacher, critic, and student checkpoints | |
| Attention and quantization receipts | |
| Data mixture and sample count | |
| Step, loss, and gradient state | |
| Latest validation result | |
| Memory and timing | |
| Decision made | |
| Next gate | |
| Blocker or risk | |

Do not report a launcher PID as proof that training is running. Include a recent
log timestamp, step, and persistent checkpoint or W&B run.

## Final handoff

At completion, provide:

| Deliverable | Required evidence |
|---|---|
| Branch | Remote branch URL and final commit |
| Code | Focused commits and changed-path summary |
| BF16 student | Repository or persistent path, hash, block map, schedule |
| MLX INT8 | Artifact path, size, coverage, conversion receipt, quality result |
| MLX INT6 | Artifact path, size, coverage, timing, quality result |
| MLX INT4 | Artifact path, size, coverage, quality result, limitations |
| NVFP4 plus VSA | Artifact path, size, kernel receipt, timing, quality result |
| Dense FP4 stretch | Artifact or `Not run` with reason |
| Training | Configs, run IDs, steps, GPU-hours, checkpoints, resume evidence |
| Data | Source counts, licenses, deduplication, preprocessing revisions |
| Quality | Locked table, human review, speech, audio, sync, diversity |
| Runtime | Comparable cold and warm measurements per target |
| Tests | Exact commands and results |
| Remaining work | Concrete risks and next experiment |

Conclude with one recommendation for Apple and one for NVIDIA. State whether
the release improved quality, preserved it, or traded quality for speed. Tie
every performance claim to the exact checkpoint, resolution, frame count,
schedule, hardware, backend, and measurement method.

## Research sources

- [FastH3 Preview v1 release](https://haoailab.com/blogs/fasth3-preview/)
- [FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/)
- [Attn-QAT](https://arxiv.org/abs/2603.00040)
- [DMD2](https://arxiv.org/abs/2405.14867)
- [TurboT2VA](https://arxiv.org/abs/2608.24674)
- [DUET](https://arxiv.org/abs/2608.09637)
- [FastGen PDD](https://research.nvidia.com/labs/genair/pdd/)
- [PARE](https://arxiv.org/abs/2605.27336)
- [S2Q-VDiT](https://arxiv.org/abs/2508.04016)
- [NVIDIA FastGen](https://github.com/NVlabs/FastGen)
