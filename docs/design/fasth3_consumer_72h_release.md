# FastH3 consumer release: three-day execution plan

## Objective and authority

The user revised the sprint direction on September 5, 2026: bring strong
joint video/audio quality and useful speed to consumer NVIDIA GPUs and
Apple silicon, targeting a release candidate in roughly three days. Model
size is flexible. The original approximately 14B, two-call design remains a
research option. Architecture, step count and training stages are means to
the consumer result, not independent release requirements.

Work only on `aryan/fasth3-14b-2step-qad-sprint`. Preserve existing checkpoints,
artifacts and W&B receipts. Use SLURM compute nodes for GPU work, with
`sbatch --export=NIL`. Keep the earlier license/notice requirements.

## Release gates

1. Recognizable prompt semantics, coherent motion, intelligible requested
   speech and usable synchronized stereo audio. Compare decoded outputs with
   branch-matched released FastH3 and Base H3 references.
2. Actual peak device/unified memory and host RAM, including encoding,
   denoising and decoding. Record swapping/offload and whether it is required.
3. End-to-end cold and warm latency on the named consumer machine, alongside
   component times. Weight-only size, GB200 timing and DiT-only timing do not
   establish consumer performance.
4. Reproducible install, exact weight revision/hash, backend and precision
   receipts, explicit fallback behavior, and a working joint-media command.
5. Team dogfooding beyond the four development prompts. Keep a held-out
   evaluation set and publish failures and limitations alongside examples.

Fewer calls primarily reduce execution time. Quantization, retained layers,
activation geometry and component residency determine memory fit. Compare
complete implementations rather than ranking parameter counts alone.

## Hardware targets

The user can provide RunPod RTX 4090/5090 access, an M5 MacBook with 24 GB
unified memory, and an M4 Max Studio with 36 GB, plus team dogfooding.
Connection details remain pending. The current local host is an 8 GB M2;
the available cluster uses GB200s.

| Target | Initial route | Required evidence |
|---|---|---|
| RTX 5090, 32 GB | Healthy VSA four-call model; native NVFP4 linear evaluation | Actual sm120 execution, memory, quality and cold/warm timing |
| RTX 4090, 24 GB | Dense or trained VSA with a supported low-memory linear format and phased components | Actual Ada kernels and full-pipeline memory fit |
| M4 Max, 36 GB | MLX INT6 quality baseline; INT4 memory comparison | Actual full pipeline and matched prompt/seed quality |
| M5, 24 GB | MLX INT4 first, then INT6 if measured headroom permits | OS headroom, swapping and full decode included |
| Other RTX 30/40/50 configurations | Memory-tier presets after the primary lanes work | Test the specific architecture/VRAM combination before claiming support |

NVFP4 hardware execution is a Blackwell route. RTX 30/40 need an appropriate
weight-only/dequantized or other supported execution path, explicitly labeled.
A packed four-bit file alone is not proof of native FP4 execution or speedup.

## First day: establish the best viable baseline

- Finish the existing six 40/36/32-block initializer experiments. Keep their
  immutable maps and compare all four locked prompts before training any.
- Preserve the passing dense/VSA DCP-export parity fix and full VSA gate
  hashes. Do not change attention algorithms while evaluating an export.
- Reuse healthy released four-call V1 models as the delivery baseline. Audit
  the current upstream Mac/runtime work before duplicating it. Validate dense
  checkpoints with dense execution and sparse checkpoints with trained VSA.
- Evaluate low-memory deployment of the intact model: quantized linears,
  phased text/DiT/VAE residency, tiled decode and safe timestep-modulation
  precomputation. Retain sensitive boundary layers at higher precision when
  measured quality requires it.
- If a pruned initializer passes, allow a single bounded 25-step recovery
  trial with FP32 master parameters and Adam moments, BF16 computation,
  branch-matched four-call V1 supervision and corrected 124/37/207 geometry.
  Continue only on decoded improvement. Do not spend the release window
  training a visibly collapsed initializer by default.

## Second day: choose the deployment frontier

- Compare intact quantized four-call and any successfully recovered pruned
  candidate on the actual consumer machines. Choose the quality/memory/time
  tradeoff empirically; different hardware may ship different precisions.
- Use targeted QAT or QAD repair only if a measured quantization regression
  warrants it. Preserve a healthy BF16 anchor. Do not automatically run every
  stage of the original training plan.
- Two-call distillation is optional and must use the exact deployed solver,
  both modalities and the four-call trajectory anchor. It must beat the
  four-call baseline sufficiently to justify its quality cost and schedule.
- Keep full VAE rendering as the quality baseline. Tiny decoding, temporal
  interpolation and reduced-resolution generation are separately labeled
  draft modes, each with its own quality comparison.

## Third day: freeze and dogfood

- Freeze model/quantization/backend combinations that passed the hardware
  and quality gates. Complete held-out multi-prompt, multi-seed checks and
  team dogfooding, including speech, motion, impacts and shot transitions.
- Package downloadable weights, hashes, exact commands, environment versions,
  inherited notices and a tested support matrix.
- Publish cold/warm end-to-end latency and peak memory for each tested target.
  Do not compare a local result with a hosted endpoint's latency as if their
  hardware, resolution and request overhead were equivalent.
- A narrow release with verified quality and hardware support is eligible;
  unsupported tiers stay experimental. If a gate fails, report the actual
  blocker rather than marking the release ready because the date arrived.

## Existing evidence and references

The repaired parity gate passes with zero video/audio error for both dense
and VSA. All 20 VSA gate tensors match by complete SHA-256. The corrected
corpus has 512 real and 64 synthetic records; all latent/text headers were
checked. The older 24-block candidates fail the visual and speech gates,
while all three reference speech clips have zero ASR word error rate.

Hao AI Lab already documents a maintained four-call Mac path and emphasizes
phased residency and measured latency. Reuse this work and verify against
its current source rather than assuming the older sprint checkout contains
all of it: [local FastH3 release](https://haoailab.com/blogs/fasth3-local/).

Keep the original detailed research plan for optional later stages:
[14B/two-call research plan](fasth3_14b_2step_qad_sprint_agent_prompt.md).
