# Agent Brief: MiniMax-H3 CUDA Port

## Objective

Land native FastVideo support for MiniMax-H3 on CUDA, to a standard that can be
proposed upstream. This port is on the critical path for the Apple Silicon track
(`docs/design/apple_silicon_minimax_h3.md`) — every downstream stage is blocked
until it exists, so **speed to a correct, parity-verified port matters more than
breadth of features**.

Done when:

- One H3 checkpoint loads and generates through `VideoGenerator.from_pretrained`.
- Every required component has a non-skip local parity PASS against the official
  reference.
- Pipeline smoke and pipeline parity both pass locally.
- The basic example runs and writes non-corrupt video **and audio**.
- `PORT_STATUS.md` is current and the §"Answers owed" table below is filled in.

## Hard constraints

1. **Do not open a pull request.** Push to the working branch and stop. The
   upstream PR opens only after the repo owner approves. This is not negotiable
   and not a judgment call.
2. Branch: `claude/minimax-h3-fastvideo-support-t02662`. Do not push elsewhere.
3. Do not touch `fastvideo/mlx_runtime/`, `fastvideo/layers/quantization/`, or
   anything Apple Silicon. That is a separate track with a separate owner.
4. Do not write speculative component code before real weights are in hand. An
   invented layer name silently corrupts the conversion mapping — see
   `.agents/lessons/2026-05-07_silent-channel-major-packing-bugs.md`.

## Start here

Follow the repo's own workflow, not an improvised one:

1. `.agents/skills/add-model-01-prep/SKILL.md` — stage weights and reference.
2. `.agents/skills/add-model/SKILL.md` — Phases 0–11.

`tests/local_tests/minimax_h3/PORT_STATUS.md` already holds a Phase-0 component
matrix, open questions, and blockers. Update it as you go; do not start a fresh
state file.

## Known blocker

`I001`: huggingface.co is unreachable from the porting sandbox (403 at the proxy
on CONNECT). There is also no H3 reference repo on GitHub — `MiniMax-AI` has no
H3 entry, so modeling code ships inside the HF repo or nowhere.

If HF is still blocked when you start, say so and stop. Do not attempt
workarounds, mirrors, or reconstruction from serving code without asking.

## Scope — already decided, do not relitigate

- **One checkpoint**, not both. Pick the more generally useful of the two once
  the repo listing is readable, and record the choice in `PORT_STATUS.md` (Q006).
- **Video and audio both in scope.** Dropping the audio output head needs
  explicit owner agreement per `add-model` Phase 0; do not drop it unilaterally.
- **MSA sparse attention deferred.** Upstream ships full attention only in the
  initial release; use existing FastVideo attention backends.
- **In-context regeneration:** port it for CUDA correctness, but keep the base
  pass runnable standalone — the Mac track never runs the second pass.

## Structural precedent

Model this on **LTX-2**, not Wan. It is the only in-tree family with a separate
audio VAE, an audio decoding stage, and a refine pass, which maps onto H3's
audio VAE and in-context regeneration:

- `fastvideo/pipelines/basic/ltx2/` — stage chain and variant layout
- `tests/local_tests/ltx2/` — component parity test shape
- `scripts/checkpoint_conversion/convert_ltx2_weights.py` — conversion shape

## Deliverables the MLX track needs

A vanilla port would skip these. They are cheap now and expensive later.

- **Teacher must load in `fastvideo/train/`, not only in inference.** DMD2
  distillation loads the teacher through FastVideo model classes. An
  inference-only port blocks the entire distillation track.
- **Clean per-module boundaries.** MLX parity tests compare module-by-module
  against these torch classes. If the DiT block, VAE, and encoders cannot be
  instantiated and called in isolation, MLX numerics get written blind.
- **The conversion key mapping is a contract.** MLX weight loading reads
  Diffusers-layout safetensors keys directly. Document the emitted key names;
  do not leave them implicit in a regex.
- **Record the sigma/timestep schedule** in `PORT_STATUS.md`. The Mac step
  distillation derives its 2-step and 8-step schedules from it.

## Answers owed

You are the first person with `config.json` access. These block Apple Silicon
sizing decisions — fill them into `PORT_STATUS.md` as soon as they are known,
before finishing the port:

| ID | Question |
|---|---|
| M001 | Dense or MoE, and total parameter count |
| M002 | Does a smaller H3 variant exist or is one planned |
| M003 | Is audio a head on the omni transformer or a distinct module |
| M004 | Can the base pass run standalone without in-context regeneration |
| M005 | Omni encoder memory cost; can its output be precomputed and freed |
| M011 | Denoised latent token count at each resolution — H3-VAE strides and patch size, **not** the ~4k conditioning figure |

Surface `M001` the moment you have it. It decides the Apple Silicon capacity
path and that work can start in parallel with the rest of the port.

## Expect to hit

- **No audio regression metric exists.** `fastvideo/tests/ssim/` is video-only.
  Phase 10 needs a mel-L1 or multi-resolution STFT gate; `fastvideo/eval/metrics/audio/`
  has usable pieces (`clap_score`, `frechet_distance`, `desync`). Raise it
  rather than quietly shipping video-only coverage.
- **`WorkloadType` may not express joint AV.** LTX-2 registers as `T2V` with
  optional audio — follow that precedent unless it genuinely does not fit (Q004).
- **Pre-commit excludes `fastvideo/models/`.** Lint will not catch style there;
  match the surrounding files by hand.

## Out of scope

MLX, quantization, distillation, training recipes, Apple Silicon anything, the
second checkpoint, MSA. Note them in `PORT_STATUS.md` and move on.

## Handoff

When done: push, update `PORT_STATUS.md` to `complete`, and report parity
results plus the answers table. **Then stop and wait for PR approval.**
