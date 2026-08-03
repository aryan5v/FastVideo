# MiniMax-H3 Port Status

## Summary
- model_family: `minimax_h3`
- workload_types: `T2V`, `I2V`, `V2V`, `T2I` + native audio (AV shim pending, see `Q004`)
- official_ref: unknown — no `MiniMax-AI/MiniMax-H3` GitHub repo exists as of 2026-08-03; reference code expected inside the HF repo
- official_ref_dir: none
- hf_weights_path: `MiniMaxAI/MiniMax-H3`
- local_weights_dir: `official_weights/minimax_h3/` (not populated)
- source_layout: unknown (expected `diffusers`)
- local_tests_readme: `tests/local_tests/minimax_h3/README.md`

## Current Phase
- phase: Phase 0 — Scope And Handoff Gate
- status: blocked
- owner: prep
- last_updated: 2026-08-03

## Component Matrix

Populated from public reporting only. Every row is unverified until
`inspect_hf_layout.py` runs against the real checkpoint.

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| H3-Omni Transformer | dit | port | unknown | unknown | `fastvideo/models/dits/minimax_h3.py` + `fastvideo/configs/models/dits/minimax_h3.py` | not_started | not_started | not_started | I001, Q002 |
| H3-VAE (visual) | vae | port | unknown | unknown | `fastvideo/models/vaes/minimax_h3vae.py` | not_started | not_started | not_started | I001, Q003 |
| Audio VAE | vae | unknown | unknown | unknown | reuse `oobleck.py`/`sa_audio.py` if arch matches, else `fastvideo/models/vaes/minimax_h3_audiovae.py` | not_started | not_started | not_started | I001 |
| Text encoder | encoder | unknown | unknown | unknown | reuse existing bucket entry if identity matches, else new | not_started | not_started | not_started | I001, Q001 |
| Processor (multimodal refs) | generic | port | unknown | unknown | `fastvideo/pipelines/basic/minimax_h3/stages/` | not_started | not_started | not_started | I001 |
| Scheduler | generic | unknown | unknown | unknown | reuse `fastvideo/models/schedulers/` if flow-match, else new | not_started | not_started | not_started | I001 |
| MSA sparse attention | attention backend | defer | unknown | unknown | `fastvideo/attention/backends/` | not_started | n/a | not_started | Q005 |

## Conversion State
- conversion_script: not written
- converted_weights_dir: `converted_weights/minimax_h3/` (not populated)
- source_layout: unknown
- strict_load_status: not_run
- passthrough_components: unknown
- retry_history: none

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| layout inspect | `python .agents/skills/add-model-01-prep/scripts/inspect_hf_layout.py MiniMaxAI/MiniMax-H3 --json` | not_run | blocked by I001 |
| weight download | `python .agents/skills/add-model-01-prep/scripts/download_hf_weights.py MiniMaxAI/MiniMax-H3 official_weights/minimax_h3` | not_run | blocked by I001 |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which text encoder does H3 ship, and does Contextual Omni Representation live in the encoder or in a separate compressor module? | prep | 1 | open | |
| Q002 | Is the H3-Omni Transformer a single joint AV denoiser, or does it carry a dedicated audio output head? Determines whether one DiT config covers both modalities. | prep | 1 | open | |
| Q003 | H3-VAE compression ratio and latent layout (temporal + spatial stride, channel count). Public material says "4x effective sequence length gain" over the prior tokenizer but gives no absolute strides. | prep | 1 | open | |
| Q004 | Does `WorkloadType` need an `AV` member, or is video-with-audio expressible as `T2V` plus a sampling flag? LTX-2 registers as `T2V` with optional audio — confirm that precedent applies. | orchestrator | 8 | open | |
| Q005 | Ship MSA (MiniMax Sparse Attention) as a FastVideo attention backend in the first PR, or defer? Upstream states the initial open-source release is full-attention only, so deferring is the default. | orchestrator | 3 | open | Recommend defer to a follow-up PR |
| Q006 | Which of the two task-specific checkpoints is in scope for the first PR? Per `add-model` Phase 0, scope must be locked on both the variant and modality axes before coding. | orchestrator | 0 | partially_resolved | Modality axis locked: one checkpoint, full output including native audio. Which of the two checkpoints still needs the real repo listing to choose. |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | 0 | all | blocker | `huggingface.co` is unreachable from the porting environment, so weights, `config.json`, and reference modeling code cannot be staged. `add-model-01-prep` cannot complete, which gates every later phase. | Agent proxy returns 403 to `CONNECT huggingface.co:443`; `hf.co`, `cdn-lfs.huggingface.co`, and `hf-mirror.com` all fail to connect. `pypi.org` and `raw.githubusercontent.com` are reachable, so this is a host policy, not a general network failure. | user | open | |
| I002 | 0 | all | major | No official reference implementation is published on GitHub. `MiniMax-AI` has 27 public repos and none is H3; `raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/{main,master}/README.md` both 404. Parity work therefore depends entirely on modeling code shipped inside the HF repo. | GitHub repo search over `org:MiniMax-AI`, 2026-08-03. | prep | open | |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | 0 | environment | How should H3 reference assets reach the port? | Add `huggingface.co` + `cdn-lfs*.huggingface.co` to the environment network policy, then re-run prep. Alternative: run prep on the user's own GPU box and commit the emitted layout/key dumps. | resolved | 2026-08-03: enable HF egress on the environment's network policy. Still 403 as of this commit; the policy change has not taken effect yet. |

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-08-03 | Do not write speculative component code before Phase 0 completes. | `ArchConfig` fields must match `transformer/config.json` one-to-one (`add-model` Phase 1). Guessing that surface produces code that must be discarded, and invented layer names silently poison the conversion mapping. | First PR waits on I001. |
| 2026-08-03 | Model LTX-2 as the structural precedent rather than Wan. | H3 emits joint audio+video and runs a second in-context regeneration pass; LTX-2 is the only in-tree family with a separate audio VAE, an audio decoding stage, and a refine stage. | Stage chain and test layout follow `fastvideo/pipelines/basic/ltx2/`. |
| 2026-08-03 | Defer MSA to a follow-up PR. | Upstream ships full attention only in the initial open-source release. | First PR targets the existing attention backends. |
| 2026-08-03 | First PR covers one checkpoint with its full output, audio included. | `add-model` Phase 0 requires explicit agreement before dropping a base-model output head; keeping audio avoids that negotiation and matches the LTX-2 precedent. | Audio VAE and an AV decoding stage are in scope. Phase 10 needs an audio metric (mel-L1 or multi-resolution STFT) separate from video SSIM. |
| 2026-08-03 | Unblock I001 by enabling HF egress on the environment rather than hand-carrying configs. | Parity and conversion need real weights, not just `config.json`; a config-only path stalls again at Phase 5. | Port resumes at `add-model-01-prep` once the policy change lands. |

## Handoff Notes
- Resolve `I001` first. Everything else in this file is unverified public
  reporting and must be re-derived from the real checkpoint.
- Once weights are reachable, restart at `.agents/skills/add-model-01-prep/SKILL.md`
  and overwrite the Component Matrix from `inspect_hf_layout.py` output.
- Lock `Q006` (checkpoint scope) before dispatching any component subagent.
