# Wan2.2-TI2V-5B Port Status

## Summary
- model_family: `wan2_2_ti2v_5b`
- workload_types: T2V and I2V (future-only; neither is public)
- official_ref: unknown
- official_ref_dir: none
- hf_weights_path: unknown
- local_weights_dir: none
- source_layout: unknown
- local_tests_readme: `tests/local_tests/wan2_2_ti2v_5b/README.md`

## Current Phase
- phase: 0 — preparation
- status: blocked
- owner: prep
- last_updated: 2026-07-09

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| Transformer | DiT | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001, Q002 |
| VAE | video VAE | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001, Q003 |
| Text conditioner | encoder | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001 |
| Image conditioner/mask | encoder/conditioner | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001, Q004 |
| Scheduler | scheduler | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001 |

## Conversion State
- conversion_script: none
- converted_weights_dir: none
- source_layout: unknown
- strict_load_status: not started
- passthrough_components: unknown
- retry_history: none

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Scaffold | `pytest tests/local_tests/wan2_2_ti2v_5b/test_wan2_2_ti2v_5b_parity_scaffold.py -v -s` | skip expected | Requires Q001/Q005; not parity evidence. |
| Transformer | Recorded after official selection | not started | Per-token timesteps and T2V/I2V conditions. |
| VAE | Recorded after official selection | not started | 48-channel latent normalization and decode. |
| Pipeline | Recorded after components pass | not started | Compare denoised latents and decoded media. |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which official source, exact revision, and weights match the intended 5B release? | product/model owner | 0 | open | None; do not download or port before selection. |
| Q002 | What is the official per-token timestep input shape, dtype, and transformer call path? | reference-study owner | 1 | open | None. |
| Q003 | What are the 48-channel VAE latent mean/std and normalization/decode semantics? | reference-study owner | 1 | open | None. |
| Q004 | What image conditioner, image-latent layout, and I2V mask semantics does the official pipeline use? | reference-study owner | 1 | open | None. |
| Q005 | Is the official checkpoint Diffusers-compatible or is a conversion script required? | conversion owner | 0 | open | None. |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | 0 | all | blocker | Official reference and matching weights are not selected. | Q001 | product/model owner | open | None. |
| I002 | 1 | transformer/VAE | blocker | No native classes or converted weights exist. | Component matrix | port owner | open | Expected until Q001–Q005 close. |
| I003 | future dependency | self-forcing | blocker | PRs #1307/#1042/#814 lack this port's required causal/KV-cache/training/media gates. | Future dependency policy | self-forcing owner | open | Keep separate. |
| I004 | future candidate | FullAttn config | medium | PR #1563 is OPEN and BLOCKED; it changes Wan config/registry/runtime surfaces. | `gh pr view 1563` on 2026-07-09 | 5B owner | open | Rebase and test independently before selection. |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | 0 | model source and cost | Choose official source/weights and authorize any gated download before implementation. | Select the weights whose architecture matches the desired 5B T2V/I2V release, then run preparation. | open | None. |

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-09 | Keep this branch preparation-only. | Initial release scope is 1.3B T2V; no 5B/I2V public commitment is authorized. | No model code, weights, training, or API on this branch. |
| 2026-07-09 | Do not merge PR #1563 yet. | It is blocked and needs a fresh focused 5B config test. | Candidate remains external. |

## Handoff Notes
- First authorized next step: resolve E001, then run the add-model preparation workflow before creating real component parity tests.
- Required completion gates: real official imports in the FastVideo environment, real weights, native component prototypes, strict loading/conversion, non-skip component parity, pipeline parity, and approved 32 GB+ Mac quality/benchmark evidence.
