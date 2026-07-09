# Wan2.2-TI2V-5B Port Status

## Summary
- model_family: `wan2_2_ti2v_5b`
- workload_types: T2V and I2V (future-only; neither is public)
- official_ref: unpinned historical candidate only; no approved source/revision/checksum
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
| Transformer | DiT | internal MLX prototype | unpinned historical candidate | unknown | `fastvideo.mlx_runtime.wan22.MLXWan22DiT` | tiny pass | blocked | real scaffold skip | Q001, Q002, Q005 |
| VAE | video VAE | internal decode helper | unpinned historical candidate | unknown | `fastvideo.mlx_runtime.wan_vae` | helper only | blocked | scaffold skip | Q001, Q003, Q005 |
| Text conditioner | encoder | undecided | unknown | unknown | undecided | blocked | blocked | scaffold skip | Q001 |
| Image conditioner/mask | encoder/conditioner | latent-only helper | unknown | unknown | `fastvideo.mlx_runtime.wan22_i2v` | latent replacement only | blocked | scaffold skip | Q001, Q004, Q005 |
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
| Transformer tiny | `pytest fastvideo/tests/mlx/test_mlx_wan22_parity.py -v -s` | local MLX gate | Per-token timesteps only; not real-weight evidence. |
| Transformer real | `pytest fastvideo/tests/mlx/test_mlx_wan22_real_weights.py -v -s` | precise skip expected | Requires staged weights plus `FASTVIDEO_WAN22_5B_REVISION` and `FASTVIDEO_WAN22_5B_SHA256`. |
| I2V latent-only | `pytest fastvideo/tests/mlx/test_mlx_wan22_i2v.py -v -s` | local MLX gate | Frame replacement/timestep only; no VAE, image encoder, mask, or public I2V claim. |
| VAE | Recorded after official selection | not started | 48-channel latent normalization and decode. |
| Pipeline | Recorded after components pass | not started | Compare denoised latents and decoded media. |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which official source, immutable revision, model-card license, and SHA256 manifest match the intended 5B release? | product/model owner | 0 | open | None; historical `FastWan2.2-TI2V-5B-FullAttn` names are not sufficient. Do not download or activate real tests before selection. |
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
| I005 | 0 | artifact contract | blocker | Real-weight parity, QAD arming, VAE decode, and benchmarks lack an approved source revision/checksum manifest. | Q001; tests intentionally skip without explicit environment pins | product/model owner | open | Record source URL/revision, transformer/VAE/decoder SHA256 values, license, and staging path. |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | 0 | model source and cost | Choose official source/weights and authorize any gated download before implementation. | Select the weights whose architecture matches the desired 5B T2V/I2V release, then run preparation. | open | None. |

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-09 | Keep this branch preparation-only. | Initial release scope is 1.3B T2V; no 5B/I2V public commitment is authorized. | No model code, weights, training, or API on this branch. |
| 2026-07-09 | Do not merge PR #1563 yet. | It is blocked and needs a fresh focused 5B config test. | Candidate remains external. |
| 2026-07-09 | Preserve fork PRs #4/#6/#7/#8/#9/#10/#11 as internal future code and tests, not release surfaces. | Avoid losing reviewed fork work while preventing unsupported launch claims. | No 5B registry, CLI, public I2V, automatic downloads, or model publication. |
| 2026-07-09 | Require strict state-dict loading in every parity/arming path. | Missing keys invalidate a parity claim. | Tests fail rather than printing missing keys or using `strict=False`. |

## Handoff Notes
- First authorized next step: resolve E001/I005 with one immutable official source and checksum manifest, then run the add-model preparation workflow before activating real component parity tests.
- Required completion gates: real official imports in the FastVideo environment, real weights, native component prototypes, strict loading/conversion, non-skip component parity, pipeline parity, and approved 32 GB+ Mac quality/benchmark evidence.
