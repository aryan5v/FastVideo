# MiniMax-H3 Local Tests

Local-only parity, registry, and pipeline-smoke tests for the `minimax_h3`
FastVideo port. Skipped in CI; run locally on GPU.

**This directory is currently prep-stage scaffolding.** No executable parity
tests exist yet because the reference assets have not been staged — see
[`PORT_STATUS.md`](./PORT_STATUS.md), issue `I001`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `minimax_h3` |
| Workload types | `T2V`, `I2V`, `V2V`, `T2I` + native audio (see `Q004` on the AV workload shim) |
| Official reference | `<TODO>` — no `MiniMax-AI/MiniMax-H3` repo exists on GitHub as of 2026-08-03; reference code is expected to ship inside the HF repo |
| Local reference dir | `<TODO>` |
| Official commit/version | `<TODO>` |
| HF weights | `MiniMaxAI/MiniMax-H3` (two task-specific checkpoints) |
| HF revision | `<TODO>` |
| Local weights dir | `official_weights/minimax_h3/` |
| Source layout | `diffusers` (expected — see below) |
| Needs conversion | `unknown` — resolve with `inspect_hf_layout.py` before Phase 3 |

> Use only the env-var **name** for tokens (e.g. `HF_TOKEN`). Never paste a token value.

### Expected component layout

Public reporting describes each H3 checkpoint as a self-contained
Diffusers-style repo:

```text
model_index.json
processor/
tokenizer/
text_encoder/
transformer/      # H3-Omni Transformer
visual_vae/       # H3-VAE
audio_vae/        # standalone audio VAE
```

Every value in this section is unverified until `inspect_hf_layout.py` has been
run against the real repo. Do not begin Phase 3 component dispatch on the
strength of this table.

## Shared Environment Setup

Run from the FastVideo repo root, in the same env used for FastVideo.

```bash
python ".agents/skills/add-model-01-prep/scripts/inspect_hf_layout.py" \
    "MiniMaxAI/MiniMax-H3" --json

python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "MiniMaxAI/MiniMax-H3" "official_weights/minimax_h3"
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: none
official_env_status: blocked
private_dep_stubs: unknown
blocked_on: huggingface.co unreachable from the porting environment (I001)
```

## Planned Tests

None are written yet. The table below is the Phase 2 scaffolding target; each
row becomes a non-skip PASS requirement before Phase 7.

| Component | Planned test | Concerns | Status |
|---|---|---|---|
| `transformer/DiT` | `test_minimax_h3_transformer.py` | H3-Omni block parity, text/image/video/audio token packing, RoPE, per-modality embeddings | `not_started` |
| `transformer/DiT` (audio head) | `test_minimax_h3_transformer_audio.py` | joint AV denoising and any dedicated audio output head | `not_started` |
| `vae` (visual) | `test_minimax_h3_visual_vae.py` | H3-VAE encode/decode parity, compression ratio, latent normalization | `not_started` |
| `vae` (audio) | `test_minimax_h3_audio_vae.py` | audio encode/decode parity, vocoder if separate | `not_started` |
| `text encoder` | `test_minimax_h3_text_encoder.py` | encoder identity + Contextual Omni Representation compression path | `not_started` |
| `processor` | `test_minimax_h3_processor.py` | multimodal reference preprocessing (image/video/audio conditioning) | `not_started` |
| `scheduler` | `test_minimax_h3_scheduler.py` | timestep/sigma schedule and guidance math | `not_started` |
| `pipeline` (smoke) | `test_minimax_h3_pipeline_smoke.py` | `VideoGenerator` end-to-end smoke | `not_started` |
| `pipeline` (in-context regen) | `test_minimax_h3_regeneration.py` | two-pass low-res → in-context regenerate parity | `not_started` |
| `registry` | `test_minimax_h3_registry.py` | sampling/pipeline registry resolution | `not_started` |

## Review Notes

- LTX-2 is the closest existing structural precedent in-tree: joint video+audio
  latents, a separate audio VAE, an audio decoding stage, and a refine pass.
  See `fastvideo/pipelines/basic/ltx2/` and `tests/local_tests/ltx2/`.
- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Audio output cannot be regression-tested with SSIM. Per
  `.agents/skills/add-model/SKILL.md` Phase 10, use a mel-spectrogram L1 or
  multi-resolution STFT metric and keep it separate from the video SSIM test.
