# MiniMax H3 port status

## Already had, then hybrid on top

The rows below are the **dense / VSA FastH3 port**. That work is complete.
Hybrid attention (window softmax + linear far branch) is an opt-in layer on
that stack, not a second H3 port. Details and pitfalls:
`fastvideo/models/dits/minimax_h3_hybrid/AGENTS.md`.

### Already had (reused by hybrid)

- Packed `[text | condition | audio | video]` layout
- DiT QKV, QK-norm, 3-axis MM-RoPE, `to_out`, AdaLN, SwiGLU, dual heads
- Dense attention and VSA-H3, Sol-Engine fusions, FP8 on wide linears
- Sequence parallel, FSDP load, MLX T2VA
- Video/audio VAEs, Qwen3-VL, schedulers (shift 12 / 3)
- T2VA / FL2VA / Ref2VA pipelines and official latent parity

### Added on top (hybrid, opt-in)

- `hybrid_attention: true` in `transformer/config.json` (default off)
- Chunk-aligned window softmax (`radius=1`, `chunk=5`, `anchor_frames=both`)
- Bidirectional `vdn_solve` linear branch, softmax gate, `to_out_linear`
- 1+1 branch-parallel when sequence parallel size is 2
- Converter overlay of `linear_branch/` + LoRA merge onto a dense `transformer/`
- MLX path auto-selected from hybrid weight keys
- Run converted checkpoints with `--no-vsa`

### Learned

- `attn.orig.to_out.0` must map before generic `attn.orig.*`; skip dropout `to_out.1`
- Extra hybrid modules are siblings of `to_q` (`attn.to_out_linear`), not `attn.hybrid.*`
- Full-cover windows skip the linear branch; the softmax gate still scales (~0.99)
- SP > 1 all-gathers into hybrid; the 1+1 split is only SP == 2
- Converter writes `transformer/` only; do not build the full DiT on CPU to unit-test hybrid

Hybrid E2E / SSIM against a converted VDN checkpoint is **not** recorded here yet.

## Status

- workloads: T2VA, FL2VA, Ref2VA joint video/audio generation
- component parity: complete
- FastVideo runtime acceptance: complete
- official end-to-end pipeline parity: complete

## Coverage

| Scope | Evidence | State |
|---|---|---|
| Qwen3-VL encoder | exact text/image/video hidden states through the production loader | complete |
| FL2VA and Ref2VA DiTs | exact video/audio heads for both model partitions | complete |
| Video VAE | exact encode, normalization, and decode through the production loader | complete |
| Video VAE streaming | exact chunked encode/decode and output-rank-only distributed decode | complete |
| Audio VAE | exact encode and normalization; decode maximum absolute drift `2.4e-7` | complete |
| Video/audio schedulers | pinned `12/3` schedule parity | complete |
| FL2VA packing | pinned row, position, tag, timestep, and RNG parity | complete |
| Ref2VA media and packing | pinned media and packing parity | complete |
| Public surface | manifest resolution, pipeline registration, and three presets | complete |
| FastVideo distributed runtime | valid joint AV outputs; SP=1/SP=4 latent consistency | complete |
| Official end-to-end pipeline | exact T2VA, FL2VA, and Ref2VA video/audio latents | complete |

## Current validation

T2VA, FL2VA, and Ref2VA match the official video/audio latents exactly.

## Decisions

- Preserve each H3 scheduler's configured shift; global `flow_shift` is invalid.
- Do not wrap the H3 DiT in global autocast; its FP32 projections must stay FP32.
- Let FSDP move CPU-offloaded Qwen parameters; do not move the wrapped conditioner as a whole.
- Load `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA.
- Keep `last_image`, `references`, and `audio_latents` on the typed request path.
- Treat the published component folders as the loading boundary.
- Keep reference videos on CPU between VAE clips and decode final pixels only on the executor's output rank.

## Evidence boundary

Completed rows summarize recorded component and FastVideo runtime runs. Registry smoke, generated media, and
FastVideo SP consistency are supporting checks, not substitutes for the recorded official comparisons.
