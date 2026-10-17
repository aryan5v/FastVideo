---
license: other
tags: [video-generation, text-to-audio-video, reference-to-video, minimax-h3, fastvideo, nvfp4]
---

# FastH3-Omni-NVFP4-QAD-v2 (OmniRef, all-NVFP4, frozen gates and norms)

FastH3 OmniRef (reference-to-video with audio, 8-forward PDD) as a full NVFP4 stack for Blackwell GPUs
(B200/B300/GB200, RTX 5090/PRO 6000, DGX Spark). This is the second quantization-aware distillation (QAD v2) of the
OmniRef student. It replaces the DiT of `FastVideo/FastH3-Omni-NVFP4-QAD`; every other component is identical.

## What QAD v2 changes versus the first release

| | `FastH3-Omni-NVFP4-QAD` (first QAD) | **QAD v2 (this repo)** |
|---|---|---|
| NVFP4 blocks | 42 of 50 (blocks 42-49 kept bf16) | **all 50** (`skip_blocks` empty) |
| VSA compression gates (`attn.to_gate_compress`) | trained | **frozen** at the original OmniRef values (still NVFP4-packed) |
| Q/K norms (`norm_q`, `norm_k`) and all other norms | trained | **frozen**, byte-identical to the original OmniRef |
| Trained tensors | q/k/v/out, gate, norms, fc_in/fc_out | q/k/v/out, fc_in/fc_out only |
| Attention during training | FP4 sparse attention | FP4 sparse attention |
| FFN activation scales | pinned MSE-clipped amax table | same table (sha256 `14b93ec3...`, `transformer/amax.sha256`) |

Rationale: training the VSA gates and norms drifted the sparse-attention tile selection (ghosting seen in the V2
QAD runs), so v2 freezes them and quantizes every block.

## Checkpoint

- Selected checkpoint: **step 120** of 200 planned (best on the teacher-forced eval).
- Teacher-forced score (lower is better): **0.303**, versus 0.362 for the NVFP4 initialization at step 0.
  Components: video 0.198, audio 0.409.
- Eval set: the OmniRef held-out rows used for the first QAD (480p and 768p).
- **Visual eval pending.** Only the teacher-forced metric has been measured; no blind or sample-based comparison
  against the first release has been run yet.

## Contents

| Component | Format |
|---|---|
| `transformer` / `transformer_ref` | Identical weights (QAD trains one student): packed NVFP4 linears in `nvfp4_weights.safetensors` (350 linears: q, k, v, out, gate, fc_in, fc_out for each of 50 blocks) plus dense bf16 shards for norms, embeddings and modulation. No bf16 skip blocks are recorded in the safetensors metadata, so the loader quantizes every block. |
| `text_encoder` | NVFP4 Qwen3-VL text encoder (unchanged) |
| `vae_decoder_nvfp4/decoder-nvfp4.safetensors` | 36-layer NVFP4 video decoder (unchanged) |
| `vae`, `audio_vae`, schedulers, processor, tokenizer | Unchanged. `vae` is still needed for reference encoding and its config. |

Size: the DiT is about 37 GB (first release: 44.5 GB with 8 bf16 blocks). `transformer` and `transformer_ref` are
identical, so the folder is about 64 GB on disk with the duplicate stored once (about 101 GB if both are uploaded
without deduplication).

## Attention

- VSA sparse attention at sparsity 0.9, tile 128, reference keep 0.1 (`fastvideo_inference.json`).
- FP4 sparse attention was part of training; bf16 sparse attention remains the safe fallback.

## Running with FastVideo

Requires the FastVideo branch with OmniRef NVFP4 support.

```bash
FASTVIDEO_H3_VAE_NVFP4_DECODER=<repo>/vae_decoder_nvfp4/decoder-nvfp4.safetensors \
python examples/inference/basic/<omniref example> --model-path <repo>
```

Load the DiT with the packed NVFP4 layer profile (`h3_dit_vsa`).

## Verification done at export

- Packed weights, block scales and alpha of sampled linears (blocks 0, 25, 49, all seven linear types) are
  byte-identical to the QAD harness's quantization of the bf16 masters, and the deployed NVFP4 linear output equals the
  harness `nvfp4_gemm` bit for bit.
- Gate weights of all 50 blocks and every dense tensor (all norms) are byte-identical to the original OmniRef.

Internal release candidate. Private; not for redistribution.
