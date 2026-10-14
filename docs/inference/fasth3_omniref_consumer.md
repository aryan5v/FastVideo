# FastH3 OmniRef on consumer devices

FastH3 OmniRef (Ref2VA) is a reference-conditioned PDD student: a prompt plus
ordered image/video/audio references become one video+audio clip in eight
fused transformer forwards. It has an extra conditioning prefix - the text, the
Qwen3-VL vision tokens of every reference, and any reference audio - on top of
the target video and audio rows.

This page covers running it on one consumer Blackwell device (RTX 5090, RTX PRO
6000, DGX Spark) at the full NVFP4 stack, and the closest equivalent on an RTX
4090, which has no FP4 hardware.

The checkpoint is a FastH3 OmniRef PDD export with a **packed NVFP4
`transformer_ref`**: NVFP4 linears for every block, with the blocks the
calibration sweep kept in bf16 (42-49) absent from the packed file and loaded
from the dense shards, plus static FFN activation scales. The export records its
own bf16 block list in the safetensors metadata, so nothing has to be declared
at load time.

## Composing the model directory

An OmniRef export carries only what distillation changed: `transformer_ref/`,
the two scheduler configs and `fastvideo_inference.json`. Everything else is
base MiniMax-H3. `examples/inference/basic/basic_fasth3_omniref_pdd.py` composes
them into one directory of symlinks and runs the stack:

```bash
python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path ./FastH3-OmniRef-QAD120-NVFP4 \
  --image reference.png --audio voice.wav \
  --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \
  --profile spark
```

`--profile` is what makes the run device-appropriate. Without it the script
keeps its data-center defaults. The composed directory is what the example
recipes under `examples/inference/basic/basic_fasth3_omniref_*.yaml` point
`model_path` at.

## What each profile does

Every Blackwell profile is the same numerics, so a clip made on one device is
the same clip as on another. Only the memory policy differs.

| Profile | Device | Geometry | Offload |
|---|---|---|---|
| `spark` | DGX Spark, GB10, 128 GB unified | 480p | encoder + VAE off the GPU; unified memory cannot hold both |
| `rtx-pro-6000` | RTX PRO 6000, sm_120, 96 GB | 768p | resident encoder, DiT and decoder |
| `rtx-5090` | RTX 5090, sm_120, 32 GB | 480p | encoder off the GPU, DiT blocks streamed one at a time |
| `rtx-4090` | RTX 4090, sm_89, 24 GB | 480p | as the 5090, with a hard allocator cap |

The NVFP4 stack the Blackwell profiles run is:

- **Linears:** NVFP4, from the packed export, with its bf16 blocks.
- **Attention:** VSA sparsity 0.9, 128-token tiles, reference keep 0.1, on the
  FP4 sparse kernel, with the dense prefix rows split off and run at 8-bit Q/K
  and bf16 P·V. The prefix split is what keeps the text, reference and audio
  rows - the conditioning the clip is built from - out of the FP4 rounding. It
  is single-rank for now: at `sp_size` above 1 every row still goes through the
  FP4 kernel.
- **Text encoder:** the NVFP4 Qwen3-VL encoder, unchanged from the 8-step V2
  NVFP4 consumer release, including the vision tokens every reference adds.
- **Decoder:** the pre-packed NVFP4 decoder (`decoder-nvfp4.safetensors`), which
  is read straight onto the GPU without materializing the full-precision
  weights of its quantized linears.

The RTX 4090 profile substitutes FP8 linears for NVFP4 and bf16 VSA attention
for the FP4 kernel; the NVFP4 decoder still runs, because its GEMMs fall back to
CUTLASS on sm_89.

## Environment

The profiles set these; set them yourself if you drive the YAML recipes.

| Variable | Value | Why |
|---|---|---|
| `FASTVIDEO_ATTENTION_BACKEND` | `VIDEO_SPARSE_ATTN_H3` | the trained backend; the contract fixes the tile size |
| `FASTVIDEO_H3_VSA_FP4` | `1` | FP4 sparse attention (sm_120/sm_121) |
| `FASTVIDEO_H3_VSA_FP4_PREFIX` | `fp8` | the prefix split; `bf16` is the higher-precision variant |
| `FASTVIDEO_H3_VAE_NVFP4_DECODER` | `/path/decoder-nvfp4.safetensors` | the pre-packed NVFP4 decoder |
| `FASTVIDEO_MINIMAX_H3_FUSIONS` | `all` | bit-exact fused kernels |
| `FASTVIDEO_VSA_TRITON` | `1` | Triton VSA on the consumer parts |
| `FASTVIDEO_VSA_SM100A`, `FASTVIDEO_FA4` | `0` | no sm_100a kernel and no FlashAttention 4 here |
| `FASTVIDEO_NVFP4_MM_BACKEND` | `cudnn` on Spark | 2.7-3x faster than CUTLASS on the H3 FFN shapes on GB10, bit-identical |
| `FASTVIDEO_H3_VAE_TILE_BATCH` | `12` Spark, `4` 5090 | decoder tiles per call |
| `FASTVIDEO_H3_VAE_INT8_OVERLAY` | `0` on Spark | the INT8 ConvRot overlay is slower and less accurate on GB10 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` on the 4090 | keeps a 24 GB card from fragmenting |

`fastvideo-kernel` must be built for the device: `12.0a` on sm_120, `12.1a` on
GB10 (which needs CUDA 13). Building on the machine resolves the right one.

## Memory

Rough per-device working sets at 480p with one image reference, after the
packed decoder is in place. The encoder is the dominant term when it is
resident; streaming it is what makes 32 GB and unified memory work.

| Device | Resident | Streamed | Fit |
|---|---|---|---|
| RTX 5090, 32 GB | DiT block, decoder | Qwen3-VL encoder, DiT blocks | expected to fit |
| RTX PRO 6000, 96 GB | encoder, DiT, decoder | - | expected to fit at 768p |
| DGX Spark, 128 GB unified | DiT, decoder | encoder, VAE | expected to fit |
| RTX 4090, 24 GB | FP8 DiT block | encoder, DiT blocks | expected to fit |

These are estimates until the devices are in hand; see the device-test checklist
in the branch notes.

## Apple Silicon

`fastvideo/mlx_runtime` has an MLX MiniMax-H3 runtime, but its pipeline is
T2V/FL2VA-class: Ref2VA is only referenced in comments. Running OmniRef on a Mac
needs an MLX Ref2VA path (reference conditioning into the packed layout, the VSA
per-reference regions, the eight-forward PDD ladder) and an MLX build of the
NVFP4 decoder. Scoped, not implemented.
