# FastWan-QAD-INT8-1.3B: Local Text-to-Video on Apple Silicon

> Draft — publish only after the final motion7 visual review confirms the EMA
> checkpoint, the Hugging Face artifacts and checksums are live, and the
> headline timing is re-measured with the current release defaults.

*[TODO: date] · FastVideo Team*

**TL;DR:** A five-second, 480p-class, text-to-video clip generated entirely on
a Mac — no cloud, no discrete GPU. FastVideo now runs its 3-step distilled
FastWan model natively on Apple Silicon through an MLX runtime, and we are
releasing **FastWan-QAD-INT8-1.3B** `[TODO: HF link]`: a model adapted with
quantization-aware distillation against MLX's own INT8 quantizer, so the
weights you download are already at home on the precision your Mac will
actually run. The recorded release measurement on an Apple M4 Max is **123.7
seconds end to end** (117.6 s of denoising) at **5.63 GiB** of MLX peak
memory `[TODO: refresh with mx.compile-on defaults — the denoise loop
measured ~1.4x faster in our compile A/B]`.

## Why a Mac release

[FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/) pushed a single RTX
5090 to a 5-second video in 1.8 seconds by co-designing the model, the
quantization format, and the runtime for Blackwell tensor cores. This release
asks the follow-up question: what does that co-design look like on the most
widespread capable consumer hardware in the world — the Apple Silicon Mac?

A Mac will not match a 5090 on raw speed, and two minutes is not 1.8 seconds.
But video generation that requires a $2,000 GPU is not local video generation
for most people. Tens of millions of Macs ship with a capable GPU and unified
memory already on someone's desk. Making those machines first-class is how
local video generation actually becomes local.

## What ships

- **FastWan-QAD-INT8-1.3B** `[TODO: HF link]` — Wan2.1-T2V-1.3B, already
  3-step DMD-distilled, further adapted on the MLX affine INT8 grid (EMA
  weights, selected by visual review). Published in two formats: Diffusers
  safetensors, and a **pre-quantized MLX checkpoint** (`mlx_dit/`) that
  roughly halves the download and skips requantization at load.
- **The MLX runtime** — a native Apple Silicon implementation of the Wan DiT
  with an on-device 3-step DMD sampler, pinned against the PyTorch reference
  by parity tests that run in CI on both Metal and CPU backends. The DiT
  forward is `mx.compile`d by default, bit-identical to eager execution.
- **A reproducible benchmark harness** — quantization × decoder × prompt
  sweeps emitting latency, peak unified memory, MS-SSIM, and side-by-side
  HTML review grids.

FastVideo source is Apache-2.0; the vendored TAEHV decoder is MIT.

## The Mac inference stack

The layers are different on a Mac, and the co-design target is MLX rather
than tensor cores.

**MLX-first DiT, dense attention.** The denoising loop — the dominant cost —
runs natively in MLX on the Metal GPU: the full Wan transformer forward with
`mx.fast.scaled_dot_product_attention`, 100% dense like FastWan-QAD. The DMD
sampler runs on-device too; no tensor leaves unified memory during the 3-step
loop, and the whole step is one compiled MLX graph.

**INT8 where it counts.** Every DiT matrix weight is quantized with MLX's
affine INT8 (group size 64) and executed with `mx.quantized_matmul`; norms
and modulation tables stay fp16. INT8 is the reliability sweet spot on
today's MLX across Apple generations.

**Memory choreography.** The UMT5 text encoder is loaded in bf16 (fp32
exponent range, fp16 memory), used once, and freed before the DiT ever
loads. Decode defaults to TAEHV, a tiny autoencoder that removes the full
Wan VAE from both the latency and the memory peak — with the full VAE one
flag away (`--decode-backend wan-vae`, bf16) when fidelity matters more
than speed.

**Pre-quantized checkpoints.** Quantizing 1.3B parameters at every startup is
wasted work. The MLX checkpoint format stores the packed INT8 weights,
scales, and biases directly: reloads skip requantization entirely, and a
cached prompt embedding makes repeat generations start in seconds.

## Quantization-aware distillation, retargeted to MLX

Post-training quantization visibly damages a 3-step model: with only three
denoising steps there is no room to recover from weights knocked off their
trained values. The original FastWan-QAD recovered NVFP4 quality by making
the model live in its deployment precision during training. We did the same
for MLX: the training-time fake quantizer transcribes MLX's affine
quantization arithmetic — the fp32 group min/max, the negative-scale
anchoring, the integer zero-point re-fit — and its decisions are pinned
against `mx.quantize`/`mx.dequantize` by tests in the suite. Training
targeted exactly the weight set the runtime quantizes, at the dtype the
runtime quantizes it.

The recipe is quantization-aware DMD: a frozen Wan2.1-1.3B teacher, a critic,
and a student initialized from the already-distilled FastWan checkpoint whose
every forward computes on the INT8 deploy grid, with gradients passing
straight through to the real weights. Training ran on 4×B200 in under a day.

## Results

MS-SSIM of each model's INT8 output against its own FP16 output on the
motion7 prompt set (shared seeds) measures how much quantization changes the
result — consistency, not absolute quality:

| Model | mean MS-SSIM (INT8 vs own FP16) |
| --- | ---: |
| stock FastWan2.1-1.3B (post-training quantization) | 0.907 |
| **FastWan-QAD-INT8 (EMA, released)** | **0.933** |

Absolute quality is judged by humans: the released EMA checkpoint was
selected from the final motion7 visual review grids `[TODO: link or embed
the reviewed side-by-side grid]`. QAD costs nothing at inference — step time
and peak memory are identical to stock.

**Speed and memory** (Apple M4 Max, 36 GB-class, 480×832×81, 3-step DMD,
INT8 DiT, TAEHV decode): **123.7 s total, 117.6 s denoise, 5.63 GiB MLX
peak** `[TODO: refresh with compile-on defaults and add the wan-vae
quality-mode row]`.

That memory figure is why this release matters beyond the M4 Max: the
pipeline peaks far below the headline capacity of even base Macs. We are
validating lower-memory configurations and will state exact supported tiers
as they pass — this post claims only what we have measured.

## How to run

```console
uv pip install -e '.[mlx]'
huggingface-cli download [TODO: final HF id] --local-dir ~/models/fastwan-qad
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root ~/models/fastwan-qad \
  --mlx-checkpoint ~/models/fastwan-qad/mlx_dit \
  --prompt "A fox runs through a misty pine forest, leaves kicking up behind it."
```

The defaults are the release configuration; see the [Apple Silicon
guide](../getting_started/installation/mps.md) for fast-reload, quality-mode,
and troubleshooting details.

## What's next

The optimization ladder that took the CUDA stack from 170 s to 1.8 s is the
ladder we are now climbing on Metal: fused kernels behind the same quality
gates, INT4 behind the same QAD recipe, and — under separate quality gates
before anything is promised — larger models and image-to-video.

## Acknowledgements

`[TODO: contributor list, compute acknowledgements (DGX B200), advisor
list, community links]`
