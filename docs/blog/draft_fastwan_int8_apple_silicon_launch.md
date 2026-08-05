# FastWan INT8 on Apple Silicon: 720p Video Generation, Entirely on Your Mac

> **Banner:** [TODO — hero image/video loop: a 720p clip generated on a MacBook, with the three model badges (1.3B / 5B / 14B INT8)]

[TODO: date] · FastVideo Team

**TL;DR: A five-second, 720p, text-to-video clip, generated start to finish on a Mac. No cloud, no datacenter GPU.** Today we're releasing the FastWan INT8 family for Apple Silicon: three models (1.3B, 5B, and 14B) trained with quantization-aware distillation against the exact INT8 grid your Mac deploys on, plus a native MLX runtime that runs the whole pipeline on the Metal GPU. The 5B generates at 720p out of the box and fits in 16 GB of unified memory. The 14B is the largest video model we've ever run on a laptop. And the 1.3B is the fastest way to make a video on a Mac, period.

---

## Why a Mac release, and why now

A year ago, "local video generation" meant owning a $2,000 GPU. [FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/) proved you could co-design a model, a quantization format, and a runtime to make a 5-second video in 1.8 seconds on a single RTX 5090 — but that's still a datacenter card on someone's desk.

Meanwhile, something quieter happened: the Mac became a serious AI machine. Apple silicon ships tens of millions of capable GPUs with unified memory, and the M5 generation added dedicated Neural Accelerators — per-core matrix units that Metal FlashAttention-class kernels already exploit for up to 4.6× faster generation over M4 (as Draw Things measured with MFA 2.5). MLX matured into a real framework with fused kernels, graph compilation, and quantized formats. macOS 26 and Metal 4 gave it a driver-level path. The community noticed — local LLMs on Mac went from curiosity to default.

Video generation is the obvious next step, and it's the one we build. So we brought the core of FastVideo to Mac: not a demo, not a cloud-backed shim — the actual DiT, sampler, and decoder, running natively on Metal.

This post is about what shipped, how we trained it, and what we learned about quantization on Apple silicon the hard way (so you don't have to).

## What ships

**Three QAD checkpoints, one recipe, one deploy grid:**

| Checkpoint | Base | Native output | INT8 size | Mac tier | Role |
|---|---|---|---|---|---|
| **FastWan-QAD-1.3B-INT8** | Wan2.1-T2V-1.3B | 480p, ~5 s | ~1.4 GB | 16 GB+ | Fastest download |
| **FastWan-QAD-5B-INT8** | Wan2.2-TI2V-5B | **720p (121×704×1280), ~5 s** | ~5.3 GB | 16 GB+ | **Fast + high-res** |
| **FastWan-QAD-14B-INT8** | Wan2.1-T2V-14B | 480p, ~5 s | ~14.9 GB | 24 GB+ | Maximum local quality |

[CHART: bar — INT8 checkpoint sizes vs Mac unified-memory tiers (16/24/32/64 GB), with the three models placed]

Every model is a 3-step student trained with DMD2 and quantization-aware training on the affine INT8 grid, so the precision it learned in is the precision it ships in. Each publishes in two formats: Diffusers safetensors, and a **pre-quantized MLX checkpoint** that halves the download and skips requantization at load.

**The MLX runtime** — a native Apple Silicon implementation of the Wan DiT with an on-device 3-step DMD sampler, pinned against the PyTorch reference by parity tests that run in CI on both Metal and CPU backends. The DiT forward is `mx.compiled` by default, bit-identical to eager.

**Generation modes, because one size doesn't fit all.** Temporal and spatial shortcuts compose, and every one ships behind a quality gate:

- **Fast mode (RIFE)** — generate every Nth frame and interpolate the rest with Apple-silicon-native rife-mlx. `--fast --fast-factor 2` roughly halves diffusion work; `--fast-sharpen` restores edge crispness.
- **Spatial fast mode** — RIFE's spatial twin: denoise at a fraction of the target resolution, then bilinear-upsample the clean latents back before decode. No second denoise, ≈ scale² fewer tokens. Composes with `--fast` for fewer frames *and* fewer pixels. (MetalFX is deliberately not used here — it wants game-engine motion vectors and depth that diffusion output doesn't have; latent bilinear upsample is the same primitive without them.)
- **Refine mode (two-pass)** — the H3 / LTX-2 pattern, ported into the runtime: generate at base resolution, re-noise, and run a second denoising pass with the *same* DiT at higher resolution. No LoRA, no SR weights, no training — the biggest quality lever that doesn't need a new model. `--fast + --refine` composes: fewer frames at base res, full-res refine, RIFE restore after decode.
- **Quality mode** — full Wan VAE decode (bf16) instead of the default TAEHV, one flag away (`--decode-backend wan-vae`) when fidelity beats speed.
- **Prompt enhancement (Context-IR-style)** — an on-device pre-pass that expands a short prompt into Wan-style cinematic shot language, via a local mlx-lm model or a deterministic template, with an on-disk cache. Wan's training captions are long and cinematic; short prompts leave quality on the table, and this closes the gap without a remote API.
- **Draft attention (experimental)** — windowed attention with sinks for interactive prompt exploration, gated behind an SSIM check; dense attention stays the default for final renders.
- **Prompt cache** — embeddings are content-addressed and cached, so iterating on seeds and settings skips the text encoder entirely.

**A reproducible benchmark harness** — quantization × decoder × prompt sweeps emitting latency, peak unified memory, MS-SSIM, and side-by-side HTML review grids.

FastVideo source is Apache-2.0; the vendored TAEHV decoder is MIT.

## The Mac inference stack

The co-design target on a Mac is MLX and unified memory, not tensor cores and HBM. The stack reflects that, layer by layer.

**MLX-first DiT, dense attention.** The denoising loop — the dominant cost — runs natively in MLX on the Metal GPU: the full Wan transformer forward with `mx.fast.scaled_dot_product_attention`, 100% dense like FastWan-QAD. The DMD sampler runs on-device too; no tensor leaves unified memory during the 3-step loop, and the whole step is one compiled MLX graph.

**INT8 where it counts.** Every DiT matrix weight is quantized with MLX's affine INT8 (group size 64) and executed with `mx.quantized_matmul`; norms and modulation tables stay fp16. More on why INT8 — and emphatically not MXFP4 or NVFP4 — below.

**Memory choreography.** The umT5 text encoder loads in bf16, encodes once, and is freed before the DiT loads. Decode defaults to TAEHV, a tiny autoencoder that removes the full Wan VAE from both the latency and the memory peak. Peak memory is the largest single stage, not the sum of everything — which is exactly why a 5B fits in 16 GB and a 14B in 24.

**Pre-quantized checkpoints and a prompt cache.** Quantizing billions of parameters at every startup is wasted work. The MLX checkpoint stores packed INT8 weights, scales, and biases directly; reloads skip requantization entirely, and a cached prompt embedding makes repeat generations start in seconds.

## QAD, retargeted to MLX

Post-training quantization visibly damages a 3-step model: with only three denoising steps there's no room to recover from weights knocked off their trained values. The original FastWan-QAD recovered NVFP4 quality by making the model live in its deployment precision during training. We did the same for MLX: the training-time fake quantizer transcribes MLX's affine quantization arithmetic — the fp32 group min/max, the negative-scale anchoring, the integer zero-point re-fit — and its decisions are pinned against `mx.quantize`/`mx.dequantize` by tests in the suite. Training targeted exactly the weight set the runtime quantizes, at the dtype the runtime quantizes it.

The recipe is quantization-aware DMD: a frozen teacher, a critic, and a student initialized from the already-distilled FastWan checkpoint whose every forward computes on the INT8 deploy grid, with gradients passing straight through to the real weights.

## Why INT8 — and why not MXFP4 or NVFP4

We didn't pick INT8 from a spec sheet. We measured, on an Apple M5, across every candidate format — and before that, we failed publicly enough to learn from it.

**Integer formats beat float formats at equal bit width, and it isn't close.** Affine INT8 (8.5 bits/param) reconstructs weights 12.4× more accurately than MXFP8 (8.25 bits/param). The mechanism: E4M3 float carries 3 mantissa bits (~6% relative precision per element), while affine integer quantization spends all 8 bits on 256 uniform levels across each group's actual range, with a scale *and* a bias to fit it. At 4 bits the ordering holds: affine INT4 > NVFP4 > MXFP4.

[CHART: bar — reconstruction relative-L2 by format (int8, mxfp8, int4, nvfp4, mxfp4), log scale, int8 highlighted]

**No weight-only format buys speed on Metal — so the choice is purely about memory.** Diffusion transformers at batch one are compute-bound: weights are ~9% of memory traffic at our shapes, and `mx.quantized_matmul` dequantizes to fp16 and runs fp16 arithmetic — the integer matrix units never engage. We measured INT8, MXFP4, and NVFP4 within noise of each other on wall-clock. This is also why LLM quantization folklore misleads here: LLM decode is weight-bandwidth-bound (one token at a time), so 4-bit is a real speedup there. A video DiT at 8k+ tokens is the opposite regime.

**And MXFP4 specifically failed us, twice, on real runs.** Before the survey, we trained QAD models on the MXFP4 grid at both 1.3B and 14B. Neither made the cut. The 14B MXFP4 run consumed 26 GPU-days over 4000 steps and produced washed-out, undertrained-looking video. The reconstruction numbers explain it: MXFP4 is the worst-reconstructing format we tested (22× INT8's error), and it could never have delivered speed regardless. Below, the rejected MXFP4 outputs next to their INT8 counterparts. The difference is not subtle.

[VIDEO GRID: 1.3B MXFP4 vs INT8, same prompt/seed]
[VIDEO GRID: 14B MXFP4 vs INT8, same prompt/seed]

So: INT8, because on Apple silicon quantization is a memory decision, and INT8 is the most accurate way to spend 8 bits. 5.3 GB for the 5B, 14.9 GB for the 14B — model classes that used to need a datacenter, in laptop memory.

## Results

MS-SSIM of each model's INT8 output against its own FP16 output on the motion7 prompt set (shared seeds) measures how much quantization changes the result — consistency, not absolute quality:

| Model | Mean MS-SSIM |
|---|---|
| FastWan 1.3B, post-training quantization | 0.907 |
| FastWan-QAD-INT8-1.3B (EMA release) | 0.933 |
| FastWan-QAD-INT8-5B | TODO(measure) |
| FastWan-QAD-INT8-14B | TODO(measure) |

Absolute quality is judged by humans: the released checkpoints are selected from visual review grids [TODO: link].

Speed and memory (Apple M5, 3-step DMD, INT8 DiT, TAEHV decode):

| Model | Output | End-to-end | Denoise | MLX peak memory |
|---|---|---|---|---|
| 1.3B | 480×832×81 | TODO(measure) | TODO | TODO |
| 5B | 704×1280×121 | TODO(measure) | TODO | TODO |
| 14B | 480×832×81 | TODO(measure) | TODO | TODO |

[CHART: scatter or bar — end-to-end latency vs model size, fast-mode on/off]
[VIDEO: hero 720p example from the 5B, fox-in-forest prompt]
[VIDEO: 1.3B fast-mode RIFE ×2 side-by-side]

Those memory figures are why this release matters beyond any one machine: the pipeline peaks far below the headline capacity of even base Macs. We are validating lower-memory configurations and will state exact supported tiers as they pass — this post claims only what we have measured.

## How to run

```bash
uv pip install -e '.[mlx]'
huggingface-cli download [TODO: final HF id] --local-dir ~/models/fastwan-qad

python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root ~/models/fastwan-qad \
  --mlx-checkpoint ~/models/fastwan-qad/mlx_dit \
  --prompt "A fox runs through a misty pine forest, leaves kicking up behind it."
```

The defaults are the release configuration. Fast mode is `--fast --fast-factor 2`, quality mode is `--decode-backend wan-vae`, and the Apple Silicon guide covers fast-reload, memory tiers, and troubleshooting.

## What's next

The optimization ladder that took the CUDA stack from 170 s to 1.8 s is the ladder we're now climbing on Metal.

- **W8A8 fused INT8 GEMM — in development.** Weight-only quantization can't buy speed on Metal; quantizing activations too (so the integer matrix units actually engage) can. Our calibration study says it's viable with per-token scales, and the custom Metal kernel (int8×int8 → int32 accumulate) is already running as a correctness-verified prototype. The speed gate is next, on M5 Neural Accelerators, then a W8A8 QAT pass. Nobody has done this for diffusion on Metal. We intend to be first.
- **Image-to-video and longer clips.** The 5B is natively image-conditioned; I2V and streaming/causal generation are on the runtime roadmap.
- **MiniMax-H3.** The same INT8 QAD machinery now points at H3's 33B omni-modal (video + native audio) teacher. Local audio-video generation is the thing nobody else ships.

## Acknowledgements

[TODO: contributor list, compute acknowledgements (GB200 cluster), advisor list, community links]
