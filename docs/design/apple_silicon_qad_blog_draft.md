# FastWan-QAD-INT8: Bringing FastVideo's Local Video Generation to Apple Silicon Macs

> **DRAFT** for haoailab.com / project blog. Placeholders marked `[TODO]`.
> Style and structure follow the FastWan-QAD launch post; this post is its
> Apple-native sequel and links back to it throughout.

*[date] · 5 min · FastVideo Team*

**TL;DR:** A 5-second 480p video, generated entirely on a Mac — no cloud, no
NVIDIA GPU, no discrete graphics card. FastVideo now runs its 3-step
distilled FastWan models natively on Apple Silicon through an MLX runtime,
and we are preparing **FastWan-QAD-INT8-1.3B**, a model trained with
Quantization-Aware Distillation against MLX's own INT8 quantizer. On the
motion7 benchmark, INT8 QAD tracks each model's own FP16 output better than
standard post-training quantization: 0.936 mean MS-SSIM for the raw student
and 0.933 for EMA, vs 0.907 for stock PTQ. The whole DiT+TAEHV path fits
around 5.6 GiB of MLX peak memory on the M4 Max measurement, with a 16 GiB
cap stress run passing at 4.7 GiB.

[FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/) pushed a single RTX
5090 to generate a 5-second video in 1.8 seconds by co-designing the model,
the quantization format, and the runtime for Blackwell tensor cores. This
release asks the follow-up question: what does that co-design look like on
the most widespread capable consumer hardware in the world — the Apple
Silicon Mac? A Mac will not match a 5090 on raw speed. But video generation
that requires a $2,000 GPU is not local video generation for most people.
Tens of millions of Macs ship with a GPU and 16 GB+ of unified memory
already on someone's desk. Making those machines first-class is how local
video generation actually becomes local.

## What We Are Releasing

- **FastWan-QAD-INT8-1.3B** `[TODO: HF link]` — Wan2.1-T2V-1.3B distilled to
  3 sampling steps with quantization-aware DMD against MLX's affine INT8
  quantizer. Raw and EMA exports are both valid candidates; the final release
  pick is the one that wins the visual grid review. It will ship in two
  formats: Diffusers-style safetensors, and a **pre-quantized MLX checkpoint**
  that halves the download and skips requantization at load.
- **The MLX runtime** — a native Apple-Silicon implementation of the Wan DiT
  (patch embed → transformer stack → unpatchify) with an on-device 3-step
  DMD sampler, pinned against the PyTorch reference by bitwise/tolerance
  parity tests that run in CI on both Metal and CPU backends.
- **The full QAD-for-MLX training recipe** — a composable `mlx_qat` callback
  for FastVideo's modular trainer plus the exact YAML we trained with, and
  the fake-quantizer whose numerics are pinned **bitwise** against
  `mx.quantize` in the test suite.
- **A memory-tier benchmark harness** — `mac-16gb` / `mac-32gb` / `mac-64gb`
  presets sweeping quantization × decoder × prompts, emitting latency, peak
  unified memory, MS-SSIM, and side-by-side HTML grids.

| Artifact | Format | Target | Tier |
| --- | --- | --- | --- |
| FastWan-QAD-INT8-1.3B | Diffusers safetensors | any FastVideo backend | reference weights |
| FastWan-QAD-INT8-1.3B-MLX | pre-quantized MLX checkpoint | Apple Silicon, 16 GB+ | flagship: INT8, half the download, no requantize at load |

Everything is Apache-2.0, same as the original FastWan-QAD release.

## The Mac Inference Stack

Like the original release, the speedups come from attacking every layer of
the stack — but the layers are different on a Mac, and the co-design target
is MLX rather than tensor cores.

**MLX-first DiT, dense attention.** The denoising loop — the dominant cost —
runs natively in MLX on the Metal GPU: full Wan transformer forward with
`mx.fast.scaled_dot_product_attention`, kept 100% dense like FastWan-QAD.
The DMD sampler runs on-device too; no tensor leaves unified memory during
the 3-step loop.

**INT8 everywhere it counts.** Every DiT matrix weight is quantized with
MLX's affine INT8 (group size 64) and executed with `mx.quantized_matmul`;
norms and modulation tables stay fp16. INT8 is the reliability sweet spot on
today's MLX across Apple generations — INT4/MXFP4 are on the roadmap behind
the same QAD gate.

**Memory choreography for 16 GB.** The UMT5 text encoder is loaded, used,
and freed (optionally in a subprocess the OS fully reclaims) before the DiT
ever loads; decode defaults to TAEHV, a tiny autoencoder that removes the
full Wan VAE from both the latency and the memory peak; allocator caps and
MPS watermarks are first-class preset options. Measured result: the full
480×832×81 pipeline peaks at **4.7 GiB of MLX memory under a hard 16 GiB
cap** — comfortable headroom for a 16 GB machine running an OS.

**Pre-quantized checkpoints.** Quantizing 1.3B parameters at every startup
is wasted work. The MLX checkpoint format stores the packed INT8 weights,
scales, and biases directly — reloads skip requantization entirely and the
download is roughly half the fp16 bytes. `[TODO: measured load_s
mlx_checkpoint vs diffusers from bench metrics]`

## Quantization-Aware Distillation, Retargeted to MLX

None of this matters if INT8 visibly degrades the video — and with
post-training quantization, it does. The original FastWan-QAD recovered
NVFP4 quality by making the model live in its deployment precision during
training. We did the same thing for MLX, which required one uncompromising
detail: **the training-time fake quantizer is a line-by-line transcription
of MLX's quantization kernel** — the fp32 group min/max, the negative-scale
anchoring, the integer zero-point re-fit, the round-half-to-even — pinned
bitwise against `mx.quantize`/`mx.dequantize` by tests that gate every PR.
If train-time and deploy-time quantizers disagree even subtly, the QAT gains
evaporate at load; ours cannot disagree, by construction.

The recipe is quantization-aware DMD: a frozen Wan2.1-1.3B teacher, a critic,
and a student whose every forward computes on the INT8 deploy grid
(gradients pass straight through to the real weights), distilled to the
3-step schedule the runtime ships. The second training run used FastWan
student/critic initialization and `gradient_accumulation_steps=4`; it is the
current launch candidate. Raw and EMA are close on the metric, so the final
choice is visual: raw is slightly sharper by mean fidelity, while EMA has
the better worst-case score and may suppress temporal artifacts.

## Results

**Does QAD survive quantization?** MS-SSIM of each model's INT8 output
against its own FP16 output, motion7 prompt set, shared seeds (higher =
less quantization damage):

| Model | mean MS-SSIM (INT8 vs own FP16) | worst prompt |
| --- | ---: | ---: |
| stock FastWan2.1-1.3B (post-training quantization) | 0.9069 | 0.8214 |
| FastWan-QAD-INT8 v2 (raw student) | 0.9360 | 0.8848 |
| FastWan-QAD-INT8 v2 (EMA) | 0.9331 | 0.8875 |

Both QAD v2 students beat stock post-training quantization on this metric.
This score is intentionally narrow: it measures quantization robustness, not
absolute video quality. We use the HTML grids to make the ship decision
because they reveal temporal smoothness, fine detail, and artifacts that a
single fidelity number cannot.

**Speed and memory** (Apple M4 Max, 480×832×81, 3-step DMD, TAEHV decode):

| Config | Denoise | Decode | Total | MLX peak |
| --- | ---: | ---: | ---: | ---: |
| INT8 + TAEHV | ~112-115s | ~2s | ~2 min | 5.6 GiB |
| INT8 + TAEHV, 16 GiB hard cap | — | — | passes | 4.7 GiB |
| FP16 + full Wan VAE | ~114s | ~122s | ~4 min | 6.9 GiB |

`[TODO: add mx.compile row once the A/B lands; add a stock 16 GB machine row]`

Two minutes is not 1.8 seconds — and that is the honest headline. It is
also a 5-second, 480p, text-to-video clip generated on battery-powered
consumer hardware with zero cloud dependency, in less memory than a browser
session. The optimization ladder that took the CUDA stack from 170s to 1.8s
is the same ladder we are now climbing on Metal, and the first rungs
(`mx.compile`, fused kernels) are already in the harness behind quality
gates.

Visual review artifacts:

- Motion7 raw grid: `bench/apple_qad_v2/qad/index.html`
- Motion7 EMA grid: `bench/apple_qad_v2/qad_ema/index.html`
- Screenshot-style qualitative grid:
  `bench/qad_v2_picture_prompts/gallery.html`

`[TODO: convert the qualitative grid into the final blog figure. Do not list
all prompt text in the post; keep it as a visual comparison.]`

## How to Run

```bash
pip install 'fastvideo[mlx]'   # Apple Silicon; installs the MLX extra

python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root <FastWan-QAD-INT8-1.3B path or HF id> \
  --mlx-quantization int8 \
  --save-mlx-checkpoint ~/models/fastwan-qad-int8-mlx \
  --prompt "A fox runs through a misty pine forest."

# later runs: reload the pre-quantized checkpoint, skip requantization
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --mlx-checkpoint ~/models/fastwan-qad-int8-mlx \
  --prompt "..."
```

`[TODO: replace with final HF id / one-line quickstart once published]`

## Next Steps

The 1.3B text-to-video model is the proof, not the ceiling. Next on the
Apple track: `mx.compile` and fused-norm kernels on by default (the SSIM
regression gate is already wired), Wan2.2-TI2V-5B for the 32 GB+ tiers —
which also brings image-to-video — and INT4/MXFP4 behind the same QAD gate
that made INT8 shippable. In parallel, the torch-MPS compatibility lane
carries FastVideo's broader model families to Macs while the MLX fast lane
grows one architecture at a time. The runbooks, parity gates, and benchmark
presets in the repo are designed so contributors can climb this ladder with
us.

We welcome feedback, contributions, and collaboration — join the FastVideo
Slack or open an issue. `[TODO: links]`

## Acknowledgements

This work builds directly on
[FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/) by the FastVideo team
at Hao AI Lab (UCSD) — the QAD recipe, the FastWan models, and the
distillation infrastructure this release retargets to Apple Silicon. TAEHV
is by Ollin Boer Bohan (MIT). `[TODO: contributor list, compute
acknowledgements (DGX B200), advisor list]`
