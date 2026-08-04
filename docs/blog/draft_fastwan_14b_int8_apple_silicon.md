# DRAFT — FastWan INT8 for Apple Silicon: 1.3B, 5B, and 14B Video Models, Locally on Your Mac

> Status: DRAFT for the Mac launch, mirroring the FastWan-QAD RTX 5090 post
> format. Every number marked `TODO(measure)` is a placeholder until the QAD
> runs (wandb `distillation_wan`: `wan2.1_14b_dmd2_3steps_mlx_int8`,
> `wan2.2_5b_dmd2_3steps_mlx_int8`) complete and the benchmark harness records
> final figures. Do not publish with placeholders.

**TL;DR: Three video models, all running fully on your Mac. FastVideo
introduces the FastWan INT8 family — 1.3B, 5B, and 14B checkpoints trained
with Quantization-Aware Distillation (QAD) against the exact INT8 grid Apple
Silicon deploys on. From a 16 GB MacBook Air to a 64 GB Mac Studio: a 3-step,
5-second video in TODO(measure) seconds, up to 720p on the 5B, up to the
largest model we have ever shipped on a laptop — no cloud, no GPU server.**

## What We Are Releasing

Three QAD checkpoints, one recipe, one deploy grid:

| Checkpoint | Base model | Native output | INT8 size | Mac tier | Role |
|---|---|---|---|---|---|
| **FastWan-QAD-1.3B-INT8** | Wan2.1-T2V-1.3B | 480p, ~5 s | ~1.4 GB | 16 GB+ | Fastest; the proven release |
| **FastWan-QAD-5B-INT8** | Wan2.2-TI2V-5B (3-step, full attention) | **720p (121×704×1280), ~5 s** | ~5.3 GB | 16 GB+ | **Fast + high-res: the value pick** |
| **FastWan-QAD-14B-INT8** | Wan2.1-T2V-14B | 480p, ~5 s | ~14.9 GB | 24 GB+ | Maximum local quality |

- All three are **3-step** students trained with DMD2 + affine-INT8 QAT, so
  the training grid *is* the deploy grid.
- The full **QAD training recipe** — configs, the affine-INT8 fake-quantizer
  with its bitwise MLX parity test, and the DMD2 training code in FastVideo.
- The **MLX runtime**: on-device DMD sampling, fused Metal attention,
  `mx.compile`, TAEHV fast decoding, memory-tier presets, and the RIFE fast
  mode.
- Apache-2.0, weights and code.

**Two ways to buy speed, one way to buy quality.** The 5B at 720p is the
fast/high-res option for 16 GB Macs; the 14B is the quality ceiling for 24 GB+
machines; the 1.3B is the instant-gratification download. All launch together
with the same runtime and the same recipe.

## Why INT8 — and Why Not MXFP4 or NVFP4

We measured this rather than assuming it, on an Apple M5 (24 GB, macOS 26.5,
MLX 0.32), across every candidate 4- and 8-bit format.

**Integer formats beat float formats at equal bit width — decisively.** Affine
INT8 (8.5 bits/param) reconstructs weights **12.4× more accurately than MXFP8**
(8.25 bits/param). The mechanism: E4M3 float carries 3 mantissa bits (~6%
relative precision per element), while affine integer quantization spends all 8
bits on 256 uniform levels across each group's actual range, with a scale *and*
a bias to fit it. At 4 bits the ordering holds: affine INT4 beats NVFP4 beats
MXFP4.

**No weight-only format buys speed on Metal — so the choice is purely about
memory.** Diffusion transformers at batch one are compute-bound: activations,
not weights, dominate memory traffic (weights are ~9% of traffic at our
shapes). `mx.quantized_matmul` dequantizes to fp16 and runs fp16 arithmetic;
the integer matrix units never engage. We measured INT8, MXFP4 and NVFP4
within noise of each other on wall-clock. This is also why the LLM
quantization folklore misleads here: LLM *decode* is weight-bandwidth-bound
(one token at a time), so 4-bit is a real speedup there; a video DiT at 8k+
tokens is the opposite regime.

**MXFP4 specifically failed us twice** — at 1.3B and at 14B, with and without
QAT — before we understood why: it is the worst-reconstructing format we
tested (22× INT8's error), it could never have delivered speed regardless, and
our early runs used an uncommitted quantizer with no parity test. All three
failure causes are now closed: the format is dropped, the affine INT8
fake-quantizer is committed with a bitwise MLX parity test, and the training
grid *is* the deploy grid.

## The Receipts: INT8 vs MXFP4, Side by Side

We did not pick INT8 from theory — we ran MXFP4 first, twice, and it failed
twice. Below are generations from the **MXFP4 QAD runs that did not qualify**
alongside their INT8 counterparts.

| Run | Format | Outcome |
|---|---|---|
| Wan2.1-1.3B QAD | **MXFP4** | TODO(embed): visible artifacts / motion smear — rejected |
| Wan2.1-1.3B QAD | **INT8** | TODO(embed): accepted — shipped as FastWan-QAD-1.3B |
| Wan2.1-14B QAD | **MXFP4** (4000 steps, 26 GPU-days) | TODO(embed): washed-out, undertrained-looking — rejected |
| Wan2.1-14B QAD | **INT8** (this release) | TODO(embed): the launch checkpoint |

The reconstruction numbers explain the videos: at equal memory, MXFP4 is the
worst-reconstructing format we measured (22× INT8's relative error), and no
4-bit format buys any speed on Metal anyway (weights are ~9% of memory
traffic at our shapes — the integer units never engage). MXFP4 lost on
quality *and* had nothing to offer on speed. There was no reason to ship it.

**Attention stays dense and fp16.** On the RTX 5090, native FP4 *attention*
was worth an Attn-QAT training stage because Blackwell's FP4 tensor cores pay
real speed. On Metal there is no low-bit attention win to recover, so we keep
attention 100% dense in fp16 and quantize weights only — no attention
degradation to train away, and no Attn-QAT stage needed.

So: **INT8, because on Apple Silicon quantization is a memory decision, and
INT8 is the most accurate way to spend 8 bits.** 14.9 GB for 14B parameters —
a model class that used to need a datacenter GPU now fits in 24 GB of unified
memory.

## The Training Recipe: QAD at 14B

Quantization-Aware Distillation trains the student *against* the exact
quantization grid it will deploy on, so the network learns to be accurate
*despite* quantization rather than inheriting the error afterward.

- **Student + critic** initialize from the already 3-step-distilled FastWan
  14B checkpoint; the **teacher** is the base 50-step Wan2.1-14B. Training is
  "adapt a good 3-step model to the INT8 grid," not learning distillation
  through a quantizer — the v1 lesson.
- **DMD2** distribution matching (3 resident networks: frozen teacher,
  trainable student, trainable critic) with the affine-INT8 fake-quantizer
  active on every student forward, straight-through gradients to the real
  weights.
- 4000 iterations on **16 NVIDIA GB200s** (2×8 HSDP mesh), ~TODO(measure)
  hours wall-clock, on the Wan-Syn 600k corpus.
- The fake-quantizer is forward-scoped (weight swap inside each wrapped
  forward), which keeps FSDP2 sharding, optimizers, and checkpointing
  untouched — the subtle part that makes QAT compose with 2D HSDP.

## The Inference Stack

Every layer is attacked, as with the 5090 release — but for Metal:

- **MLX-native DiT**: the 14B transformer runs in MLX with fused
  `mx.fast` attention and normalization kernels and `mx.compile` over the
  denoise graph.
- **On-device DMD sampling**: the 3-step DMD loop never leaves the device —
  no host round-trips, MLX lazy execution intact.
- **INT8 pre-quantized checkpoints**: save/load the deploy artifact directly;
  the loader casts to fp16 and quantizes exactly along the training grid.
- **TAEHV fast decoding** with checksum-verified weights.
- **Fast mode**: **RIFE** (generate every Nth frame, interpolate the
  rest with Apple-silicon-native rife-mlx), with a MetalFX spatial
  half-resolution render path on the roadmap — temporal and spatial
  shortcuts that multiply.
- **Memory-tier presets**: 24/32/64 GB, with sequenced residency (encode →
  free encoder → denoise → free DiT → decode) so peak memory is the largest
  stage, not the sum.

## Why Now: Apple Silicon Caught Up

- **M5 Neural Accelerators**: per-core matrix units; Metal FlashAttention-class
  kernels are up to 4.6× faster on M5 than M4 (Draw Things' MFA 2.5 numbers) —
  attention is where diffusion compute concentrates.
- **MLX matured**: fused `mx.fast` kernels, `mx.compile`, quantized
  checkpoint formats, and a stable allocator with explicit memory caps.
- **macOS 26 / Metal 4**: the driver-level path the probes verify is actually
  engaged (our runtime refuses to trust a silent fallback — see
  `accel_probe.py`).

## Quality and Speed

TODO(measure): SSIM/LPIPS vs the bf16 50-step teacher and vs the mxfp4
checkpoint; VBench; wall-clock and time-to-first-frame on M5 24 GB / 32 GB /
64 GB; RIFE and MetalFX ablations; memory peaks per tier.

## What Is Next

FastWan-14B-INT8 is Release 0 of the Apple Silicon track: the entire stack —
INT8 QAD, the MLX runtime, sequenced residency, fast modes — is
model-agnostic. The same machinery is now pointed at **MiniMax-H3**, whose
33B omni-modal (video + native audio) teacher gets the same treatment: a
distilled student, INT8, local on your Mac.
