# Where Mac Speed Actually Comes From

Researched answer to two questions: what is everyone else shipping, and is int8
even faster on M5. Closes the format discussion.

## Is int8 faster on M5? No. No weight-only format is.

The survey measured int8, mxfp4 and nvfp4 within noise of each other. The
literature explains why, and it is not the reason I first gave:

> Weight-only quantization gives essentially no diffusion speedup because
> diffusion is compute-bound even at batch one, so low-bit weights are simply
> upcast and the integer compute path is never engaged. When multiplying FP16
> activations against dequantized-to-FP16 weights, every GEMM operates at FP16
> arithmetic intensity, and the GPU's INT8 tensor operation units sit completely
> idle.

`mx.quantized_matmul` dequantizes to fp16 and does fp16 arithmetic. The M5
neural accelerators' integer paths never run. **The format was never the
problem — weight-only quantization cannot produce speed on a diffusion model,
in any format.**

This is also why the LLM evidence misled us. LLM *decode* is memory-bound on the
KV cache and weights, so 4-bit is a real speedup there — that is the Ollama
NVFP4 result and the reason mlx-community standardised on 4-bit group 64 across
~4,800 models. A diffusion transformer at batch one is the opposite regime.

**Conclusion: on Mac, quantization is a memory decision only. Pick the bit width
that fits the model and stop expecting speed from it.**

## What others actually ship

| Who | Approach |
|---|---|
| **Draw Things** | **int8** as the practical default, described as "virtually lossless"; first choice for large models on 8 GB Macs. In-app quantization. |
| Draw Things (speed) | **Metal FlashAttention** — 43–120% faster generation, and **v2.5 with Neural Accelerators is up to 4.6× on M5 over M4**. This, not quantization, is where their speed comes from. |
| mlx-community | 4-bit group 64 as the LLM default across ~4,800 models. LLM regime, does not transfer. |
| JANG | Adaptive per-layer bit widths — attention 4–8 bits, MLP 2–6 bits. Lower average width than uniform 4-bit with better fidelity. |
| FlexQ | **INT6** post-training quantization via algorithm-system co-design. int6 is a real engineering target, not an oddity. |
| Ideogram 4.0 | **Fused INT8 GEMM kernel** for diffusion transformers — native INT8 *compute*, not weight-only. On consumer GPUs. |

The pattern is clear: **serious Mac diffusion ships int8 for memory and gets its
speed from attention kernels.** Draw Things is the strongest existing product on
this hardware and that is exactly their split.

## Where the innovation actually is

MXFP was never it. Four real openings, in order of payoff.

### 1. W8A8 with a fused integer GEMM for Metal — the unclaimed one

Quantize **activations as well as weights** so the integer compute path
engages. Ideogram did this for consumer GPUs with a fused INT8 GEMM; nobody has
done it for diffusion on Metal against M5's neural accelerators.

This is the only route to a real speedup from quantization, and it is
genuinely novel work rather than a port. It is also hard — a custom Metal
kernel, plus activation quantization needs timestep-aware calibration because
DiT activation distributions shift across the denoising trajectory.

Highest payoff, highest effort, and the thing that would make FastVideo-on-Mac
distinctive.

### 2. Metal FlashAttention-class attention — the proven one

Draw Things gets 43–120% from this and 4.6× on M5. Attention is where the
activation traffic and compute both concentrate at 25k latent tokens. Verify
what the MLX runtime currently dispatches for attention and whether an
MFA-class path is reachable.

Lower effort than #1, already proven on this hardware, and it compounds with
everything else.

### 3. Per-layer mixed precision — cheap and immediate

JANG's result: attention at 4–8 bits and MLP at 2–6 bits beats uniform 4-bit at
a *lower* average width. The existing `mlx_qat` callback already excludes norms
and modulation tables; extending that to a per-layer-type width policy is a
small change with a measured precedent.

### 4. int6 — the memory sweet spot

FlexQ validates INT6 as a serving target. Predicted ~0.022 relative error at
6.5 bits/param, which puts a 14B at ~11.4 GB. Still pending measurement, still
the cheapest open question.

## What this means for the plan

**Quantization track:** settled. Affine int8 now; int6 if the follow-up survey
holds. Drop MXFP and NVFP4 entirely — worse reconstruction at equal memory, and
no speed advantage to compensate.

**Speed track:** this is where the work is, and it was previously mis-assigned
to quantization. In order: fewer denoising steps (already the plan), attention
kernels (#2), sparse attention, resolution-plus-upscale, then W8A8 (#1) as the
ambitious bet.

**The MXFP failure is now fully explained.** Three independent reasons, none of
which were visible at the start: it is the worst-reconstructing format tested,
it could not have delivered speed regardless because weight-only never engages
the integer path, and the runs themselves used an uncommitted quantizer with no
parity test.

## Sources

- [Draw Things — Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [Draw Things — MFA v2.5 with Neural Accelerators on M5](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural)
- [Realizing Native INT8 Compute for Diffusion Transformers (Ideogram 4.0)](https://arxiv.org/html/2606.14598v1)
- [When Quantization Is Free: int4 KV Cache on Apple Silicon](https://arxiv.org/html/2605.05699)
- [FlexQ: INT6 post-training quantization](https://arxiv.org/pdf/2508.04405)
- [GGUF vs AWQ vs GPTQ vs MLX, 2026](https://www.digitalapplied.com/blog/gguf-vs-awq-vs-gptq-vs-mlx-llm-quantization-formats-2026)
- [The Quantization Method Apple Silicon Actually Rewards](https://medium.com/@alexandru_vasile/i-benchmarked-every-quantization-method-for-apple-silicon-llms-heres-what-actually-wins-7b3e7edff4ef)
