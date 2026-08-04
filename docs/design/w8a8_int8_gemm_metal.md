# W8A8 Fused INT8 GEMM for Metal — Design + Calibration Evidence

Status: design, 2026-08-04. The "unclaimed" speed lever from
`mac_quantization_findings.md` #1: weight-only quantization cannot buy speed
on Metal (every GEMM runs fp16 arithmetic on dequantized weights; the integer
matrix units sit idle). The only route to quantization-derived speed is
quantizing **activations as well as weights** (W8A8) so the integer compute
path engages. Ideogram shipped this for consumer GPUs (fused INT8 GEMM);
nobody has done it for diffusion on Metal against M5 Neural Accelerators.

## Gate 1 — activation calibration: PASS (with per-token scales)

Data: `w8a8/calib_5b.json` (GB200 job 1112) — per-Linear input-activation
stats for the Wan2.2-TI2V-5B DiT, 64 samples × the 3 DMD timesteps at 720p.

1. **Per-timestep drift is modest (~20–30%).** attn2.to_q absmax 22.7 (t1000)
   → 16.1 (t522); most layers vary less. A single scale per layer (or per
   step-bucket) suffices for the 3-step DMD schedule — no per-timestep
   recalibration machinery needed.
2. **Bulk fits int8 with per-token scales.** With per-tensor absmax scaling
   the bulk (p999) uses ~57–88 of 127 levels on most layers — viable.
   Per-token (per-row) scales recover the rest and are the standard choice.
3. **The outlier risk is concentrated in the text-conditioning path.**
   `attn2.to_q/k/v` and `condition_embedder.text_embedder.linear_2` have
   p999/absmax as low as 0.22 (bulk uses only ~28/127 levels under absmax
   scaling). These are a small fraction of FLOPs — keep them at fp16 or give
   them per-token scales; do not let them force the whole model coarse.

Verdict: W8A8 is viable at int8 with per-token activation scales + fp16 for
the cross-attention conditioning projections.

## Gate 2 — integer GEMM kernel: OPEN (the hard one)

MLX 0.32 (the pinned release) exposes only **weight-only**
`mx.quantized_matmul` (x float, w quantized → fp16 arithmetic). There is no
quantized×quantized matmul. So W8A8 requires a **custom Metal kernel**
(`mx.fast.metal_kernel`) doing int8×int8 → int32 accumulate → requantize,
fused with the on-the-fly activation quantization (per-token scale) and the
dequant-to-fp16 output stage.

- Correctness can be developed/validated on any Apple Silicon Mac.
- The **speed gate needs the M5**: only there do the Neural Accelerator
  integer units exist. Benchmark target: beat the fp16 GEMM at DiT shapes
  (8192×5120×5120 class) by a meaningful margin; anything ≤1.2× is not worth
  the complexity (see the survey's traffic accounting — weights are ~24% of
  traffic at dim 5120).
- Fallback if the custom kernel underdelivers: per-token-scaled activation
  quantization feeding the existing weight-only path (no speed win, small
  memory win) — document and stop.

## Gate 3 — W8A8 QAT: after Gate 2

Activation quantization error compounds across denoising steps, so a QAT pass
(W8A8 fake-quant on weights *and* activations, timestep-aware scales from Gate
1) is required for quality. Structure follows the proven int8 QAD recipe with
the activation fake-quant added to the same callback. GB200s available.

## References

- Calibration: `w8a8/calib_5b.json` (this repo task dir on the cluster)
- Format decision + traffic accounting: `mac_quantization_findings.md`
- Ideogram INT8 GEMM precedent: https://arxiv.org/html/2606.14598v1
