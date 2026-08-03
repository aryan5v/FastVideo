# Mac QAD Two-Product Strategy (2026-08-03)

Owner framing: deliver two classes of local video generation on Apple
Silicon — **really good quality**, and **really fast 480p** — with 720p or
higher in scope for the 14B. Every model ships through the QAD recipe
(DMD2 + affine int8 QAT) so the training grid is the deploy grid.

## The two products

| | **Fast / wide** | **Quality** |
|---|---|---|
| Model | **Wan2.2-TI2V-5B** int8 | **Wan2.1-14B** int8 |
| Base checkpoint | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` (already 3-step DMD, dense attention) | `FastVideo/FastWan2.1-T2V-14B-Diffusers` (already 3-step) + teacher `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Native shape | 121×704×1280 (**720p**, 24 fps) | 77×448×832 (480p); 720p follow-up via `FastVideo/Wan-Syn_77x768x1280_250k` |
| Weights at int8 | ~5.3 GB | ~14.9 GB |
| Mac tier | **16 GB+** (the wide install base) | **24 GB+** |
| Corpus | `FastVideo/Wan2.2-Syn-121x704x1280_32k` (4.2k files) | `FastVideo/Wan-Syn_77x448x832_600k` (~1.6 TB) |
| Full run | job TBD after smoke (8×GB200, 2×4 mesh) | job 1092 (16×GB200, 2×8 mesh) |
| Smoke | job 1093 (queued) | job 1091 (running) |

## Why Wan 2.2 5B is worth it (and is "better than 2.1" for our purposes)

- **720p-native at 5B.** Wan2.2-TI2V-5B was built for consumer-resolution
  generation; quality per GB beats Wan2.1-14B at 720p and below, and it is
  the only one of the two that fits 16 GB Macs at int8.
- **Already 3-step, dense attention.** The FullAttn variant needs no sparse
  kernel on Metal, and the v2 recipe ("adapt a good 3-step model to the int8
  grid") applies verbatim — no step distillation, just QAD adaptation.
- **Cheap to run.** 5B × 3 roles is under half the 14B training state; the
  corpus is 4.2k files, not 13.9k; data-free DMD precedent exists in-tree
  (`examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/`).
- 14B remains the quality ceiling for 24 GB+ machines and the 720p/1080p
  aspiration. The products share the recipe, the runtime, and the launch.

## Do we need QAT? Yes — decided

- The only shipped QAD model (FastWan-QAD-1.3B) is a QAT artifact; PTQ-only
  was never accepted at 3 steps, where the student is most error-sensitive.
- QAT is free at training time (we are training anyway) and the affine int8
  fake-quantizer is committed with a bitwise MLX parity test.
- On Mac, attention stays dense fp16 — no Attn-QAT stage (unlike the 5090's
  FP4-attention path). Weights-only int8 is the correct Metal grid (M5 survey).

## What we borrow from MiniMax-H3 (architecture study, 2026-08-03)

1. **Base + regenerate two-pass** (H3: 768p base → in-context 2K regeneration
   with the original context). Validates the fast/quality split: for the
   Quality tier, generate at base res, then a second refinement pass with the
   *same* model conditioned on the low-res result — no dedicated SR module.
   (H3's Regenerate-2K is API-only, but the pattern is documented.)
2. **AdaLN modulation caching.** H3 keeps ~13B of AdaLN projections out of
   the inference-resident set by precomputing per (timestep, modality). The
   Wan 3-step schedule has exactly 3 timesteps — cache the modulation tables
   at load. Free runtime win in the MLX runtime.
3. **Context-IR-style prompt enrichment.** H3's first stage rewrites raw
   prompts into structured shot/soundscape descriptions and is "critical to
   final quality." A local prompt-upscaling pre-pass (small LLM, no training)
   is a cheap quality lever for the Mac runtime; H3's prompting guides are
   the template.
4. **Fixed-seed posterior sampling + fp16-rounded conditioning latents** for
   reproducible I2V conditioning (when I2V enters the Mac track).

Not borrowed: the ViT VAE decoder (Wan VAE is fixed), MSA sparse attention
(deferred everywhere), the packed AV sequence (H3-specific).


## Runtime deltas for the 5B (checkpoint inspection, 2026-08-03)

The FastWan2.2-TI2V-5B checkpoint uses the **same key layout as Wan2.1**
(blocks.N.{attn1,attn2,ffn,norm2}, condition_embedder.*, patch_embedding,
proj_out, scale_shift_table) and **has** attn/ffn biases, so the MLX loader
surface is compatible. Three semantic deltas the MLX runtime must handle
before the 5B ships:

1. `qk_norm: "rms_norm_across_heads"` — norm_q/norm_k weights are [3072]
   (full inner dim), not per-head [128]; attention needs across-heads RMSNorm.
2. 48-channel latents — patch_embedding [3072, 48, 1, 2, 2]; the Wan2.2 TI2V
   VAE emits 48 channels (vs 16 in Wan2.1); TAEHV does not cover it, so the
   decode path needs the full Wan2.2 VAE (or a new tiny decoder later).
3. `expand_timesteps: true` — time_proj is [18432, 3072] and the DiT consumes
   per-frame timestep embeddings; the sampling loop must expand scalar steps.

## Sequencing (not reckless)

1. 14B: smoke 1091 → full 1092 (auto-chained). ~40 h train when it starts.
2. 5B: smoke 1093 queued; **5B full run held** until 5B smoke passes AND the
   14B full run is underway — no two 4000-step runs competing for nodes.
3. Runtime wins land independently of training: prompt-embedding cache
   (done, `mlx-release-wins`), AdaLN caching (from H3), sequenced residency.
