# FastH3 MLX Bake-Off — Agent Brief (36 GB Mac)

You are an agent running on an **Apple Silicon Mac with 36 GB unified memory**.
Your job: run the FastH3 (MiniMax H3, 4-step DMD2-distilled 33B) **MLX
generation bake-off** for the **int8 / int6 / int4** deploy grids, using the
merged MLX runtime in the fork, and report quality + metrics back. Use ALL of
the FastH3 runtime advancements: **AdaLN precompute cache, cached 4-step
DMD2-ladder sampling, int-grid quantization of attention/FFN only**.

Follow the steps in order. If anything fails, diagnose and fix it in the fork
branch (commit to `fasth3-mlx-runtime`, fork-only — never push upstream), or
report the failure with the full error.

## 0. Context (why this exists)

FastVideo upstream released `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2`:
a full 33B DMD2-distilled student walking the trained 4-step ladder
(sigma grid = 5 points, 4 DiT forwards; video shift 12, audio shift 3;
guidance-free). A cluster-side weight-level probe already ranked the grids on
these exact weights (int8 ≈ 0.00003, int6 ≈ 0.0005, int4 ≈ 0.0089 rel-L2;
mxfp8 ≈ 0.022; mxfp4 ≈ 0.12+, worst). This bake-off CONFIRMS those numbers at
the forward level and measures real speed + memory on this Mac — **the
decision table decides which grids become the shipped MLX artifacts**.

## 1. Environment setup

```bash
cd ~
git clone --branch fasth3-mlx-runtime https://github.com/aryan5v/FastVideo.git fv-h3
cd fv-h3
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "mlx>=0.32" numpy safetensors huggingface_hub
# torch is NOT needed on this Mac (decode happens on the GPU cluster).
python -c "import mlx.core as mx; print(mx.__version__, mx.metal.is_available())"
```

Notes: the repo is importable from the checkout (`PYTHONPATH=$PWD` or run
from the repo root). Check disk: you need ~66 GB (source weights) + ~60 GB
(converted checkpoints) free.

## 2. Get the weights (one-time, 66 GB)

```bash
huggingface-cli login   # token required — the repo may gate while its license review completes
huggingface-cli download FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2 \
  --local-dir ~/models/FastH3-Preview-v0.2
```

Verify the layout: `~/models/FastH3-Preview-v0.2/transformer/config.json` +
`diffusion_pytorch_model.safetensors.index.json` (14 shards) exist.

## 3. Convert to pre-quantized MLX checkpoints (streamed — never holds the
66 GB bf16 resident; this is THE reference build path)

```bash
python scripts/fasth3/convert_h3_preview.py \
  --model-root ~/models/FastH3-Preview-v0.2/transformer \
  --out ~/models/FastH3-MLX \
  --formats "int8 int6 int4 fp16"
```

- `fp16` is the reference (33 GB — the runtime's natural dtype; bf16 does
  NOT fit 36 GB and is CUDA-side anyway).
- int grids are affine, group 64, attention/FFN only; input/output
  projections, norms, embeddings stay high-precision (the converter handles
  this automatically; the released fp32 keep-set is respected).
- Each output (~2 min) is a shippable `mlx_h3_dit.safetensors` artifact.
  Record peak memory (`mx.get_peak_memory`) from the converter output.
- If MLX reports the mxfp8/mxfp4 kernels unsupported on this build, note it;
  don't block on it.

## 4. Run the bake-off

```bash
python -m fastvideo.benchmarks.mlx_h3_quant_survey \
  --mlx-checkpoints ~/models/FastH3-MLX \
  --formats "int8 int6 int4" \
  --latents-out ~/survey_latents
```

The survey uses the **FastH3 runtime path** for everything:
- **AdaLN precompute cache**: all modulation tables for the ladder computed
  once; the ~13B of AdaLN projection weights are DROPPED before measurement;
  resident memory is reported AFTER the drop.
- **Cached 4-step DMD2-ladder denoise** (`forward_with_cache`) — video shift
  12 / audio shift 3, data-ward velocity; per-step ms from the cached path.
- **Reconstruction**: one forward at the ladder midpoint vs the fp16
  reference — the forward-level confirmation of the weight-level probe.
- Latents (video rows + audio rows) are dumped per format to
  `~/survey_latents/latents_<format>.npz` for offline cluster decode.

Also run ONE mxfp4 speed-only probe (format `mxfp4`), solely to answer the
E1 question — is mxfp4 materially faster than int8 on this Mac (real
accelerator dispatch) or within noise (emulation)? Quality columns for MX
are ignored (already ranked worst).

Sanity: `int8` per-step should be the baseline; `int6`/`int4` should be
within a few percent (weights are <25% of traffic at DiT shapes — if a
4-bit grid is dramatically faster, investigate and report).

## 5. Quality eyeballs (latent handoff to the cluster)

The MLX VAE ports do not exist yet, so **decode on the GPU cluster**:

```bash
# on the cluster (SLURM, vlm-aryan account):
python scripts/fasth3/decode_survey_latents.py \
  --latents-dir ~/survey_latents \
  --vae-root /mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/<snapshot> \
  --out ~/survey_mp4
```

Transfer `~/survey_latents` there (or mount), then pull the mp4s back and
eyeball: motion coherence, high-motion detail, audio-video sync, artifacts.
Rank int8/int6/int4 qualitatively.

## 6. Report back (markdown, `~/fasth3_bakeoff_report.md`)

| Format | fwd rel-L2 vs fp16 | per-step ms | resident GiB (after AdaLN cache) | load s | eyeball verdict |
|---|---|---|---|---|---|
| int8 | | | | | |
| int6 | | | | | |
| int4 | | | | | |
| mxfp4 (speed-only) | n/a | | | | |

Plus:
- MLX version + chip (`mx.metal.device_name()`), macOS version
- The E1 verdict: mxfp4 vs int8 per-step ratio (≥15% faster = real hardware path; ~1× = emulation)
- Any failures + fixes committed (branch `fasth3-mlx-runtime`)
- Recommendation: which grid(s) should ship as the 33B MLX artifact (int8/int6/int4), including the ~10.1–19.1 GiB resident trade-off for the 36 GB tier
- The exact artifact paths (`~/models/FastH3-MLX/<fmt>/`) — these are the candidates for HF publication

## 7. Rules

- **Do not** run a bf16 resident model (66 GB — will not fit 36 GB); fp16 is
  the reference.
- **Do not** modify upstream code — fork branch `fasth3-mlx-runtime` only.
- **Do not** upload artifacts anywhere without explicit approval.
- Keep the process memory-light: one format resident at a time; never the
  diffusers bf16 + a conversion simultaneously.
- If the Mac you're on is NOT 36 GB (check `sysctl hw.memsize` first), adapt:
  int4 only (10.1 GiB) on 24 GB; on < 24 GB, report "not runnable" instead of
  forcing it.