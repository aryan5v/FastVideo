# FastH3 — Status & Handoff for Code Review

Author: FastVideo fork workstream (branch `fasth3-mlx-runtime`)
Date: 2026-08-11
Audience: senior engineer reviewing the FastH3 (MiniMax H3 consumer) program.

## 1. Where everything lives

**GitHub (fork only — nothing upstream):**
- Repo: `https://github.com/aryan5v/FastVideo`
- Branch: **`fasth3-mlx-runtime`** (all FastH3 work; ~30 commits on top of upstream main)
- Baseline: upstream `hao-ai-lab/FastVideo` main @ `fb7be2fe` (includes MiniMax H3 `#1674`, merged 2026-08-04)

**Key paths in the branch:**

| Path | What |
|---|---|
| `docs/design/fasth3_roadmap.md` | The program roadmap: 14B+8B launch, 33B flagship, QAD/QAT structure, 64-GPU plan, dataset decision |
| `fastvideo/mlx_runtime/minimax_h3.py` | **MLX H3 runtime**: float64 packing geometry, dual rectified-flow scheduler (video shift 12 / audio shift 3, data-ward velocity), DiT with AdaLN precompute cache (frees ~40% of params), affine int8 loader (attention/FFN only), pre-quantized checkpoint I/O. mxfp4/nvfp4 deliberately excluded (M5 survey). |
| `fastvideo/tests/mlx/test_mlx_minimax_h3_parity.py` + `tiny_h3.py` | Parity gate: tiny torch-vs-MLX full forward, packing vs upstream builder, scheduler grids/steps, AdaLN cache vs faithful path, int8 SNR — 7/7 green on CPU |
| `fastvideo/train/models/h3/h3.py` | **H3 training wrapper**: joint AV latents, per-modality timesteps, packed layout, data-ward velocity, corpus dataset (safetensors, dedup by prompt sha1) |
| `fastvideo/train/methods/distribution_matching/h3_dmd2.py` | **Dual-scheduler DMD2 method** (the novel piece): paired (video, audio) rollout/losses, continuous score timesteps, guidance-free teacher path |
| `fastvideo/models/loader/fsdp_load.py` | Fork patch: `FASTVIDEO_FORCE_UNIFORM_BF16` escape hatch (H3's fp32 keep-set breaks FSDP2 mixed-dtype allgather); relaxed mixed-dtype guard when no replicated params |
| `fastvideo/train/utils/moduleloader.py` | Fork patch: accept 3-element `model_index.json` format (H3's `['diffusers', cls, {kwargs}]`) |
| `scripts/fasth3/preprocess_h3_corpus.py` | Corpus preprocessor: VGGSound clips → H3 video VAE latents (posterior seed-42, normalize, patchify) + audio VAE (mode) + Qwen3-VL-32B `hidden_states[50]`; resume via skip-if-exists; distributed barrier; CUDA generators |
| `scripts/fasth3/prune_h3_depth.py` | **S1 depth pruning**: ablation sensitivity sweep (identity hooks) → keep N blocks (first/last guaranteed) → writes a diffusers-layout pruned checkpoint the training loader can init from |
| `examples/train/configs/distribution_matching/minimax_h3/dmd2_t2va_33b.yaml` | 33B DMD Quality: 16 GPUs, 2×8 HSDP, 6-step shift-12 video grid, 4000 iters, 1e-6 LR |
| `examples/train/configs/distribution_matching/minimax_h3/dmd2_t2va_33b_smoke.yaml` | Overfit smoke: 8 GPUs (2 nodes), 30 steps, 3-step grid |
| `examples/train/configs/distribution_matching/minimax_h3/dmd2_t2va_14b.yaml` | 14B DMD Quality: student inits from the S1-pruned checkpoint |
| `fastvideo/tests/modal/fasth3_gpu_verify.py` | Modal launcher (hao-ai-lab): golden gate (device-keyed), 33B T2VA smoke, parity suite |
| `fastvideo/mlx_runtime/minimax_h3.py` + `tiny_h3.py` parity | see above |

**Cluster (SLURM, `vlm-aryan@nv-vllm-slinky-login-node`):**
- Task dir: `/mnt/nfs/vlm-aryan/fasth3-33b-20260806/` — `repo/` (branch checkout), `scripts/*.sbatch` + `*_run.sh` (job launchers), `logs/`, `outputs/`, `data/`
- HF weights cached: `/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3` (~135 GB)
- Corpus: `/mnt/nfs/vlm-aryan/fasth3-33b-20260806/data/h3_corpus/` — 24,720 clips processed (extension to ~49.9k running/queued), 3,196 unique text embeddings

## 2. What was validated (evidence)

| Gate | Result |
|---|---|
| Cluster env (H3 arch/scheduler/VAEs/MLX packing in container) | PASS |
| 33B T2VA end-to-end smoke on 4×GB200 (real mp4 out) | PASS |
| Golden gate (bitwise single-block vs GB200 golden; minted for the cluster env) | PASS |
| MLX parity suite (tiny torch-vs-MLX) | 7/7 PASS (CPU) |
| Preprocess smoke (1 clip: video `(1,24,37,32,56)` + audio `(400,32)` + Qwen3-VL embed) | PASS |
| **Training overfit smoke: 30/30 steps, ~12 s/step on 8×GB200, checkpoint saved** | PASS — full DMD step: prepare_batch → rollout (no_grad loop + grad target) → DMD loss (critic+teacher) → critic flow-matching → backward → optimizer |

**Bugs found & fixed during bring-up (all committed):**
1. Per-rank RNG (`seed + global_rank`) → divergent rollout loop lengths → FSDP collective deadlock. **Fix: full loop on all ranks** (the base DMD2's "wasted" steps are load-bearing for distributed alignment).
2. H3's fp32 keep-set modules → FSDP2 mixed-dtype allgather hang. **Fix: `FASTVIDEO_FORCE_UNIFORM_BF16=1`** (bound on the model instance; precision lives in the optimizer state).
3. Grad-checkpoint recompute needs the forward context → wrap `backward` in `set_forward_context`.
4. Training loader vs 3-element `model_index.json`; H3 pipeline config resolution (base `DiTArchConfig` drops H3 fields); FSDP2 wrapper hides model attrs (read arch from pipeline config); dataset double-batch-dim; `build_row_timesteps` required args; missing `model_paths` loader arg; batch dim on patchified rows.
5. Corpus manifest CSV quoting (labels starting with `"` merged lines in `csv.DictReader` → 24.8k/50k) — rebuilt with `csv.writer` escaping (v2: 49,999 rows); preprocess resume via skip-if-exists.

## 3. What is running / queued right now

| Job | State | Notes |
|---|---|---|
| `fasth3-preproc` (2162) | PENDING | Corpus extension: processes the missing ~25,279 clips (v2 manifest) → ~49.9k total. Resume-safe. ~15 h once a node frees |
| `fasth3-33b-dmd` (2163) | PENDING | 33B DMD Quality: 16 GPUs / 4 nodes, 4000 iters, 6-step grid. EMA callback removed (FSDP2 shard-vs-shadow size mismatch; EMA optional). ~8–10 h once nodes free |
| Cluster | — | `h3-dmd2-v6` (vlm-wlsaidhi) holds 8 nodes; pool is tight; jobs schedule as nodes free |

Both jobs are single-`sbatch` relaunches (launchers in the task dir `scripts/`).

## 4. Known limitations / next steps

- **EMA callback**: incompatible with the FSDP2 DTensor params as-is (upstream issue) — removed from configs; worth fixing upstream.
- **No validation callback yet** in the training config (loss-only tracking on W&B `fasth3`); the H3 modular pipeline can drive validation once training is stable.
- **S1 recovery distillation** (feature-distill at full steps, per roadmap) is not yet a separate recipe — the first 14B run does combined capacity-recovery + step-distillation via DMD (acceptable first iteration; roadmap documents the cleaner separation).
- **MLX side** still to do: H3 sampler wiring, BigVGAN audio VAE decode, ViT video decoder port, hardware-tier bands (all scoped in `fasth3_roadmap.md`; the DiT/packing/scheduler core + parity is done).
- Corpus captions are raw VGGSound labels; Context-IR-style enrichment is a planned enhancement.

## 5. The 8B question (why it's the right call — see chat companion doc)

8B (12 blocks) is structurally identical to the 14B plan — same training stack, same configs, only `--keep-blocks 12` in the pruner. It changes the reach math substantially: int8 ~8.2 GB (resident ~4.9 GB with the AdaLN cache) fits **16 GB Macs**; nvfp4 ~4.3 GB fits **16 GB VRAM** NVIDIA cards; ~1.85× faster per step than 14B. Recommendation: make **8B the co-primary launch size** (16 GB floor) alongside 14B (24 GB floor), 33B as the flagship — the pruning/DMD pipeline is size-agnostic, so this is nearly free to add.

## 5c. Dataset strategy (2026-08-11 review) — captions + synthetic

**Problem**: raw VGGSound labels are class names ("people marching"), not
descriptions — a hard ceiling on prompt adherence that more clips cannot
raise. **Captions first, then synthetic.**

1. **Descriptive captions in the preprocessor** (`scripts/fasth3/preprocess_h3_corpus.py`
   phase B): the same Qwen3-VL-32B that produces the conditioning embeddings
   generates one detailed sentence per clip from its first/mid/last frames
   (stock `transformers`, FSDP across ranks, keyed by clip_id, resume-safe).
   Phase C embeds the captions; phase D finalizes manifests from latents +
   captions. The old raw-label text embeddings are orphaned, not reused.
2. **Synthetic corpus from the 33B teacher** (`scripts/fasth3/generate_synthetic_corpus.py`):
   DMD2 matches a distribution — it needs teacher score over prompts
   (FastWan/Wan-Syn precedent). The 50-step teacher runs a curated prompt
   set at 480p/5s → mp4s with muxed stereo audio → consumed by the
   preprocessor (latents + prompt embeddings). Exactly-matched teacher
   distribution, controlled prompt coverage, coherent AV pairing by
   construction, scales with GPU time. First slice 10–25k prompts
   (~2–6 GPU-days on 4×GB200), run between jobs.
3. **VGGSound**: the ~200k-clip source stays as filler volume, extended
   after captions land (the extension job was restarted with the caption
   pipeline; latents are reused via skip-if-exists — no double processing
   of the heavy VAE work).

## 5b. The 5B option (ultra-wide tier, fast-follow experiment)

5B ≈ 8 blocks (5.2B total, 3.1B resident non-AdaLN). int8 ~5.5 GB / resident ~3.3 GB; nvfp4 ~2.9 GB / resident ~1.7 GB → reaches **8–12 GB VRAM laptops** (the NVIDIA volume market) and gives 16 GB Macs big headroom. ~2.8× faster per step than 14B. The catch: 33B→5B is a **6.4× compression** — the steepest in the line — so the DMD loss floor is higher and the quality gate is genuinely at risk (mushy joint-AV output is the failure mode).

Recommendation: do **not** commit 5B at launch; run it as a cheap fast-follow (~1 week on 8–16 GPUs: `--keep-blocks 8` + one DMD run, config is a copy of the 14B one). If it clears the acceptable-quality bar for the tier → launch line becomes 5B/8B/14B/33B across the hardware spectrum. If not, the loss is a week, not months. Same reasoning as the roadmap's parked 4B probe, with the reach target moved from legacy 8 GB Macs to volume 8–12 GB VRAM.
