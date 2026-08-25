# FastH3 Release Adoption + MLX Focus Plan

Author: fork workstream (`fasth3-mlx-runtime`), 2026-08-11
Status: plan — nothing implemented yet.

## 0. What FastVideo released (facts)

`FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` (and v0.1):
- **Full 33B** (50-layer, same H3 arch, hidden 5376/56 heads) — **NOT pruned**.
- **4-step DMD2 distillation**, **data-free**, guidance-distilled, bf16.
- Diffusers modular layout; only `transformer/` differs from the base release;
  video/audio VAEs, Qwen3-VL-32B encoder, schedulers are unmodified H3 copies.
- Student was trained with **block-sparse VSA** (64-token tiles, 90% sparsity)
  and carries **trained sparse-gate weights** (`attn.to_gate_compress`, 50 keys);
  runs **dense by default**.
- **Trained ladder: `[999, 749, 500, 250]`** on a shared 1000-step grid, each
  scheduler applying its own shift (video 12 / audio 3).
- **v0.2 is step 2900 of a 4000-step run** — an *intermediate preview*, quality
  still maturing (most visibly on high-motion detail).

Upstream code landed alongside (to rebase in):
- `examples/inference/basic/basic_fasth3.py` (#1731): the few-step example +
  the exact ladder→sigma mapping (the reference for our MLX scheduler support)
- VSA-H3 inference backend (#1731), 64-token-tile routing (#1745)
- **AdaLN rank-reduced pruner** (#1699, −39% params / −23 GiB VRAM) +
  `scripts/checkpoint_conversion/minimax_h3_adaln_prune.py` (#1712)
- Qwen3-VL truncated encoder build (#1711, −13.7 GB), text-encoder memory (#1732)
- VAE decode parallel + peak-memory fixes (#1744, #1734, #1703), FA4 packed-varlen
  (#1742), fused NVLink all-to-all (#1740), Sol-Engine fusions (#1735)

## 1. What we can take (strategic)

| Asset | Take for the program |
|---|---|
| **4-step 33B student** | **Adopt as the 33B flagship** — replaces our planned 33B DMD run from scratch. Only *adaptation* runs needed (nvfp4 CUDA, int8 MLX), not full training. |
| **Trained ladder [999,749,500,250]** | Confirmed 4-step operating point; port to **both** CUDA and MLX schedulers (and reuse for 14B/8B QAD later if it holds). |
| **Data-free DMD2** (upstream ran it without a corpus) | De-prioritizes corpus for the 33B step-distill; **corpus/captions/synthetic still matter for the small students** (capacity reduction + feature distill). |
| **VSA sparse gates** (trained in the weights) | CUDA speed lever (dense is default; VSA opt-in). MLX: dense ignores the gates; a block-sparse MLX path is a later question (parked). |
| **AdaLN rank-reduced pruner** (−39% params) | A lighter 33B target that fits **more MLX memory tiers** — pairs with our AdaLN-cache; candidate "FastH3-Lite" for 24–36 GB Macs. |
| **Perf commits** (VAE, text-encoder, FA4, all-to-all) | Mostly CUDA-specific; the **text-encoder trimming confirms our precompute-embeddings design**; VAE tiling lessons transfer to the MLX VAE ports. |

## 2. MLX focus plan

**Goal: first MLX H3 artifacts, then the reach ladder (64 → 24 → 16 GB Macs).**

### Phase A — FastH3-Preview 33B on MLX (1–2 weeks, 0 GPU)
1. **Fork sync**: rebase `fasth3-mlx-runtime` onto current upstream main (perf
   commits + `basic_fasth3.py` + AdaLN converter + VSA backend). Re-apply our
   MLX + training patches.
2. **Converter**: `mlx_h3_dit_from_diffusers_safetensors` already reads the
   diffusers layout; **handle `attn.to_gate_compress`** (exclude on the dense
   path — 50 keys, no MLX use) + verify the int8 build of the 33B student
   (attention/FFN only, group 64; ~35 GB int8 → **64 GB Mac flagship**).
3. **Ladder support** in `fastvideo/mlx_runtime/minimax_h3.py`: accept an
   explicit ladder (e.g. `[999, 749, 500, 250]`) and map it to the H3
   continuous convention *exactly as `basic_fasth3.py` does* (per-modality
   shift-12/3 sigma grids). Add a 4-step dual-scheduler path.
4. **Parity test**: FastH3-Preview 4-step MLX-vs-torch on the released student
   (CPU/small clips) with the ladder.

### Phase B — MLX VAE completion (the missing end-to-end pieces, 0 GPU)
- **Audio VAE decode** (BigVGAN/SnakeBeta/Kaiser, stereo = 2×mono) — the only
  genuinely new MLX component.
- **Video VAE decode** (36-layer ViT decoder, tiled, ~5.2 GB fp16) — port with
  tiling on day one.
- End-to-end example (extend `mlx_*_prompt_to_video` pattern) → **first MLX H3
  video+audio output**; reuse `--refine`, `--fast-spatial`, `--enhance-prompt`
  from PR #27.

### Phase C — artifacts & tiers
- Export **FastH3-Preview int8 `mlx_h3_dit.safetensors`** (pre-quantized,
  AdaLN-dropped) → hardware tier: 64 GB Mac = 33B 4-step flagship.
- Evaluate upstream's **AdaLN-pruned 33B** (`minimax_h3_adaln_prune.py`) as a
  second artifact (24–36 GB Mac target) — pairs with our AdaLN-cache.

### Phase D — our student ladder (the real GPU work going forward)
- **14B / 8B / 5B** students are still ours to train (the release has no small
  sizes): S1 prune → data-free-or-captioned DMD → QAT adaptations (int8 MLX /
  nvfp4 CUDA). Use the released ladder + config as the DMD reference; the 33B
  student becomes a second teacher/oracle for the small students.
- Corpus: **captions + synthetic still needed** for the small-student feature
  distill; 33B step-distill no longer depends on it.

## 3. Sequencing / what happens to current jobs

| Item | Decision |
|---|---|
| Caption preprocess (2174) | **Keep running** — captions still needed for the small students (and any future higher-quality 33B rerun). |
| Our 33B DMD relaunch | **Cancel the from-scratch plan** — adopt the release. Optionally later: a full 4000-step + captioned 33B *quality* rerun if the v0.2 preview underwhelms (the release is step 2900 preview state). |
| QAT adaptations | Now target the **released 33B**: nvfp4 (CUDA) + int8 (MLX) off the 4-step student. |
| S1 pruner + 14B/8B/5B | Unchanged scope — the primary GPU work. Add a `--from-pruned` option for staged 5B-from-14B. |
| Fork sync | Do Phase A-1 first (rebased main) before further MLX code. |

## 4. Open questions for the reviewer
1. Adopt the release as flagship vs. run our own 33B to completion (captions +
   full 4000 steps)? (Cost vs. the preview's "still maturing" caveat.)
2. Is the data-free DMD2 recipe (upstream config) the right base for the small
   students, or captioned/synthetic feature-distill first (before DMD)?
3. VSA on MLX: invest in a block-sparse MLX attention path (gates are trained),
   or keep dense + windowed-attention fallback?
