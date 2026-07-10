# Experiment: Flag-gated windowed self-attention in MLX Wan DiT (end-to-end)

**Date:** 2026-07-10  
**Machine:** Apple Silicon (M4 Max class)  
**MLX:** 0.31.2  
**Branch:** `aryan/mac-m5-window-integ`  
**Code:** `fastvideo/mlx_runtime/fastwan.py` (self-attn gate) + `fastvideo/benchmarks/eval_windowed.py`  
**Artifacts:** `bench/window_eval/summary.json`, `bench/window_eval/fox_*.mp4`

## Motivation

Prior microbench (`docs/experiments/m5-attention.md`) showed chunked symmetric 1D
windowed attention is **7–21×** faster than full SDPA at `S=32760`, but mean cosine
vs full on *random* Q/K/V was low (0.22–0.37). Real DiT activations have spatial/
temporal locality, so quality might still be fine — **only a full denoise on the
distilled model can tell**. That is this experiment.

## Method

- **Gate:** `FASTVIDEO_MLX_WINDOW` (int; `0`/unset = full SDPA, byte-identical path).
  Optional `FASTVIDEO_MLX_WINDOW_SINK` (default 0). Applied to **self-attention only**
  in `MLXWanTransformerBlock`; cross-attention to text stays full.
- **Model:** run-2 QAD distilled checkpoint `/Users/aryank/models/qad_int8_v2`, **int8**
  weights, **TAEHV** decode, 3-step DMD (`1000,757,522`), seed **1024**.
- **Prompt / shape:** fox-forest, `480×832×81`.
- **Windows:** `[0, 4096, 2048, 1024]` (0 = full = SSIM reference).
- **Metrics:** `denoise_steady_step_s` (median of steps after first), peak GiB, MS-SSIM
  of each windowed mp4 vs the `window=0` mp4 (pytorch-msssim via bench helper).

Flag-off sanity (separate smoke): `FASTVIDEO_MLX_WINDOW` unset, fp16 + TAEHV fox →
`ms_ssim_vs_ref = 1.0` (self-ref cell), steady ≈ 29.2 s/step. Confirms the gate does
not alter the historical path when unset.

## Results

| window | s/step | speedup vs full | peak GiB | ms_ssim_vs_full |
|--------|--------|-----------------|----------|-----------------|
| 0 (full) | 31.321 | 1.00× | 4.330 | 1.0000 |
| 4096 | 20.831 | 1.50× | 4.330 | **0.2270** |
| 2048 | 22.772 | 1.38× | 4.330 | **0.2189** |
| 1024 | 21.862 | 1.43× | 4.330 | **0.2218** |

Source: `bench/window_eval/summary.json`. Videos kept under `bench/window_eval/`.

### Eyeball notes

Mid-frame stills (and the full clips) tell the same story as MS-SSIM. **Full attention**
produces a coherent orange fox mid-stride in a misty green pine forest — the prompt is
actually satisfied. **All three windowed settings** produce abstract, psychedelic
color fields (swirling teal / orange / purple bands, checker-like texture, no
recognizable animal or trees). They do not look like “slightly softer foxes”; they look
like **failed densoise / mush**. W=4096 is no closer to a fox than W=1024 by eye, which
matches the flat ~0.22 SSIM band. Mean absolute pixel error vs full is ~39–41 on 0–255.

### Speed note

End-to-end denoise only gains **~1.4–1.5×**, far below the 7–21× microbench for a single
attention call. Peak memory is unchanged (~4.3 GiB). Likely causes: Python chunk loop
overhead across ~30 layers, smaller SDPA kernels under-utilizing Metal, and remaining
non-attention work. Even if quality were perfect, this 1D path is not yet a shipping
speed win without more kernel-side work.

## Recommendation

**Do not ship 1D windowed self-attention.** Quality collapses (MS-SSIM ≈ 0.22; videos
are mush, not foxes). The receptive field of a 1D window on a **flattened
(frame × H × W)** token grid is the likely root cause: neighboring tokens in sequence
are not a coherent 3D neighborhood, so local attention drops the long-range spatial
and temporal mixing the DiT needs.

**Follow-up (keep the idea, fix the window geometry):** implement a **3D-aware local
window** (per-token neighborhood in latent frame/height/width, optional temporal
radius + small global sinks) and re-run this same eval harness. Until that lands,
leave the flag at default full attention; the gate stays for experiments only.

## How to reproduce

```bash
source /Users/aryank/claude-fastvideo/FastVideo/.venv/bin/activate
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD
# flag-off sanity
unset FASTVIDEO_MLX_WINDOW
python -m fastvideo.benchmarks.mlx_fastwan_bench \
  --model-root /Users/aryank/models/qad_int8_v2 \
  --prompt "A fox runs through a misty pine forest, leaves kicking up behind it." \
  --modes fp16 --decoders taehv --height 480 --width 832 --num-frames 81 --seed 1024
# full window sweep
python -m fastvideo.benchmarks.eval_windowed \
  --model-root /Users/aryank/models/qad_int8_v2 \
  --output-dir bench/window_eval
```
