# FastH3 34-block release-candidate plan (rank-16 AdaLN target)

Status: plan of record, 2026-09-09. Supersedes the 22-block full-rank "14B"
sizing in `fasth3_roadmap.md` §0 for the consumer track: with AdaLN rank-16
factorization the 34-block map is **13.9B parameters** (~13.1 GB int8 resident,
~6.6 GB nvfp4 resident) and keeps 68% of Base-H3 depth instead of 44%.

## Why 34 blocks + rank-16

- AdaLN modulation is 13.0B of the 33.0B parent (39%) and is reachable only on
  the 1-D timestep curve, so an SVD basis at rank 16 removes ~12.9B with
  measured near-zero induced modulation error (`convert_minimax_h3_adaln_rank.py
  --report-only`; job 6814 screened ranks 16/32).
- The user's rank-16 generation from the job-6972 export showed no visible
  quality loss, confirming the factorization is safe on a pruned student.
- 34 blocks is the deepest map that fits the consumer resident-memory brief
  with int8 on 24 GB Macs and nvfp4 on 16 GB VRAM cards.

## Stage order (each stage gated; no stage starts on an ungated parent)

1. **Map + recovery** (running: jobs 7034-7038). Constrained re-cut maps
   (`select_h3_block_map.py`: audio veto, no gap stretching, >=4 late-half
   removals) + detail-band recovery (`low_sigma_interval_fraction 0.5`,
   `audio_velocity_weight 4`, `audio_seam_weight 2`, real-data anchor or
   prompt-only 58k per arm). Arms isolate: data anchor (A/C), exposure (A/B),
   map (7036 vs 7037/7038), lineage (7037 Base vs 7038 recovered-42).
2. **Gate** (CPU, `audio_fidelity_gate.py` + `verify_speech_asr.py`, reference
   `diagnostics/base-h3-controls/6871-TORCH_SDPA`, all media decoded
   TORCH_SDPA): speech WER 0; voicing periodicity >= base; hf4 ratio >= base;
   silence delta <= 0.15; video motion >= 0.7x base; contrast >= 0.85x base;
   Laplacian variance >= 0.8x base. The exact-speech WER gate alone is
   insufficient: Whisper passes muffled audio (jobs 6950/6974).
3. **Rank-16 conversion** of the gated winner: re-fit the basis on the
   *deployed* sigma grid, per-modality slice error report (audio rows
   64512:96767 of each `adaln_proj.linear`), then the five-prompt panel again.
   Rank-32 fallback costs 0.1B if the audio slice error is the largest.
4. **DMD2 / PDD** on the rank-16-equivalent full-rank weights (distill before
   quantizing; never fold pruning into step distillation). The joint DMD2
   adapter now has a scale-relative denominator floor and optional grad
   sanitization (jobs 6951/6953 failed on the absolute 1e-6 floor). Four-call
   grid, Base H3 frozen score teacher, learned critic; PDD only if DMD2 misses
   the motion gate.
5. **QAT then QAD** per deploy grid (int8-MLX affine group 64, nvfp4-CUDA),
   keeping modulation/timestep-embed/patch-embed/output-heads/norm affines and
   the FP16 rank-16 AdaLN island out of the quantized grid. QAD's critic term
   is the only quantization-stage component that can add quality; it is polish,
   not repair.

## Audio failure model (why the gates above)

Recovery KD under the shift-12/shift-3 grid trains ~2/49 video and ~4/49 audio
intervals below sigma 0.2, so L2 regression removes exactly the low-noise
detail: spectral brightness, transient kurtosis and voicing periodicity decay
monotonically with update count (42-block: step0 > 500 > 1000 > +200; 34-block:
6972 at 200 updates beats 6950 at 500 from the same initializer). The
detail-band sampler and audio weights attack that mechanism directly; the
fidelity gate measures it instead of trusting WER or energy-normalized KD loss,
which both stayed flat while audio muffled.

## Compute ledger

- Detail-band arm: 300 updates x ~35 s + export + panel + gates ~= 4.5 h on 4 GPUs.
- Re-cut chain: extraction ~20 min CPU-held + step-0 panel ~15 min + 200-update
  recovery ~3 h + panel/gates ~= 4.5 h on 4 GPUs.
- Rank-16 conversion + panel: < 1 h CPU + 15 min GPU.
- DMD2 pilot: 500 updates ~4 h; QAT/QAD adaptations: hours each.
