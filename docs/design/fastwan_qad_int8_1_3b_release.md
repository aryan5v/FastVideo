# FastWan-QAD-INT8-1.3B Apple Release Record

This is the committed launch-evidence and publication checklist. Do not commit
generated videos or visual grids.

## Reproducible evidence

| Field | Recorded result |
| --- | --- |
| Hardware | Apple M4 Max, 36 GB-class unified memory (MLX: 38.65 GB) |
| Runtime | macOS 14+, Python 3.12, MLX 0.31.2 |
| Shape | 480x832, 81 frames, 3-step DMD (`1000,757,522`) |
| Path | MPS prompt encode, MLX INT8 DiT, TAEHV decode |
| Total / denoise / peak | 123.7 s / 117.6 s / 5.63 GiB |

QAD v2 raw and EMA measure 0.9360 and 0.9331 mean MS-SSIM against each
model's own FP16 output. This is quantization consistency, not absolute quality
and not a checkpoint-selection metric.

The recorded evidence above predates the release-default changes (compiled
DiT forward, bf16 prompt encode, TAEHV default, INT8 default). Re-run the
recorded command with current defaults and refresh this table before
publication.

## Checkpoint selection

2026-07-10 — the release owner selected **EMA** as the official checkpoint.
The final motion7 visual review (gate 3 below) confirms this selection
before anything is published; if the review contradicts it, the selection is
re-opened, not overridden.

## Mandatory pre-publication gate

1. Fresh source install with `uv pip install -e '.[mlx]'`; retain passing
   Metal and MLX CPU suite output.
2. Run `motion7` at the recorded shape; validate all MP4s, timings, memory,
   Diffusers-to-MLX conversion, and cold pre-quantized-checkpoint reload.
   Include stock-FastWan FP16 and stock INT8-PTQ columns via `--reference`
   so the review judges absolute quality and motion, not only consistency.
3. The release owner's EMA selection must be confirmed by the final motion7
   visual review; do not publish model or blog first.
4. Do not claim 16 GB support without a separate pass on a physical 16 GB Mac.

## Model-card and blog requirements

The model card must identify the chosen raw/EMA checkpoint, base provenance,
exact revision, model artifacts, SHA-256 checksums, fixed generation command,
hardware/software evidence, intended T2V-only use, and limitations. It must
include FastVideo's Apache-2.0 notice and TAEHV's MIT notice.

The launch blog may quote the M4 Max measurement and may name the EMA
checkpoint as the release, contingent on the visual-review confirmation in
gate 3. It must not claim physical 16 GB support, use the invalid 0.9860
run-1 EMA score, call fake-quant parity bitwise, or claim everything is
Apache-2.0.
