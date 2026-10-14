# FastH3 OmniRef NVFP4 QAD

This page describes how FastH3 OmniRef (Ref2VA, 8-forward PDD) is quantized to NVFP4 with FP4 sparse
attention, and how quantization-aware distillation (QAD) recovers the remaining error. There are two
stages: calibration produces the starting model, and QAD trains it against the bf16 checkpoint.

## Target numerics

The deployed student uses three formats:

- **Linears:** NVFP4 W4A4 for all DiT linears (q/k/v/out, VSA gates, FFN `fc_in`/`fc_out`).
  - FFN inputs use static, calibrated activation scales.
  - Attention projections and gates use a unit activation scale.
- **Attention:** FP4 sparse attention for VSA tile 128, the SageAttention3-style kernel that runs on
  sm_120/sm_121. Two-level P quantization and K smoothing are on. K smoothing is exact in bf16, because
  subtracting the mean key shifts every score row by a constant.
- **Kept in bf16:**
  - blocks chosen by the calibration sweep (currently 42-49, the blocks with the largest audio error);
  - AdaLN/modulation, embeddings, `proj_in`/`proj_out` and the audio projections. The last group stays
    fp32, as at inference.

## Stage A: calibration (`scripts/quantization/minimax_h3_omniref/`)

1. **Collect activation maxima.** `calibrate_omniref_nvfp4.py` runs calibration rows through the
   unmodified bf16 pipeline, including all PDD forwards, reference rows, Qwen3-VL vision tokens and
   reference audio.
   - `h3_amax_collector.py` records the input amax of every FFN linear, split by token group: text,
     reference video, target video, reference audio and target audio.
   - It also writes held-out bf16 references at seed `s` and `s+1`. The `s+1` samples give the seed noise
     floor.
   - A convergence check compares the first half of the rows with the full set, with a 5% tolerance per
     layer.
2. **Choose the variant.** `sweep_omniref_nvfp4.py` scores clip rules per FFN layer by layer-output error:
   - the calibrated max;
   - per-row p99 or p90;
   - fixed fractions of the max;
   - per-element percentiles;
   - dynamic scales.

   It also attributes teacher-forced error to blocks, then picks among whole-model variants (clip rule ×
   number of bf16 blocks) by teacher-forced error against the shipped V2 NVFP4 model. `select` writes:
   - `variant.json`: the selected variant (scale overrides and bf16 linears) for
     `eval_omniref_nvfp4.py --variant`;
   - `amax-<variant>.json` (e.g. `amax-mse_bf16_top8.json`): the flat `b<block>.<module>` table, i.e. the
     calibrated `amax.json` with the variant's clipped scales. Both the converter's `--act-amax` and QAD's
     `qad.linears.amax_json` read it. A variant with dynamic scales has no static table;
   - `sweep_report.json`, whose `selected_skip_blocks` lists the variant's bf16 blocks for
     `qad.linears.skip_blocks`.
3. **Export.** `convert_minimax_h3_modelopt_nvfp4_dit.py --act-amax amax-<variant>.json` writes the packed
   NVFP4 export. FFN layers need an amax entry; attention and gate linears without one keep the unit scale.
   `--skip-blocks <blocks>` omits those blocks' linears from the packed file and records them in its
   safetensors metadata, making a mixed export; the loader follows that recorded list and keeps those blocks
   on the dense shards, so the bf16 choice carries into inference rather than being declared at run time.
   Without `--skip-blocks` the converter quantizes every block, and the bf16 blocks then take effect only in
   Gate A through `--variant` and in QAD through `skip_blocks`.
4. **Gate A.** Two checks:
   - `eval_omniref_nvfp4.py` (T1) measures teacher-forced per-forward error on held-out rows. Each DiT call
     is replayed with the NVFP4 linears on identical inputs, so trajectory divergence doesn't enter.
   - `omniref_t4_metrics.py` (T4) measures reference fidelity (DINOv2/CLIP), first/last-frame
     PSNR/LPIPS and an aesthetic score, against the bf16 seed noise floor. It also writes side-by-side
     MP4s.

In practice plain max calibration does not converge on about 27 FFN layers, almost all `fc_out`. A single row sets the
max, because of "massive activation" text tokens. Clipping rules barely change audio error. Keeping the
audio-heavy late blocks in bf16 is what moves it. The sweep selected MSE-optimal clips with blocks 42-49 in
bf16.

## Stage B: QAD (`fastvideo/train/`)

QAD runs in the modular trainer:

- **Model plugin:** `fastvideo/train/models/minimax_h3/omniref_pdd.py`.
- **Method:** `fastvideo/train/methods/knowledge_distillation/pdd_qad.py`.
- **Config:** `examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml`.

### Starting point

The student starts exactly at the deployed calibrated model. Before step 1 the run checks this and refuses to
train if any check fails:

- **Bit identity:** the student uses the teacher's own weights. With quantization disabled, it must equal the
  bf16 teacher bit for bit (`init/noquant_vs_teacher_max_abs == 0`).
- **Calibrated quantization:**
  - NVFP4 linears come from `fastvideo/layers/quantization/nvfp4_qad.py`. Their training forward is
    bit-identical to the deployed NVFP4 linear, and the backward is a full-precision straight-through
    estimator.
  - The calibration table is pinned by sha256 (`qad.linears.amax_sha256`) and copied into the run
    directory.
- **Boundary modules:** `proj_in`, the time embedder, `proj_out` and the audio projections stay fp32 and are
  frozen. They are replicated outside FSDP mixed precision, so training sees the same numerics as inference.
- **FP4 attention emulator:** `fastvideo-kernel/.../attn_qat_vsa_train.py` is a Triton block-sparse
  forward/backward that reproduces the deployed sm_12x kernel's block order and rounding.
  - `scripts/distill/minimax_h3_omniref_qad/fidelity_check.py` compares it with the real kernel on captured
    inputs (gate rel-L2 ≤ 0.01; measured 4.7e-4).
  - The slow PyTorch reference (`attn_qat_vsa_reference.py`) is the numeric spec.

`e1_stage_a_match.py` checks the whole starting point against Stage A. Per-forward error and the student's
own 8-step samples match the deployed NVFP4 export bit for bit.

### Objective

For each training row, the frozen bf16 teacher walks the 8-forward PDD ladder. At the same inputs and
timesteps, the student minimizes:

- **Output regression:** per-forward x0 error, normalized per forward and modality by its step-0 value.
- **Layer-wise reconstruction** (`pdd_qad_recon.py`): each attention and FFN sub-layer gets the teacher's
  input (stop-gradient) and matches the teacher sub-layer's output.
  - It is weighted per row group: target video 1, target audio 1, conditioning 0.25. Audio is only about
    1.4% of tokens, so without this it barely trains.
  - With `recon_normalize: step0` it is divided by its step-0 value, so `recon_lambda` balances the two
    terms. The raw reconstruction gradients are about 1000× smaller than the output term.

From `rollout_start_step`, a fraction of rows (`rollout_fraction`) uses the student's own trajectory to
reduce exposure bias. `objective: recon` alone is available as a warm-up, but on its own it moves the error
very slowly.

### Evaluation and guards

- **Teacher-forced eval** every `eval_every` steps on held-out rows across all cases and resolutions.
- **Full eval** every `full_eval_every` steps, on `max_rollout_rows` rows. It adds the student's own 8-step
  sample vs the teacher at the same seed, keyframe alignment, and saved audio and video latents.
- **Noise floor:** `noise_floor_eval` re-runs the step-0 eval at seed+1.
- **Best checkpoint:** selected by the teacher-forced score. A checkpoint only counts if audio and keyframe
  alignment stay within step-0 × (1 + max(2σ, 2%)).
- **Auto-stop:** after `patience` consecutive bad evals. The teacher-forced score is checked at every eval;
  teacher alignment (own sample and keyframes) only at full evals, and a teacher-forced-only eval leaves that
  count unchanged.
- **Audio gate:** `audio_stats_gate.py` decodes saved samples and compares spectral centroid, energy above
  8 kHz, dynamic range and loudness spread with the teacher's own seed-to-seed spread.
- **Step-0 cache:** `pdd_qad_step0.py` caches the step-0 evaluation by a fingerprint of the teacher and
  student checkpoints, calibration, quantization plan, eval set and score weights, so reruns skip it.
- **Resume:** regular checkpoints carry the step-0 baseline (loss normalizers, reconstruction normalizers,
  tolerances) and the guard state (best score, bad-eval counts) as `method.qad_baseline`, so a resumed run
  keeps its loss scale and stop history. Checkpoints without it cannot be resumed.

## Running

1. Fill in the placeholder paths in the config: checkpoint, calibration table, data manifest, eval manifest
   and output directory. Set `amax_sha256`.
2. Launch on 4 nodes × 4 GPUs (SP 4):

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml
```

Useful tools in `scripts/distill/minimax_h3_omniref_qad/`:

- `generate_omniref_latents.py`: builds held-out eval rows from precomputed Ref2VA conditioning.
- `decode_eval_samples.py`: side-by-side MP4s.
- `eval_attention_variants.py`: training-free FP4 attention modes.
- `profile_forwards.py`, `bench_components.py`: cost estimates.
- `step0_cache_from_log.py`: builds a step-0 cache from an older run's log.

## Current status

### Calibration

Ratios are teacher-forced error vs V2 NVFP4, worst over forwards, on calibration rows.

| Variant | Video | Audio |
|---|---|---|
| Plain max | 0.63× | 2.06× |
| MSE clip + 4 bf16 blocks | 0.58× | 1.83× |
| **MSE clip + 8 bf16 blocks (selected)** | **0.53×** | **1.56×** |

On held-out rows the selected variant scores 0.46-0.54× V2 on video and 1.05-1.47× on audio. Every T4
metric passes except last-frame LPIPS, which has a large seed noise floor.

### FP4 attention, no training

Two-level P plus K smoothing beats single-level P:

| | Single-level P | Two-level P + K smoothing |
|---|---|---|
| Score | 0.347 | 0.295 |
| Audio sample vs teacher | 0.59 | 0.38 |

### QAD run

4 nodes, lr 5e-6, 16 states per update, NVFP4 linears plus FP4 attention. Step 0 is the calibrated model.

| Step | Score | Video x0 | Audio x0 | Video sample vs teacher | Audio sample vs teacher | First keyframe |
|---|---|---|---|---|---|---|
| 0 | 0.327 | 0.209 | 0.445 | 0.441 | 0.442 | 0.321 |
| 30 | 0.290 | 0.189 | 0.391 | 0.428 | 0.479 | 0.265 |
| 60 | 0.283 | 0.185 | 0.380 | 0.402 | 0.454 | 0.275 |

The seed noise σ is about 0.02 for the video sample, 0.14 for the audio sample and 0.04 for the first
keyframe.

- **What has improved:** teacher-forced error on both modalities and the video sample. Both are beyond
  noise.
- **Still unresolved:** the audio sample is within noise.
- **Gains are slowing.** Most of the remaining error comes from FP4 attention. With bf16 attention, NVFP4
  linears alone are at about 0.11 video x0.

### Open questions

- Whether bf16 attention for audio and text query rows is worth its cost.
- A higher learning rate once progress plateaus.
- Adding the `full_scene_audio_image_reference` case to training and eval. It is listed in the data config,
  but the current data has no rows for it.
