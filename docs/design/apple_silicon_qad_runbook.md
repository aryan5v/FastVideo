# Runbook: Wan2.1-1.3B → 3-step INT8 QAD on the DGX B200

Operator instructions for launching the Mac-targeted quantization-aware DMD
distillation (roadmap M4 Phase B) on a DGX with B200 GPUs. Everything below
runs from a clone of `aryan5v/FastVideo` on branch
the approved future QAD integration branch. The training recipe is
`examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml`:
frozen Wan2.1-T2V-1.3B teacher + critic, trainable student whose linear
weights are fake-quantized every forward onto MLX's affine INT8 deploy
grid (the `mlx_qat` callback; serialized quantizer state is pinned exactly and
the Metal/PyTorch forward representation is bounded to one source-dtype epsilon by
`fastvideo/tests/mlx/test_mlx_affine_qat_parity.py`), distilled to the
3-step FastWan schedule `[1000, 757, 522]`.

Use **4 GPUs** (the recipe default) — do not grab all 8 unless told to.
Expected wall time for the full run is roughly 4–8 hours on 4×B200, but run
the smoke test first and extrapolate from its measured seconds/step.

**If only N (< 4) GPUs are free:** the run works on any count ≥ 1; the only
rule is that the HSDP shard dim must equal the GPU count. Add these overrides
to every launch command below (shown for N=3):

```bash
NUM_GPUS=3 bash examples/train/run.sh <config> \
  --training.distributed.num_gpus 3 \
  --training.distributed.hsdp_shard_dim 3 \
  ...
```

Global batch scales with GPU count (1 per GPU), which is fine for DMD at this
scale; wall time scales inversely (~6–11 h on 3 GPUs). Prefer waiting for a
4th GPU only if it frees up within the hour; otherwise just run on 3. The
same rule applies upward: if 8 are genuinely free and idle, `NUM_GPUS=8` with
`hsdp_shard_dim 8` roughly halves the wall time.

## 0. Preflight

```bash
nvidia-smi          # expect B200s; confirm >= 4 idle
python3 --version   # 3.10–3.12
```

Verify credentials are present (both are expected to be preconfigured on the
box — do not write them into the repo):

```bash
test -n "$WANDB_API_KEY" && echo "wandb ok"
hf auth whoami || huggingface-cli whoami   # HF auth for model + dataset pulls
```

## 1. Clone and install

```bash
git clone https://github.com/aryan5v/FastVideo.git && cd FastVideo
git checkout aryan/future/fastwan-qad-5b-i2v
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,mlx]"
```

B200 is sm_100: torch must be a recent CUDA build (the pinned deps are).
If any attention backend fails to import or pick a kernel at startup, force
the portable one — correctness is identical, it is only somewhat slower:

```bash
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

Sanity-check the QAT machinery on this box before spending GPU time:

```bash
pytest fastvideo/tests/mlx/test_mlx_affine_qat_parity.py -q
pytest fastvideo/tests/training/test_mlx_qat_callback.py -q
```

## 2. Dataset

```bash
python scripts/huggingface/download_hf.py \
  --repo_id "FastVideo/Wan-Syn_77x448x832_600k" \
  --local_dir "data/Wan-Syn_77x448x832_600k" \
  --repo_type "dataset"
```

This is large (order of a terabyte) — start it in `tmux` early and check free
disk first (`df -h .`). The recipe reads it from
`data/Wan-Syn_77x448x832_600k` relative to the repo root.

## 3. Smoke run (mandatory, ~30–60 min)

```bash
tmux new -s qad-smoke
NUM_GPUS=4 bash examples/train/run.sh \
  examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml \
  --training.loop.max_train_steps 100 \
  --training.checkpoint.output_dir outputs/smoke_mlx_int8 \
  --training.tracker.run_name wan2.1_qad_int8_smoke \
  --callbacks.validation.every_steps 50
```

What to verify before proceeding:

1. The log contains a line like `mlx_qat: fake-quantizing N weights (int8,
   group_size=64, ...)` with N in the hundreds — that is the QAT callback
   arming. If instead it raises `mlx_qat matched no weights`, stop and
   report.
2. No crash referencing `parametrizations` together with FSDP/HSDP/DTensor.
   The QAT callback wraps weights with torch parametrizations after model
   setup; a sharding interaction here is the one known integration risk. If
   it crashes, capture the full traceback and stop — do not work around it
   by removing the callback (that silently turns the run into plain DMD).
3. Loss values are finite, W&B run `wan2.1_qad_int8_smoke` is logging under
   project `distillation_wan`, and the step-50/100 validation clips are not
   black/NaN garbage (blurry is fine at step 100).
4. Note the steady seconds/step from the log and extrapolate: full run =
   4000 × s/step. Report that number back.

## 4. Full run

```bash
tmux new -s qad-full
NUM_GPUS=4 bash examples/train/run.sh \
  examples/train/configs/distribution_matching/wan/dmd2_t2v_mlx_int8.yaml \
  --callbacks.validation.every_steps 200
```

Output/checkpoints land in `outputs/wan2.1_dmd2_3steps_mlx_int8`
(checkpoint every 20 steps, last 3 kept), W&B run
`wan2.1_dmd2_3steps_mlx_int8`. The job is resumable:
`--training.checkpoint.resume_from_checkpoint <output_dir>/checkpoint-<step>`.
Monitor W&B; the DMD generator loss is noisy by nature — judge by the
validation clips trending sharper/more coherent, not by the loss curve alone.

## 5. Export (1 GPU) — both raw and EMA

Export twice: the raw student, and the EMA shadow weights (usually visibly
smoother). Both are evaluated on the Mac side; the better one ships.

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint outputs/wan2.1_dmd2_3steps_mlx_int8 \
  --output-dir outputs/wan2.1_qad_int8_diffusers \
  --role student

python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint outputs/wan2.1_dmd2_3steps_mlx_int8 \
  --output-dir outputs/wan2.1_qad_int8_ema_diffusers \
  --role student --ema
```

Each auto-picks the latest checkpoint and writes a Diffusers-style model dir.
If the `--ema` export errors, report the traceback and still deliver the raw
export — it unblocks Mac evaluation while the EMA path gets fixed.

## 6. Deliverables

Report back: (a) the W&B run URL, (b) measured seconds/step and total wall
time, (c) the path to `outputs/wan2.1_qad_int8_diffusers`, and (d) 2–3
validation clips from late in training. Mac-side evaluation then happens per
`docs/design/apple_silicon_fastvideo.md` (M4 exit criteria): load the export
through the MLX runtime, quantize INT8 on load (the grid the student trained
on), `--save-mlx-checkpoint`, and run the benchmark suite against the
INT8-PTQ baseline.
