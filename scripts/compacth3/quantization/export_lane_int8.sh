#!/bin/bash
# Export pre-quantized int8 DiT weights for the FastH3 DMD2 student.
# Run:  sbatch /mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/exp.sbatch /mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/export_lane_int8.sh
set -euo pipefail
SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
RUN=${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3
CKPT=${RUN}/inference/checkpoint-1400
OUT=${CKPT}/exports/int8

mkdir -p "$OUT"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${M}:${SPRINT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$M"
echo "=== exporting int8 -> $OUT ==="
"$PY" "${SPRINT}/export_quant_dit.py" \
  --lane int8 \
  --model-path "$CKPT" \
  --out "$OUT" \
  --num-gpus 1
echo "=== done: $OUT ==="
ls -la "$OUT"
