#!/bin/bash

#SBATCH --job-name=h3-native-v3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=hpc-rack-3
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-native-v3-%j.out

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
BASE_ROOT="${BASE_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3}"
MIN_RESOLUTION_COUNT="${MIN_RESOLUTION_COUNT:-10}"

cd "$REPO"
if [ ! -f "$OUTPUT_ROOT/FROZEN_MANIFEST.json" ]; then
  "$VENV/bin/python" "$HARNESS/derive_filtered_dataset.py" \
    --base-root "$BASE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --min-resolution-count "$MIN_RESOLUTION_COUNT"
fi
if [ ! -f "$OUTPUT_ROOT/READY.json" ]; then
  "$VENV/bin/python" "$HARNESS/finalize_dataset.py" --root "$OUTPUT_ROOT"
fi
"$VENV/bin/python" "$HARNESS/derive_filtered_dataset.py" \
  --base-root "$BASE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --min-resolution-count "$MIN_RESOLUTION_COUNT" \
  --verify-only
