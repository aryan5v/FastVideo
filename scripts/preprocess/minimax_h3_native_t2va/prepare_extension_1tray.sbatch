#!/bin/bash
# Freeze an immutable extension and seed exact reusable chunks on a compute tray.

#SBATCH --job-name=h3-native-prep
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-native-prep-%j.out

set -euo pipefail

export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
CONFIG="${CONFIG:-${HARNESS}/v10_sources_v2.json}"
BASE_ROOT="${BASE_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1}"
NEW_ROOT="${NEW_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2}"
EXTEND_SOURCE="${EXTEND_SOURCE:-h3_t2av_video_nuva_50k_720_mixed_len}"

cd "$REPO"

if [ ! -f "$NEW_ROOT/FROZEN_MANIFEST.json" ]; then
  "$VENV/bin/python" "$HARNESS/freeze_sources.py" \
    --config "$CONFIG" \
    --output-root "$NEW_ROOT" \
    --extend-existing "$BASE_ROOT" \
    --extend-source "$EXTEND_SOURCE"
fi
"$VENV/bin/python" "$HARNESS/freeze_sources.py" \
  --config "$CONFIG" \
  --output-root "$NEW_ROOT" \
  --verify-existing
"$VENV/bin/python" "$HARNESS/validate_heldout_media.py" \
  --manifest "$NEW_ROOT/validation/heldout64.json"
"$VENV/bin/python" "$HARNESS/schema_dry_run.py"
"$VENV/bin/python" "$HARNESS/seed_encoded_chunks.py" \
  --base-root "$BASE_ROOT" \
  --new-root "$NEW_ROOT"
