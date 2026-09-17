#!/bin/bash

#SBATCH --job-name=h3-native-final
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-native-final-%j.out

set -euo pipefail

export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
NEW_ROOT="${NEW_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2}"

cd "$REPO"
if [ ! -f "$NEW_ROOT/READY.json" ]; then
  "$VENV/bin/python" "$HARNESS/finalize_dataset.py" --root "$NEW_ROOT"
fi
"$VENV/bin/python" "$HARNESS/finalize_dataset.py" --root "$NEW_ROOT" --verify-only
