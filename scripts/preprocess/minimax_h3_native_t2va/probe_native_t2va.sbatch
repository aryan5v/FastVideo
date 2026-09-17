#!/bin/bash
#SBATCH --job-name=h3-native-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-native-probe-%j.out

set -euo pipefail
export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
MODEL="${MODEL:-/mnt/lustre/vlm-k1kong/models/MiniMax-H3}"
WORKLIST="${1:-${WORKLIST:-}}"
: "${WORKLIST:?WORKLIST is required as the first positional argument}"
case "$WORKLIST" in
  */work/worklist.json) ;;
  *)
    echo "WORKLIST must end in /work/worklist.json: $WORKLIST" >&2
    exit 2
    ;;
esac
SET_ROOT="${WORKLIST%/work/worklist.json}"
DATASET_ROOT="${SET_ROOT%/*}"
LOG_ROOT="${LOG_ROOT:-${DATASET_ROOT}/logs}"
RECORD_ID="${RECORD_ID:-${2:-}}"
mkdir -p "$LOG_ROOT"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$HARNESS/encode_worker.py" \
  --worklist "$WORKLIST" \
  --model-path "$MODEL" \
  --worker-tag "probe_${SLURM_JOB_ID}_$(hostname)" \
  --probe-only \
  ${RECORD_ID:+--record-id "$RECORD_ID"} \
  --timing
