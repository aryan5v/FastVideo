#!/bin/bash

#SBATCH --job-name=h3-native-enc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-native-enc-%j.out

set -euo pipefail

export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
MODEL="${MODEL:-/mnt/lustre/vlm-k1kong/models/MiniMax-H3}"
NUM_WORKERS="${NUM_WORKERS:-4}"

WORKLIST="${1:-${WORKLIST:-}}"
if [ -z "$WORKLIST" ]; then
  echo "WORKLIST is required as the first positional argument" >&2
  exit 2
fi
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
LINGER_MINUTES="${2:-${LINGER_MINUTES:-60}}"

mkdir -p "$LOG_ROOT"
cd "$REPO"
pids=()
for gpu in $(seq 0 $((NUM_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29650 + gpu)) \
    "$VENV/bin/python" "$HARNESS/encode_worker.py" \
      --worklist "$WORKLIST" \
      --model-path "$MODEL" \
      --worker-tag "${SLURM_JOB_ID}_$(hostname)_g${gpu}" \
      --linger-minutes "$LINGER_MINUTES" \
      ${LIMIT_CHUNKS:+--limit-chunks "$LIMIT_CHUNKS"} \
      ${TIMING:+--timing} \
      > "${LOG_ROOT}/h3-native-enc-${SLURM_JOB_ID}-g${gpu}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
