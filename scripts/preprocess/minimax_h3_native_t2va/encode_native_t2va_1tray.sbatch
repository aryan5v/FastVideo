#!/bin/bash
# One GB200 tray runs four independent native-shape H3 preprocessing workers.
# Do not submit production workers until a --probe-only job has passed.

#SBATCH --job-name=h3-native-enc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/logs/h3-native-enc-%j.out

set -uo pipefail

export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
MODEL="${MODEL:-/mnt/lustre/vlm-k1kong/models/MiniMax-H3}"
NUM_WORKERS="${NUM_WORKERS:-4}"

WORKLIST="${WORKLIST:-${1:-}}"
if [ -z "$WORKLIST" ]; then
  echo "WORKLIST is required as the first positional argument" >&2
  exit 2
fi

mkdir -p /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/logs
cd "$REPO"
pids=()
for gpu in $(seq 0 $((NUM_WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29650 + gpu)) \
    "$VENV/bin/python" "$HARNESS/encode_worker.py" \
      --worklist "$WORKLIST" \
      --model-path "$MODEL" \
      --worker-tag "${SLURM_JOB_ID}_$(hostname)_g${gpu}" \
      --linger-minutes "${LINGER_MINUTES:-60}" \
      ${LIMIT_CHUNKS:+--limit-chunks "$LIMIT_CHUNKS"} \
      ${TIMING:+--timing} \
      > "/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/logs/h3-native-enc-${SLURM_JOB_ID}-g${gpu}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
