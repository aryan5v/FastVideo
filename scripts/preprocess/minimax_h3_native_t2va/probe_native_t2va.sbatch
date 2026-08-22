#!/bin/bash
#SBATCH --job-name=h3-native-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/logs/h3-native-probe-%j.out

set -euo pipefail
export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo/.venv}"
HARNESS="${HARNESS:-${REPO}/scripts/preprocess/minimax_h3_native_t2va}"
MODEL="${MODEL:-/mnt/lustre/vlm-k1kong/models/MiniMax-H3}"
WORKLIST="${WORKLIST:-${1:-}}"
: "${WORKLIST:?WORKLIST is required as the first positional argument}"
RECORD_ID="${RECORD_ID:-${2:-}}"
mkdir -p /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1/logs
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$HARNESS/encode_worker.py" \
  --worklist "$WORKLIST" \
  --model-path "$MODEL" \
  --worker-tag "probe_${SLURM_JOB_ID}_$(hostname)" \
  --probe-only \
  ${RECORD_ID:+--record-id "$RECORD_ID"} \
  --timing
