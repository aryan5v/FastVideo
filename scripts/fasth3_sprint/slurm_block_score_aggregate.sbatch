#!/usr/bin/env bash
#SBATCH --job-name=h3-score-aggregate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
KEEP_BLOCKS="${KEEP_BLOCKS:-20}"
if ! [[ "${KEEP_BLOCKS}" =~ ^(20|24)$ ]]; then
  echo "KEEP_BLOCKS must be 20 or 24" >&2
  exit 2
fi

REPO_ROOT="${SPRINT_ROOT}/repo"
RUN_ID="h6-block-score-aggregate"
if [[ "${KEEP_BLOCKS}" != 20 ]]; then
  RUN_ID="h36-block-score-aggregate-${KEEP_BLOCKS}b-v1"
fi
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export PYTHONPATH=${REPO_ROOT}:${SPRINT_ROOT}/python-packages
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_DIR}
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/fasth3_sprint/aggregate_minimax_h3_block_scores.py \
      --partials '${SPRINT_ROOT}/eval/h6-block-score' \
      --output-dir '${SPRINT_ROOT}/manifests/h6-block-score' \
      --expected-shards 16 \
      --keep-blocks '${KEEP_BLOCKS}' \
      --run-id '${RUN_ID}'
  "
