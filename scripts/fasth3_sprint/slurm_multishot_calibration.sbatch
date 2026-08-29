#!/usr/bin/env bash
#SBATCH --job-name=h3-multishot-cal
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
: "${MAX_PROMPTS:=0}"

REPO_ROOT="${SPRINT_ROOT}/repo"
MODEL_ROOT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
OUTPUT_DIR="${OUTPUT_DIR:-${SPRINT_ROOT}/data/h6-multishot-calibration}"
RUN_ID="${RUN_ID:-h6-multishot-calibration-supplement}"
LOG_DIR="${SPRINT_ROOT}/logs/h6-multishot-calibration"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/multishot-calibration"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export PYTHONPATH=${REPO_ROOT}:${SPRINT_ROOT}/python-packages
    export PYTHONDONTWRITEBYTECODE=1
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_DIR}
    export TRITON_CACHE_DIR=${SPRINT_ROOT}/caches/triton/multishot-calibration
    export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
    export FASTVIDEO_FA4=0
    export FASTVIDEO_MINIMAX_H3_FUSIONS=0
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/fasth3_sprint/build_multishot_calibration_supplement.py \
      --model-root '${MODEL_ROOT}' \
      --source-revision f624f08c6c279ab43534c003e556fc5b295b6558 \
      --source-commit '${SOURCE_COMMIT}' \
      --output-dir '${OUTPUT_DIR}' \
      --run-id '${RUN_ID}' \
      --max-prompts '${MAX_PROMPTS}'
  "
