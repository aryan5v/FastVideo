#!/usr/bin/env bash
#SBATCH --job-name=h3-block-score
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${JOB_INDEX:?JOB_INDEX is required}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
: "${NUM_JOBS:=4}"
: "${NPROC_PER_NODE:=4}"
: "${MAX_RECORDS_PER_SHARD:=0}"
: "${MAX_BLOCKS:=0}"
: "${ALLOW_INCOMPLETE_CALIBRATION:=0}"

EXTRA_ARGS=()
if [[ "${ALLOW_INCOMPLETE_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-incomplete-calibration)
fi

REPO_ROOT="${SPRINT_ROOT}/repo"
MODEL_ROOT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
CORPUS_ROOT="/mnt/nfs/vlm-aryan/fasth3-33b-20260806/data/h3_corpus"
SUPPLEMENT_ROOT="${SPRINT_ROOT}/data/h6-multishot-calibration"
OUTPUT_DIR="${OUTPUT_DIR:-${SPRINT_ROOT}/eval/h6-block-score}"
LOG_DIR="${SPRINT_ROOT}/logs/h6-block-score-job-${JOB_INDEX}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/block-score-${JOB_INDEX}"

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
    export TRITON_CACHE_DIR=${SPRINT_ROOT}/caches/triton/block-score-${JOB_INDEX}
    export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
    export FASTVIDEO_MINIMAX_H3_FUSIONS=0
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node='${NPROC_PER_NODE}' \
      scripts/fasth3_sprint/score_minimax_h3_blocks.py \
      --model-root '${MODEL_ROOT}' \
      --source-revision f624f08c6c279ab43534c003e556fc5b295b6558 \
      --source-commit '${SOURCE_COMMIT}' \
      --corpus-root '${CORPUS_ROOT}' \
      --supplement-root '${SUPPLEMENT_ROOT}' \
      --output-dir '${OUTPUT_DIR}' \
      --job-index '${JOB_INDEX}' \
      --num-jobs '${NUM_JOBS}' \
      --max-records-per-shard '${MAX_RECORDS_PER_SHARD}' \
      --max-blocks '${MAX_BLOCKS}' \
      ${EXTRA_ARGS[*]}
  "
