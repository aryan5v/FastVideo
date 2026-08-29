#!/usr/bin/env bash
#SBATCH --job-name=fasth3-baseline
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CHECKPOINT_ROLE:?CHECKPOINT_ROLE is required}"
: "${ATTENTION:?ATTENTION is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${MAX_PROMPTS:=12}"
: "${UPLOAD_VIDEOS:=0}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
OUTPUT_DIR="${SPRINT_ROOT}/videos/${RUN_ID}"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/quick_gate_prompts.json"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${SPRINT_ROOT}/caches/wandb" "${SPRINT_ROOT}/caches/triton"

exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1
echo "$(date -Is) run=${RUN_ID} job=${SLURM_JOB_ID} commit=${SOURCE_COMMIT}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
    export HF_XET_CACHE=${SPRINT_ROOT}/caches/hf-xet
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_DIR}
    export WANDB_CACHE_DIR=${SPRINT_ROOT}/caches/wandb
    export PYTHONPATH=${REPO_ROOT}:${SPRINT_ROOT}/python-packages
    export PYTHONDONTWRITEBYTECODE=1
    export TOKENIZERS_PARALLELISM=false
    export TRITON_CACHE_DIR=${SPRINT_ROOT}/caches/triton/${RUN_ID}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p \"\${TRITON_CACHE_DIR}\" \"\${HF_XET_CACHE}\"
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/fasth3_sprint/run_baseline_matrix.py \
      --model-path '${MODEL_PATH}' \
      --checkpoint-role '${CHECKPOINT_ROLE}' \
      --attention '${ATTENTION}' \
      --prompts '${PROMPTS}' \
      --output-dir '${OUTPUT_DIR}' \
      --run-id '${RUN_ID}' \
      --source-commit '${SOURCE_COMMIT}' \
      --max-prompts '${MAX_PROMPTS}' \
      --profile strict \
      --no-fa4 \
      --no-compile \
      $(if [[ "${UPLOAD_VIDEOS}" == "1" ]]; then echo --upload-videos; else echo --no-upload-videos; fi)
  "

echo "$(date -Is) run=${RUN_ID} complete manifest=${OUTPUT_DIR}/run_manifest.json"
