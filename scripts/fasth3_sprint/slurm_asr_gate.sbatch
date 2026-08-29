#!/usr/bin/env bash
#SBATCH --job-name=h3-asr-gate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
RUN_ID="h0-vsa-sync-asr-review"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
WAV="${SPRINT_ROOT}/eval/h0-vsa-sync-gate/00_speech_exact_presenter.wav"
OUTPUT="${SPRINT_ROOT}/eval/h0-vsa-sync-gate/asr_review.json"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1
echo "$(date -Is) run=${RUN_ID} job=${SLURM_JOB_ID} wav=${WAV}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_DIR}
    export WANDB_CACHE_DIR=${SPRINT_ROOT}/caches/wandb
    export PYTHONPATH=${REPO_ROOT}:${SPRINT_ROOT}/python-packages
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/fasth3_sprint/verify_speech_asr.py \
      --wav '${WAV}' \
      --expected 'Fast video and clear audio arrive together.' \
      --output '${OUTPUT}' \
      --run-id '${RUN_ID}'
  "

echo "$(date -Is) run=${RUN_ID} complete manifest=${OUTPUT}"
