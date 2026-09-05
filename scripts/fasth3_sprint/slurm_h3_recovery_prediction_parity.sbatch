#!/usr/bin/env bash
#SBATCH --job-name=h3-export-parity
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
DIAGNOSTICS="${SPRINT_ROOT}/diagnostics/recovery-failure"
LOG_DIR="${SPRINT_ROOT}/logs/h36-recovery-export-parity"
mkdir -p "${DIAGNOSTICS}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/h36-recovery-export-parity"
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export WANDB_DIR="${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/h36-recovery-export-parity"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"

for source_kind in dense vsa; do
  checkpoint="${SPRINT_ROOT}/runs/h36-recovery/${source_kind}-activation/checkpoint-200"
  export_dir="${SPRINT_ROOT}/exports/h36-recovery-comparison/${source_kind}-activation-step200-raw"
  run_id="h36-export-parity-${source_kind}-step200-v1"
  echo "$(date -Is) branch=${source_kind} checkpoint=${checkpoint} export=${export_dir}"
  srun \
    --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
    --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
    --container-workdir="${REPO_ROOT}" \
    --export=ALL \
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      "${REPO_ROOT}/scripts/fasth3_sprint/verify_recovery_export_prediction_parity.py" \
        --checkpoint "${checkpoint}" \
        --export "${export_dir}" \
        --receipt "${DIAGNOSTICS}/${source_kind}-activation-step200-prediction-parity.json" \
        --wandb-project fasth3-14b-2step-qad-sprint \
        --wandb-run-id "${run_id}"
done
