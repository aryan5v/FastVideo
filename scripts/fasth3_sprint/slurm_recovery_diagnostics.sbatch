#!/usr/bin/env bash
#SBATCH --job-name=h3-recovery-diag
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
export SPRINT_ROOT
REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_ROOT="${SPRINT_ROOT}/logs/slurm"
mkdir -p "${LOG_ROOT}"

echo "$(date -Is) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /bin/bash "${REPO_ROOT}/scripts/fasth3_sprint/run_recovery_diagnostics.sh"
