#!/usr/bin/env bash
#SBATCH --job-name=h3-recovery-compare
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_ROOT="${SPRINT_ROOT}/logs/slurm"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_ROOT}"

echo "$(date -Is) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} commit=${SOURCE_COMMIT}"
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /bin/bash "${REPO_ROOT}/scripts/fasth3_sprint/run_recovery_comparison.sh"
