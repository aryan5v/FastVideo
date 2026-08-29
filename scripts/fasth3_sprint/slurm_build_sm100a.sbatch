#!/usr/bin/env bash
#SBATCH --job-name=h3-sm100a-build
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/kernel-build"
mkdir -p "${LOG_DIR}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    export SPRINT_ROOT='${SPRINT_ROOT}'
    export WHEEL_DIR='${SPRINT_ROOT}/artifacts/kernel-wheels'
    export RECEIPT_PATH='${SPRINT_ROOT}/manifests/sm100a_kernel_receipt.json'
    export SOURCE_COMMIT='$(git -C "${REPO_ROOT}" rev-parse HEAD)'
    export PYTHONDONTWRITEBYTECODE=1
    export TORCH_EXTENSIONS_DIR='${SPRINT_ROOT}/caches/torch-extensions'
    mkdir -p \"\${TORCH_EXTENSIONS_DIR}\"
    bash scripts/fasth3_sprint/build_sm100a_kernel.sh
  "
