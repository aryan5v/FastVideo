#!/bin/bash
#SBATCH --job-name=h3-backend-parity
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export CODE_ROOT="${PWD:?batch working directory is required}"
export MODEL=${SPRINT_ROOT}/checkpoints/h18-candidates/dense-uniform-48b
export SOURCE_CHECKPOINT=${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200
export AUDIT_ROOT=${SPRINT_ROOT}/diagnostics/backend-parity/job-${SLURM_JOB_ID}
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${MODEL}/transformer/candidate_validation_receipt.json"
test -s "${SOURCE_CHECKPOINT}/metadata.json"
test ! -e "${AUDIT_ROOT}"
mkdir -p "${AUDIT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
CUDA_VISIBLE_DEVICES=0 "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/compare_h3_attention_backends.py" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --model-path "${MODEL}" \
  --output "${AUDIT_ROOT}/parity.json" --min-cosine 0.999
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
