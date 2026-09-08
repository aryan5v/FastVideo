#!/bin/bash
#SBATCH --job-name=h3-base-controls
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
ATTENTION_BACKEND="${1:-TORCH_SDPA}"
case "${ATTENTION_BACKEND}" in
  TORCH_SDPA|FLASH_ATTN) ;;
  *) echo "unsupported attention backend: ${ATTENTION_BACKEND}" >&2; exit 2 ;;
esac
export ATTENTION_BACKEND
export SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
CODE_ROOT="${PWD:?batch working directory is required}"
export CODE_ROOT

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "required file is missing or empty: $1" >&2
    exit 1
  fi
}

require_file "${CODE_ROOT}/CODE_COMMIT"

PARENT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe"
CONTROL48="${SPRINT_ROOT}/checkpoints/h18-candidates/dense-uniform-48b"
PROMPTS="${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json"
AUDIT_ROOT="${SPRINT_ROOT}/diagnostics/base-same-state-parity/${SLURM_JOB_ID}-${ATTENTION_BACKEND}"
require_file "${PARENT}/modular_model_index.json"

if [[ -e "${AUDIT_ROOT}" ]]; then
  echo "refusing to overwrite existing audit root: ${AUDIT_ROOT}" >&2
  exit 1
fi
mkdir -p "${AUDIT_ROOT}"
echo "starting E0 controls from ${CODE_ROOT} into ${AUDIT_ROOT}"
export SPRINT_ROOT CODE_ROOT PARENT CONTROL48 PROMPTS AUDIT_ROOT

srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online WANDB_DIR="${AUDIT_ROOT}"
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
CUDA_VISIBLE_DEVICES=0 "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/compare_h3_attention_backends.py" \
 --source-checkpoint "${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200" \
 --model-path "${PARENT}" --output "${AUDIT_ROOT}/parity.json" --min-cosine 0.999 --grid-points 50 --same-state-flow-parity
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
