#!/bin/bash
#SBATCH --job-name=h3-missing-prompts
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

export OUT=${SPRINT_ROOT}/data/user-prompts-20260906/missing-encoded-v1
mkdir -p "${SPRINT_ROOT}/logs/missing-prompts"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="$CODE_ROOT" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export WANDB_MODE=online
/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python scripts/fasth3_sprint/encode_h3_missing_prompts.py --model-root /mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe --prompts "${SPRINT_ROOT}/data/user-prompts-20260906/train.jsonl" --receipt "${SPRINT_ROOT}/data/user-prompts-20260906/index-v1/receipt.json" --output "$OUT"
'
