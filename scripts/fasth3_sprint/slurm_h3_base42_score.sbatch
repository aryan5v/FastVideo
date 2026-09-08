#!/bin/bash
#SBATCH --job-name=h3-base-controls
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
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
AUDIT_ROOT="${SPRINT_ROOT}/diagnostics/base42-score-screen/${SLURM_JOB_ID}-${ATTENTION_BACKEND}"
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
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 \
 "${CODE_ROOT}/scripts/fasth3_sprint/score_minimax_h3_blocks.py" \
 --model-root "${PARENT}" --source-revision 42ed227ee7df40d41602854ae760620d6eb651fe \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" \
 --corpus-root "${SPRINT_ROOT}/data/day1-mask-split/train" \
 --output-dir "${AUDIT_ROOT}/partials" --job-index 0 --num-jobs 1 \
 --max-records-per-shard 12 --allow-incomplete-calibration
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/aggregate_minimax_h3_block_scores.py" \
 --partials "${AUDIT_ROOT}/partials" --output-dir "${AUDIT_ROOT}/maps" \
 --expected-shards 4 --expected-records 48 --allow-incomplete-categories --keep-blocks 42 \
 --run-id "base42-score-screen-${SLURM_JOB_ID}"

date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
