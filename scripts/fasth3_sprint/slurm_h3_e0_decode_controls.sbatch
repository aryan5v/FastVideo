#!/usr/bin/env bash
#SBATCH --job-name=h3-e0-controls
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
if [[ -z "${CODE_ROOT:-}" ]]; then
  CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export CODE_ROOT
test -s "${CODE_ROOT}/CODE_COMMIT"

PARENT="${SPRINT_ROOT}/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
CONTROL48="${SPRINT_ROOT}/checkpoints/h18-candidates/dense-uniform-48b"
PROMPTS="${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json"
AUDIT_ROOT="${SPRINT_ROOT}/diagnostics/e0-decode-controls/${SLURM_JOB_ID}"
test -s "${PARENT}/transformer/model.safetensors.index.json"
test -s "${CONTROL48}/transformer/candidate_validation_receipt.json"
test ! -e "${AUDIT_ROOT}"
mkdir -p "${AUDIT_ROOT}"
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
for ITEM in parent control48; do
  if [[ "${ITEM}" == parent ]]; then MODEL="${PARENT}"; ROLE=unpruned-fast-h3-v1; else MODEL="${CONTROL48}"; ROLE=dense-uniform-48b-step0; fi
  OUT="${AUDIT_ROOT}/${ITEM}-sdpa"
  "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
    --model-path "${MODEL}" --checkpoint-role "${ROLE}" --attention dense --attention-backend TORCH_SDPA \
    --prompts "${PROMPTS}" --output-dir "${OUT}" --run-id "e0-${ITEM}-sdpa-${SLURM_JOB_ID}" \
    --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 \
    --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 \
    --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
done
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
