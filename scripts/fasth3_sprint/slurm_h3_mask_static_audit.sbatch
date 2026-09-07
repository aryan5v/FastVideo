#!/usr/bin/env bash
#SBATCH --job-name=h3-mask-static-audit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
: "${SPRINT_ROOT:?}"
: "${CODE_ROOT:?Immutable snapshot required}"
: "${CHECKPOINT:?}"
: "${FULL_EXPORT:?}"
: "${COMPACT_EXPORT:?Fresh destination required}"
: "${AUDIT_ROOT:?Fresh destination required}"
test -s "${FULL_EXPORT}/transformer/model.safetensors"
test ! -e "${COMPACT_EXPORT}"
test ! -e "${AUDIT_ROOT}"
mkdir -p "${AUDIT_ROOT}"
export SPRINT_ROOT CODE_ROOT CHECKPOINT FULL_EXPORT COMPACT_EXPORT AUDIT_ROOT
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export FASTVIDEO_MINIMAX_H3_FUSIONS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online WANDB_DIR="${AUDIT_ROOT}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/extract_h3_mask_export.py" \
  --checkpoint "${CHECKPOINT}" --full-export "${FULL_EXPORT}" --compact-export "${COMPACT_EXPORT}"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/validate_h3_candidate.py" \
  --transformer "${COMPACT_EXPORT}/transformer" --expected-source-kind dense --expected-layers 24
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/verify_recovery_export_prediction_parity.py" \
  --checkpoint "${CHECKPOINT}" --export "${COMPACT_EXPORT}" --masked-to-static \
  --receipt "${AUDIT_ROOT}/prediction-parity.json" \
  --wandb-project fasth3-14b-2step-qad-sprint --wandb-run-id "hard-static-parity-${SLURM_JOB_ID}"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "${COMPACT_EXPORT}" --checkpoint-role hard-mask-recovered-step200 \
  --attention dense --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
  --output-dir "${AUDIT_ROOT}/media-bf16" --run-id "hard-static-media-${SLURM_JOB_ID}" \
  --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 \
  --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 \
  --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
