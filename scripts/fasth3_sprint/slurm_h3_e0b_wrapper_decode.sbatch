#!/bin/bash
#SBATCH --job-name=h3-e0b-wrapper
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export CODE_ROOT="${PWD:?batch working directory is required}"
export PARENT=/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558
export SOURCE_CHECKPOINT=${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200
export VALIDATION_DATA=${SPRINT_ROOT}/data/day1-mask-split/validation
export AUDIT_ROOT=${SPRINT_ROOT}/diagnostics/e0b-wrapper-decode/${SLURM_JOB_ID}
for required in \
  "${CODE_ROOT}/CODE_COMMIT" \
  "${PARENT}/modular_model_index.json" \
  "${SOURCE_CHECKPOINT}/metadata.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "required file is missing or empty: ${required}" >&2
    exit 1
  fi
done
if [[ -e "${AUDIT_ROOT}" ]]; then
  echo "refusing to overwrite ${AUDIT_ROOT}" >&2
  exit 1
fi

srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_MINIMAX_H3_FUSIONS=0
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
export CODE_COMMIT="$(cat "${CODE_ROOT}/CODE_COMMIT")"
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${CODE_ROOT}/scripts/fasth3_sprint/run_h3_training_wrapper_rollout.py" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --model-path "${PARENT}" \
  --validation-data "${VALIDATION_DATA}" --output-dir "${AUDIT_ROOT}" --samples 2
test -s "${AUDIT_ROOT}/rollout_receipt.json"
CUDA_VISIBLE_DEVICES=0 "${PY}" \
  "${CODE_ROOT}/scripts/fasth3_sprint/decode_h3_training_wrapper_latents.py" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --model-path "${PARENT}" \
  --input-dir "${AUDIT_ROOT}"
test -s "${AUDIT_ROOT}/decode_receipt.json"
test "$(find "${AUDIT_ROOT}" -maxdepth 1 -name "sample-*.mp4" | wc -l)" -eq 2
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
