#!/bin/bash
#SBATCH --job-name=h3-34-tri-bench
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
# 34-block candidate tri-bench on the same five novel prompts / seed / geometry
# as job 7960 so the tables merge: standard full-rank 50-call (job-7048
# export-200), DMD2 4-call (job-7102 export-500), rank-16 AdaLN 50-call
# (checkpoints/base34-recut-r16-v1). Cold = clip 00, warm = clips 01-04.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/34-tri-bench/job-${SLURM_JOB_ID}"
export PROMPTS="${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/bench_five_new_prompts.json"
export SEED=20260912
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${PROMPTS}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
HF=/mnt/nfs/vlm-aryan/hf-cache/hub
run_one() {
  local role="$1" model="$2" steps="$3"
  "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
    --model-path "${model}" --checkpoint-role "${role}" \
    --attention dense --attention-backend TORCH_SDPA \
    --prompts "${PROMPTS}" --output-dir "${OUTPUT_ROOT}/${role}" \
    --run-id "four-way-${role}-${SLURM_JOB_ID}" \
    --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" \
    --height 480 --width 832 --num-frames 124 --seed "${SEED}" --steps "${steps}" \
    --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
}
run_one 34block-standard-7048 "${SPRINT_ROOT}/runs/base34-recut-detail-band/job-7048/export-200" 50
run_one 34block-dmd4-7102 "${SPRINT_ROOT}/runs/base34-recut-dmd4/job-7102/export-500" 5
run_one 34block-r16-7048 "${SPRINT_ROOT}/checkpoints/base34-recut-r16-v1" 50
date -Is > "${OUTPUT_ROOT}/completed_at.txt"
'
