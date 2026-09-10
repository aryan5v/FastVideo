#!/bin/bash
#SBATCH --job-name=h3-four-way-bench
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
# Fair four-way comparison on five completely new prompts (verified absent from
# the 58k index and all prior panels): Base H3 50-call, 42-block all-call
# (job-6929 export-200), 42-block DMD2 4-call (job-7903 export-500), and the
# released FastH3 V1 4-call. Same prompts, same seed, same geometry, same
# SDPA backend and bf16 precision; each model uses its native step schedule.
# Per-clip wall_seconds: clip 00 is the cold generation right after model load,
# clips 01-04 are warm; peak_memory_mb comes from the generator receipt.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/four-way-bench/job-${SLURM_JOB_ID}"
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
run_one base-h3-50call "${HF}/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe" 50
run_one 42block-allcall-6929 "${SPRINT_ROOT}/runs/base42-prompt58k-recovery/job-6929/export-200" 50
run_one 42block-dmd4-7903 "${SPRINT_ROOT}/runs/best42-dmd4-resume/job-7903/export-500" 5
run_one fasth3-v1-4call "${HF}/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558" 5
date -Is > "${OUTPUT_ROOT}/completed_at.txt"
'
