#!/bin/bash
# Generate the locked five-prompt panel from one immutable exported checkpoint.
# MODEL, ROLE, and OUTPUT_ROOT are supplied with sbatch --export.
#SBATCH --job-name=h3-single-five
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
test -n "${MODEL:?}"
test -n "${ROLE:?}"
test -n "${OUTPUT_ROOT:?}"
test -s "$CODE_ROOT/CODE_COMMIT"
test -s "$MODEL/transformer/config.json"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="$CODE_ROOT" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_MINIMAX_H3_FUSIONS=0
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online WANDB_DIR="$OUTPUT_ROOT"
/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python scripts/fasth3_sprint/run_baseline_matrix.py \
  --model-path "$MODEL" --checkpoint-role "$ROLE" --attention dense --attention-backend TORCH_SDPA \
  --prompts examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json \
  --output-dir "$OUTPUT_ROOT" --run-id "single-five-${ROLE}-${SLURM_JOB_ID}" \
  --source-commit "$(cat "$CODE_ROOT/CODE_COMMIT")" --max-prompts 5 \
  --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 \
  --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "$OUTPUT_ROOT/completed_at.txt"
'
