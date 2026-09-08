#!/bin/bash
#SBATCH --job-name=h3-threeway-five
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --array=0-2
#SBATCH --time=02:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export PANEL_ROOT="$SPRINT_ROOT/videos/threeway-five/job-${SLURM_ARRAY_JOB_ID}"
case "${SLURM_ARRAY_TASK_ID}" in
 0) export ROLE=base-h3; export POINTS=50; export MODEL=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe ;;
 1) export ROLE=fasth3-dense4; export POINTS=5; export MODEL=/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558 ;;
 2) export ROLE=uniform42-step500; export POINTS=50; export MODEL="$SPRINT_ROOT/preserved/uniform42-step500-job6924/export-500" ;;
 *) exit 2 ;;
esac
test -s "$CODE_ROOT/CODE_COMMIT"
test -s "$MODEL/transformer/config.json"
export OUTPUT_ROOT="$PANEL_ROOT/$ROLE"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
# One array task owns the review page; all media URLs are relative to the panel.
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
 cp "$CODE_ROOT/scripts/fasth3_sprint/threeway_five_review.html" "$PANEL_ROOT/index.html"
fi
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="$CODE_ROOT" bash -lc '
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
 --output-dir "$OUTPUT_ROOT" --run-id "threeway-five-${ROLE}-${SLURM_ARRAY_JOB_ID}" \
 --source-commit "$(cat "$CODE_ROOT/CODE_COMMIT")" --max-prompts 5 \
 --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps "$POINTS" --num-gpus 4 \
 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "$OUTPUT_ROOT/completed_at.txt"
'
