#!/bin/bash
#SBATCH --job-name=h3-u42-pdd4
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export BRANCH=pdd4
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/uniform42-pdd4/job-${SLURM_JOB_ID}"
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export STUDENT="${OUTPUT_ROOT}/student-pdd32"
export BASE_GATE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"

test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"$PY" scripts/checkpoint_conversion/expand_minimax_h3_pdd.py \
 --src "${SPRINT_ROOT}/preserved/uniform42-step500-job6924/export-500" --dst "$STUDENT"
"$PY" scripts/fasth3_sprint/build_h3_recovery_config.py \
 --template examples/train/configs/fasth3_uniform42_pdd4.yaml --student "$STUDENT" \
 --output "$OUTPUT_ROOT/config.yaml" --depth 42
for TARGET in 8 500; do
 RESUME=()
 if [[ "$TARGET" == 500 ]]; then RESUME=(--training.checkpoint.resume_from_checkpoint "$OUTPUT_ROOT/checkpoint-8"); fi
 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "$OUTPUT_ROOT/config.yaml" --training.loop.max_train_steps "$TARGET" \
  --training.checkpoint.output_dir "$OUTPUT_ROOT" \
  --training.checkpoint.training_state_checkpointing_steps "$([[ "$TARGET" == 8 ]] && echo 8 || echo 50)" \
  --training.tracker.run_name "uniform42-pdd4-${SLURM_JOB_ID}-to${TARGET}" "${RESUME[@]}"
 test -s "$OUTPUT_ROOT/checkpoint-${TARGET}/dcp/.metadata"
done
test -s "${OUTPUT_ROOT}/checkpoint-500/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-500" \
 --output-dir "${OUTPUT_ROOT}/export-500" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-500" --checkpoint-role uniform42-pdd4 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-500" --run-id "uniform42-pdd4-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
