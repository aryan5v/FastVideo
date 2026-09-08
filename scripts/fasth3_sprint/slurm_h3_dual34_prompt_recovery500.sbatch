#!/bin/bash
#SBATCH --job-name=h3-34-prompts500
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export BRANCH="${1:?uniform or activation}"
case "$BRANCH" in uniform|activation) ;; *) exit 2 ;; esac
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/${BRANCH}34-prompts-recovery/job-${SLURM_JOB_ID}"
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export STUDENT="${SPRINT_ROOT}/runs/activation34-recovery/job-6928/student34"
if [[ "$BRANCH" == uniform ]]; then export STUDENT="${OUTPUT_ROOT}/student34"; fi
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
# Matched fresh500-update arms; no paired-only updates carried into either arm.
# No paired flow loss on prompt placeholders. Actual consumed IDs are recorded.
# User authorizes experimental recovery without waiting for external parity/review gates.
if [[ "$BRANCH" == uniform ]]; then
 PARENT="${SPRINT_ROOT}/preserved/uniform42-step500-job6924/export-500"
 MAP="0,1,2,4,5,6,7,9,10,11,12,14,15,16,17,19,20,21,22,24,25,26,27,29,30,31,32,34,35,36,37,39,40,41"
 "$PY" scripts/checkpoint_conversion/prune_minimax_h3_blocks.py --src "$PARENT/transformer" \
  --dst "$STUDENT/transformer" --block-map "$MAP" --strategy uniform --source-model "$PARENT" --source-revision job6924-step500
 "$PY" scripts/fasth3_sprint/assemble_h3_stage.py --base "$BASE" --student "$STUDENT"
 "$PY" scripts/fasth3_sprint/run_baseline_matrix.py --model-path "$STUDENT" \
  --checkpoint-role uniform34-step0 --attention dense --attention-backend TORCH_SDPA \
  --prompts examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json \
  --output-dir "$OUTPUT_ROOT/media-0" --run-id "uniform34-step0-${SLURM_JOB_ID}" \
  --source-commit "$(cat "$CODE_ROOT/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
  --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
  --profile strict --no-fa4 --no-compile --no-upload-videos
fi
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" --template "${CODE_ROOT}/examples/train/configs/fasth3_base42_recovery.yaml" --student "${STUDENT}" --output "${OUTPUT_ROOT}/config.yaml" --depth 34

 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${OUTPUT_ROOT}/config.yaml" \
  --training.loop.max_train_steps 500 \
  --method.allow_prompt_only true \
  --training.data.data_path "${SPRINT_ROOT}/data/user-prompts-20260906/index-v2" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "base34-${BRANCH}-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-500/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-500" \
 --output-dir "${OUTPUT_ROOT}/export-500" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-500" --checkpoint-role base34-prompt-recovered500 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-500" --run-id "base34-prompts-500-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
