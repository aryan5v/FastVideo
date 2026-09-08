#!/bin/bash
#SBATCH --job-name=h3-a34-prompt200
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export BRANCH=prompt58k
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base34-${BRANCH}-recovery/job-${SLURM_JOB_ID}"
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export STUDENT="${SPRINT_ROOT}/runs/activation34-recovery/job-6928/export-500"
export BASE_GATE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${STUDENT}/transformer/model.safetensors"
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
# Submit afterok6928 and6929. Fresh optimizer at recovered34 weights; all58669 prompt IDs are eligible.
# No paired flow loss on prompt placeholders. Actual consumed IDs are recorded.
# User authorizes experimental recovery without waiting for external parity/review gates.
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" --template "${CODE_ROOT}/examples/train/configs/fasth3_base42_recovery.yaml" --student "${STUDENT}" --output "${OUTPUT_ROOT}/config.yaml" --depth 34

 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${OUTPUT_ROOT}/config.yaml" \
  --training.loop.max_train_steps 200 \
  --method.allow_prompt_only true \
  --training.data.data_path "${SPRINT_ROOT}/data/user-prompts-20260906/index-v2" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "base34-${BRANCH}-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-200/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-200" \
 --output-dir "${OUTPUT_ROOT}/export-200" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-200" --checkpoint-role base34-prompt-recovered200 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-200" --run-id "base34-prompts-200-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
