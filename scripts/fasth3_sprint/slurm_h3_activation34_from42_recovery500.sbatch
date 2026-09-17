#!/bin/bash
#SBATCH --job-name=h3-a34-to500
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export BRANCH=activation34
case "${BRANCH}" in activation34) ;; *) exit 2 ;; esac
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/${BRANCH}-recovery/job-${SLURM_JOB_ID}"
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export STUDENT="${OUTPUT_ROOT}/student34"
export BASE_GATE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${SPRINT_ROOT}/preserved/activation42-step500-job6878/export-500/transformer/model.safetensors"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
export PARENT="${SPRINT_ROOT}/preserved/activation42-step500-job6878/export-500"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 \
 "${CODE_ROOT}/scripts/fasth3_sprint/score_minimax_h3_blocks.py" --model-root "${PARENT}" \
 --source-revision recovered-activation42-job6878-step500 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" \
 --corpus-root "${SPRINT_ROOT}/data/day1-mask-split/train" --output-dir "${OUTPUT_ROOT}/partials" \
 --job-index 0 --num-jobs 1 --max-records-per-shard 12 --allow-incomplete-calibration
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/aggregate_minimax_h3_block_scores.py" \
 --partials "${OUTPUT_ROOT}/partials" --output-dir "${OUTPUT_ROOT}/maps" --expected-shards 4 \
 --expected-records 48 --allow-incomplete-categories --keep-blocks 34 --run-id "activation34-map-${SLURM_JOB_ID}"
"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/prune_minimax_h3_blocks.py" \
 --src "${PARENT}/transformer" --dst "${STUDENT}/transformer" \
 --block-map "${OUTPUT_ROOT}/maps/activation_34_block_map.json" --strategy recovered42-activation34-screen48 \
 --source-model "${PARENT}" --source-revision recovered-activation42-job6878-step500
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/assemble_h3_stage.py" --base "${BASE}" --student "${STUDENT}"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${STUDENT}" --checkpoint-role activation34-step0 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-0" --run-id "activation34-step0-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
# User authorizes experimental recovery without waiting for external parity/review gates.
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" --template "${CODE_ROOT}/examples/train/configs/fasth3_base42_recovery.yaml" --student "${STUDENT}" --output "${OUTPUT_ROOT}/config.yaml" --depth 34

 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${OUTPUT_ROOT}/config.yaml" \
  --training.loop.max_train_steps 500 \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "base42-${BRANCH}-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-500/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-500" \
 --output-dir "${OUTPUT_ROOT}/export-500" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-500" --checkpoint-role activation34-recovered500 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-500" --run-id "activation34-500-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
