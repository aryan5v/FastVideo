#!/bin/bash
#SBATCH --job-name=h3-b42-dmd4-res
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/best42-dmd4-resume/job-${SLURM_JOB_ID}"
export RESUME_CKPT="${SPRINT_ROOT}/runs/best42-dmd4/job-7194/checkpoint-400"
export CONFIG_PATH="${CODE_ROOT}/examples/train/configs/fasth3_best42_dmd4.yaml"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${RESUME_CKPT}/dcp/.metadata"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
# Resume the timed-out 500-phase run (job 7194) from checkpoint-400.
export TARGET_STEPS=500
srun --export=ALL --kill-on-bad-exit=1 --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
 --nnodes "${SLURM_JOB_NUM_NODES}" --nproc_per_node 4 --node_rank "${SLURM_PROCID}" \
 --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
 -m fastvideo.train.entrypoint.train --config "${CONFIG_PATH}" \
 --training.loop.max_train_steps "${TARGET_STEPS}" \
 --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
 --training.checkpoint.training_state_checkpointing_steps 100 \
 --training.checkpoint.resume_from_checkpoint "${RESUME_CKPT}" \
 --training.tracker.run_name "best42-dmd4-resume-${SLURM_JOB_ID}"
'
srun --nodes=1 --ntasks=1 --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"$PY" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-500" \
 --output-dir "${OUTPUT_ROOT}/export-500" --role student --load-precision bf16 --verify
"$PY" scripts/fasth3_sprint/run_baseline_matrix.py --model-path "${OUTPUT_ROOT}/export-500" \
 --checkpoint-role activation42-dmd4 --attention dense --attention-backend TORCH_SDPA \
 --prompts examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json \
 --output-dir "${OUTPUT_ROOT}/media-500" --run-id "activation42-dmd4-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
'
