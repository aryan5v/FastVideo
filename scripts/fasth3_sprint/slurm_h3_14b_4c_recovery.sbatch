#!/bin/bash
#SBATCH --job-name=h3-14b-4c-kd
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export STUDENT="${SPRINT_ROOT}/release-candidates/14B-4C-FAST-BRANCH-v0"
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/release14b-4c-aware-recovery-v1"
test -s "${STUDENT}/FROZEN_MANIFEST.json"
test -s "${STUDENT}/production_media_parity.json"
test -s "${STUDENT}/training-config.yaml"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node 4 -m fastvideo.train.entrypoint.train \
  --config "${STUDENT}/training-config.yaml" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.tracker.run_name "release14b-4c-aware-recovery-v1"
'
test -s "${OUTPUT_ROOT}/checkpoint-1000/metadata.json"
test -d "${OUTPUT_ROOT}/checkpoint-1000/dcp"
