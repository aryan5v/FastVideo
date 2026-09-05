#!/usr/bin/env bash
#SBATCH --job-name=h3-rescue-gate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail

: "${SOURCE_KIND:?SOURCE_KIND must be dense or vsa}"
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP must be 25, 50, 75, 100, or 200}"
case "${CHECKPOINT_STEP}" in 25|50|75|100|200) ;; *) exit 2 ;; esac
case "${SOURCE_KIND}" in
  dense) ATTENTION=dense ;;
  vsa) ATTENTION=vsa ;;
  *) exit 2 ;;
esac

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
CHECKPOINT="${SPRINT_ROOT}/runs/h36-four-call-rescue/${SOURCE_KIND}/checkpoint-${CHECKPOINT_STEP}"
EXPORT="${SPRINT_ROOT}/exports/h36-four-call-rescue/${SOURCE_KIND}-step${CHECKPOINT_STEP}-bf16"
OUTPUT="${SPRINT_ROOT}/videos/h36-four-call-rescue-${SOURCE_KIND}-step${CHECKPOINT_STEP}"
RUN_ID="h36-four-call-rescue-gate-${SOURCE_KIND}-step${CHECKPOINT_STEP}-v1"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json"
mkdir -p "$(dirname "${EXPORT}")" "${OUTPUT}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"
test -s "${CHECKPOINT}/metadata.json"
test -s "${CHECKPOINT}/dcp/.metadata"

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export WANDB_DIR="${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"

overwrite=()
[[ -e "${EXPORT}" ]] && overwrite+=(--overwrite)
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    -m fastvideo.train.entrypoint.dcp_to_diffusers \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${EXPORT}" \
      --role student \
      --load-precision bf16 \
      --verify \
      "${overwrite[@]}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    "${REPO_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
      --model-path "${EXPORT}" \
      --checkpoint-role "four-call-rescue-${SOURCE_KIND}-step${CHECKPOINT_STEP}" \
      --attention "${ATTENTION}" \
      --prompts "${PROMPTS}" \
      --output-dir "${OUTPUT}" \
      --run-id "${RUN_ID}" \
      --source-commit "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
      --max-prompts 4 \
      --height 480 --width 832 --num-frames 124 --steps 5 --num-gpus 4 \
      --profile strict --no-fa4 --no-compile --no-upload-videos

