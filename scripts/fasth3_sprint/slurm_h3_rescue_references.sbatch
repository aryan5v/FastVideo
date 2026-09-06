#!/usr/bin/env bash
#SBATCH --job-name=h3-rescue-refs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json"
OUTPUT_ROOT="${SPRINT_ROOT}/videos/h36-rescue-references"
BASE_H3="/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe"
DENSE_V1="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
VSA_V1="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/snapshots/b65818d41939b5085451074fe8ca8b799f8d4921"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
mkdir -p "${OUTPUT_ROOT}" "${SPRINT_ROOT}/caches/wandb"
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

generate() {
  local role="$1" model="$2" attention="$3" grid_points="$4"
  local run_id="h36-rescue-reference-${role}-v1"
  local output="${OUTPUT_ROOT}/${run_id}"
  local cache="${TMPDIR:-/tmp}/fasth3-${SLURM_JOB_ID}/${run_id}"
  if [[ -s "${output}/run_manifest.json" ]] && [[ "$(find "${output}" -maxdepth 1 -name '*.mp4' | wc -l)" -eq 4 ]]; then
    return
  fi
  [[ -e "${output}" ]] && mv "${output}" "${output}.incomplete-job${SLURM_JOB_ID}-$(date +%s)"
  export TRITON_CACHE_DIR="${cache}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache}/torchinductor"
  export WANDB_DIR="${SPRINT_ROOT}/logs/${run_id}"
  mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${WANDB_DIR}"
  srun \
    --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
    --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
    --container-workdir="${REPO_ROOT}" \
    --export=ALL \
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      "${REPO_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
        --model-path "${model}" --checkpoint-role "${role}" --attention "${attention}" \
        --prompts "${PROMPTS}" --output-dir "${output}" --run-id "${run_id}" \
        --source-commit "${SOURCE_COMMIT}" --max-prompts 4 \
        --height 480 --width 832 --num-frames 124 --seed 2026082050 \
        --steps "${grid_points}" --num-gpus 4 \
        --profile strict --no-fa4 --no-compile --no-upload-videos
}

# Base H3 is the full-quality reference; each released V1 variant is its
# branch-matched four-call behavior reference.
generate base-h3-native50 "${BASE_H3}" dense 51
generate fasth3-v1-dense-4call "${DENSE_V1}" dense 5
generate fasth3-v1-vsa-4call "${VSA_V1}" vsa 5
date -Is > "${OUTPUT_ROOT}/completed_at.txt"
