#!/usr/bin/env bash
#SBATCH --job-name=h3-recovery-4diag
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json"
OUTPUT_ROOT="${SPRINT_ROOT}/videos/h36-recovery-four-prompt-diagnostics"
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
  local kind="$1" stage="$2" model="$3" grid_points="$4"
  local attention=dense
  [[ "${kind}" == vsa ]] && attention=vsa
  local run_id="h36-four-prompt-${kind}-${stage}-grid${grid_points}-v1"
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
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    "${REPO_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
      --model-path "${model}" \
      --checkpoint-role "diagnostic-${kind}-${stage}" \
      --attention "${attention}" \
      --prompts "${PROMPTS}" \
      --output-dir "${output}" \
      --run-id "${run_id}" \
      --source-commit "${SOURCE_COMMIT}" \
      --max-prompts 4 \
      --height 480 --width 832 --num-frames 124 --seed 2026082050 \
      --steps "${grid_points}" --num-gpus 4 \
      --profile strict --no-fa4 --no-compile --no-upload-videos
}

for kind in dense vsa; do
  initial="${SPRINT_ROOT}/checkpoints/h18-candidates/${kind}-activation"
  step100="${SPRINT_ROOT}/exports/h36-recovery-diagnostics/${kind}-activation-step100-raw"
  step200="${SPRINT_ROOT}/exports/h36-recovery-comparison/${kind}-activation-step200-raw"
  test -s "${initial}/transformer/config.json"
  test -s "${step100}/transformer/model.safetensors"
  test -s "${step200}/transformer/model.safetensors"
  generate "${kind}" step0 "${initial}" 5
  generate "${kind}" step100 "${step100}" 5
  generate "${kind}" step200 "${step200}" 5
  generate "${kind}" step200-long "${step200}" 51
done

date -Is > "${OUTPUT_ROOT}/completed_at.txt"
