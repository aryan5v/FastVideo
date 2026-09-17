#!/usr/bin/env bash
# Export and evaluate the failed H3 recovery lineage without starting training.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
REPO_ROOT="${SPRINT_ROOT}/repo"
EXPORT_ROOT="${SPRINT_ROOT}/exports/h36-recovery-diagnostics"
OUTPUT_ROOT="${SPRINT_ROOT}/videos/h36-recovery-diagnostics"
RECEIPT_ROOT="${SPRINT_ROOT}/eval/h36-recovery-diagnostics"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/showcase_prompt.json"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SEED=2026082050

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export HF_XET_CACHE="${SPRINT_ROOT}/caches/hf-xet"
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXPORT_ROOT}" "${OUTPUT_ROOT}" "${RECEIPT_ROOT}" "${WANDB_CACHE_DIR}"

export_checkpoint() {
  local kind="$1" step="$2"
  local checkpoint="${SPRINT_ROOT}/runs/h36-recovery/${kind}-activation/checkpoint-${step}"
  local output="${EXPORT_ROOT}/${kind}-activation-step${step}-raw"
  test -s "${checkpoint}/dcp/.metadata"
  if [[ -s "${output}/transformer/model.safetensors" ]]; then
    return
  fi
  local overwrite=()
  [[ -e "${output}" ]] && overwrite+=(--overwrite)
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint "${checkpoint}" --output-dir "${output}" --role student --verify "${overwrite[@]}"
}

generate_one() {
  local kind="$1" stage="$2" model="$3" calls="$4"
  local attention=dense
  [[ "${kind}" == vsa ]] && attention=vsa
  local run_id="h36-diagnostic-${kind}-${stage}-${calls}call"
  local output="${OUTPUT_ROOT}/${run_id}"
  if [[ -s "${output}/run_manifest.json" ]] && [[ "$(find "${output}" -maxdepth 1 -name '*.mp4' | wc -l)" -eq 1 ]]; then
    return
  fi
  if [[ -e "${output}" ]]; then
    mv "${output}" "${output}.incomplete-job${SLURM_JOB_ID:-manual}-$(date +%s)"
  fi
  local cache="${TMPDIR:-/tmp}/fasth3-${SLURM_JOB_ID:-manual}/${run_id}"
  export TRITON_CACHE_DIR="${cache}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache}/torchinductor"
  export WANDB_DIR="${SPRINT_ROOT}/logs/${run_id}"
  mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${WANDB_DIR}"
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    scripts/fasth3_sprint/run_baseline_matrix.py \
    --model-path "${model}" --checkpoint-role "diagnostic-${kind}-${stage}" \
    --attention "${attention}" --prompts "${PROMPTS}" --output-dir "${output}" \
    --run-id "${run_id}" --source-commit "${SOURCE_COMMIT}" --max-prompts 1 \
    --height 480 --width 832 --num-frames 124 --seed "${SEED}" \
    --steps "$((calls + 1))" --profile strict --no-fa4 --no-compile --no-upload-videos
}

for kind in dense vsa; do
  export_checkpoint "${kind}" 100
  # Step 200 was already exported for the comparison matrix; reuse that exact
  # raw export so the audit covers the artifact that visibly failed.
  step200="${SPRINT_ROOT}/exports/h36-recovery-comparison/${kind}-activation-step200-raw"
  test -s "${step200}/transformer/model.safetensors"
  initial="${SPRINT_ROOT}/checkpoints/h18-candidates/${kind}-activation"
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    scripts/fasth3_sprint/audit_recovery_weights.py \
    --initial-model "${initial}" --recovered-model "${step200}" \
    --checkpoint "${SPRINT_ROOT}/runs/h36-recovery/${kind}-activation/checkpoint-200" \
    --output "${RECEIPT_ROOT}/${kind}-step0-vs-step200-weight-audit.json"

  generate_one "${kind}" step0 "${initial}" 4
  generate_one "${kind}" step100 "${EXPORT_ROOT}/${kind}-activation-step100-raw" 4
  generate_one "${kind}" step200 "${step200}" 4
  generate_one "${kind}" step200-long "${step200}" 50
done

date -Is > "${RECEIPT_ROOT}/completed_at.txt"
