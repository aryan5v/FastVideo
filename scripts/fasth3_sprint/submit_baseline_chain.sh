#!/usr/bin/env bash
# Submit the SM100a build, the mandatory V1 synchronization gate, and baseline matrices.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
SLURM_LOG_DIR="${SPRINT_ROOT}/logs/slurm"
BUILD_SCRIPT="${REPO_ROOT}/scripts/fasth3_sprint/slurm_build_sm100a.sbatch"
BASELINE_SCRIPT="${REPO_ROOT}/scripts/fasth3_sprint/slurm_baseline.sbatch"
mkdir -p "${SLURM_LOG_DIR}"

submit_wrapped() {
  local job_name="$1"
  local gpus="$2"
  local time_limit="$3"
  local dependency="$4"
  local command="$5"
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
  fi
  sbatch --parsable --export=NIL \
    --partition=all --nodes=1 --ntasks=1 --gres="gpu:${gpus}" \
    --time="${time_limit}" --job-name="${job_name}" \
    --output="${SLURM_LOG_DIR}/${job_name}-%j.out" \
    "${dependency_args[@]}" --wrap="${command}"
}

if [[ -n "${KERNEL_JOB_ID:-}" ]]; then
  if [[ ! -f "${SPRINT_ROOT}/manifests/sm100a_kernel_receipt.json" ]]; then
    echo "KERNEL_JOB_ID was provided but the persistent SM100a receipt is missing" >&2
    exit 1
  fi
  build_job="${KERNEL_JOB_ID}"
  sync_dependency=""
else
  build_command="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' /bin/bash '${BUILD_SCRIPT}'"
  build_job="$(submit_wrapped h3-sm100a-build 1 02:00:00 "" "${build_command}")"
  sync_dependency="${build_job}"
fi

submit_baseline() {
  local job_name="$1"
  local dependency="$2"
  local model_path="$3"
  local checkpoint_role="$4"
  local attention="$5"
  local run_id="$6"
  local max_prompts="$7"
  local command
  command="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' MODEL_PATH='${model_path}' CHECKPOINT_ROLE='${checkpoint_role}' ATTENTION='${attention}' RUN_ID='${run_id}' MAX_PROMPTS='${max_prompts}' UPLOAD_VIDEOS=0 /bin/bash '${BASELINE_SCRIPT}'"
  submit_wrapped "${job_name}" 4 06:00:00 "${dependency}" "${command}"
}

vsa_model="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
dense_model="FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
base_model="MiniMaxAI/MiniMax-H3"

sync_job="$(submit_baseline h3-vsa-sync "${sync_dependency}" "${vsa_model}" released-v1-vsa vsa h0-vsa-sync-gate 1)"
base_job="$(submit_baseline h3-base-matrix "${sync_job}" "${base_model}" base-model dense h0-base-four-call-matrix 12)"
vsa_job="$(submit_baseline h3-vsa-matrix "${sync_job}" "${vsa_model}" released-v1-vsa vsa h0-vsa-four-call-matrix 12)"
dense_job="$(submit_baseline h3-dense-matrix "${sync_job}" "${dense_model}" released-v1-dense dense h0-dense-four-call-matrix 12)"

printf 'build_job=%s\nsync_job=%s\nbase_job=%s\nvsa_job=%s\ndense_job=%s\n' \
  "${build_job}" "${sync_job}" "${base_job}" "${vsa_job}" "${dense_job}"
