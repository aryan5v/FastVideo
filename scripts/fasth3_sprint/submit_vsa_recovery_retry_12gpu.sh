#!/usr/bin/env bash
# Queue a 12-GPU VSA scale gate, then the 200-step VSA recovery after it passes.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
PARTITION="${PARTITION:-all}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_ROOT="${SPRINT_ROOT}/logs/slurm"
LAUNCHER="${REPO_ROOT}/scripts/fasth3_sprint/slurm_h3_recovery.sbatch"
mkdir -p "${LOG_ROOT}"

test -s "${SPRINT_ROOT}/runs/h18-recovery-gates/vsa-activation/checkpoint-2/metadata.json"
test -s "${SPRINT_ROOT}/runs/h36-recovery/dense-activation/checkpoint-200/metadata.json"
test ! -e "${SPRINT_ROOT}/runs/h36-recovery/vsa-activation/checkpoint-200"

submit_vsa() {
  local run_mode="$1"
  local target_steps="$2"
  local job_suffix="$3"
  local dependency="$4"
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
  fi
  sbatch --parsable --export=NIL \
    --partition="${PARTITION}" \
    --nodes=3 \
    --ntasks=3 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --time=24:00:00 \
    --job-name="h3-vsa-${job_suffix}-12g" \
    --output="${LOG_ROOT}/h3-vsa-${job_suffix}-12g-%j.out" \
    "${dependency_args[@]}" \
    --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' SOURCE_KIND=vsa MAP_STRATEGY=activation TARGET_STEPS='${target_steps}' RUN_MODE='${run_mode}' RUN_TAG=hsdp3x4 /bin/bash '${LAUNCHER}'"
}

scale_gate_job="$(submit_vsa scale_gate 1 scale-gate-s1 "")"
recovery_job="$(submit_vsa long 200 recovery-s200 "${scale_gate_job}")"

printf 'scale_gate_job=%s\nrecovery_job=%s\norder=scale-gate-afterok-recovery\npartition=%s\n' \
  "${scale_gate_job}" "${recovery_job}" "${PARTITION}"
