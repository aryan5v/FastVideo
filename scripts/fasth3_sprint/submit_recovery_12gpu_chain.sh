#!/usr/bin/env bash
# Queue the 200-step Dense then VSA BF16 recovery runs on exactly 12 GPUs.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
PARTITION="${PARTITION:-all}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_ROOT="${SPRINT_ROOT}/logs/slurm"
LAUNCHER="${REPO_ROOT}/scripts/fasth3_sprint/slurm_h3_recovery.sbatch"
mkdir -p "${LOG_ROOT}"

test -s "${SPRINT_ROOT}/runs/h18-recovery-gates/dense-activation/checkpoint-2/metadata.json"
test -s "${SPRINT_ROOT}/runs/h18-recovery-gates/vsa-activation/checkpoint-2/metadata.json"
test ! -e "${SPRINT_ROOT}/runs/h36-recovery/dense-activation/checkpoint-200"
test ! -e "${SPRINT_ROOT}/runs/h36-recovery/vsa-activation/checkpoint-200"

submit_recovery() {
  local source_kind="$1"
  local dependency="$2"
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
    --job-name="h3-rec-long-${source_kind}-activation-s200-12g" \
    --output="${LOG_ROOT}/h3-rec-long-${source_kind}-activation-s200-12g-%j.out" \
    "${dependency_args[@]}" \
    --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' SOURCE_KIND='${source_kind}' MAP_STRATEGY=activation TARGET_STEPS=200 RUN_MODE=long /bin/bash '${LAUNCHER}'"
}

dense_job="$(submit_recovery dense "")"
vsa_job="$(submit_recovery vsa "${dense_job}")"

printf 'dense_job=%s\nvsa_job=%s\norder=dense-afterok-vsa\npartition=%s\n' \
  "${dense_job}" "${vsa_job}" "${PARTITION}"
