#!/usr/bin/env bash
# Briefly hold one schedulable GPU node while retrying the real 12-GPU chain.
# A second copy may remain pending to give the dynamic-node provisioner demand.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
SUBMITTER="${REPO_ROOT}/scripts/fasth3_sprint/submit_recovery_12gpu_chain.sh"
LOCK_PATH="${SPRINT_ROOT}/logs/slurm/recovery-12gpu-submit.lock"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-15}"
START_SECONDS="$(date +%s)"
QUEUE_USER="${USER:-${SLURM_JOB_USER:-$(id -un)}}"
READY_RECEIPT="${READY_RECEIPT:-}"

mkdir -p "${SPRINT_ROOT}/logs/slurm"

already_queued() {
  squeue -u "${QUEUE_USER}" -h -n h3-rec-long-dense-activation-s200-12g \
    -o '%i' | grep -q '[0-9]'
}

receipt_ready() {
  if [[ -z "${READY_RECEIPT}" ]]; then
    return 0
  fi
  [[ -s "${READY_RECEIPT}" ]] || return 1
  python3 - "${READY_RECEIPT}" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if receipt.get("remaining_missing_count") == 0 else 1)
PY
}

while (( "$(date +%s)" - START_SECONDS < MAX_WAIT_SECONDS )); do
  if already_queued; then
    echo "$(date -Is) dense recovery is already queued"
    exit 0
  fi

  registered_nodes="$(sinfo -N -h -p all -o '%N' | sort -u | wc -l | tr -d ' ')"
  echo "$(date -Is) registered_nodes=${registered_nodes} waiting_for=3 receipt_ready=$(receipt_ready && echo yes || echo no)"
  if (( registered_nodes >= 3 )) && receipt_ready; then
    exec 9>"${LOCK_PATH}"
    if flock -n 9; then
      if already_queued; then
        echo "$(date -Is) another submitter queued dense recovery"
        exit 0
      fi
      SPRINT_ROOT="${SPRINT_ROOT}" bash "${SUBMITTER}"
      exit 0
    fi
  fi
  sleep "${POLL_SECONDS}"
done

echo "$(date -Is) timed out before three GPU nodes registered" >&2
exit 75
