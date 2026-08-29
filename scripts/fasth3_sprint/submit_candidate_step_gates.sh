#!/usr/bin/env bash
# Queue four fresh finite steps and four persistent resume steps in <=32-GPU rounds.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${DENSE_ACTIVATION_JOB_ID:?DENSE_ACTIVATION_JOB_ID is required}"
: "${DENSE_UNIFORM_JOB_ID:?DENSE_UNIFORM_JOB_ID is required}"
: "${VSA_ACTIVATION_JOB_ID:?VSA_ACTIVATION_JOB_ID is required}"
: "${VSA_UNIFORM_JOB_ID:?VSA_UNIFORM_JOB_ID is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}"

submit_gate() {
  local source_kind="$1"
  local map_strategy="$2"
  local gate_stage="$3"
  local dependency="$4"
  sbatch --parsable --export=NIL \
    --dependency="afterok:${dependency}" --kill-on-invalid-dep=yes \
    --partition=all --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:4 \
    --exclusive --time=02:00:00 \
    --job-name="h3-${gate_stage}-${source_kind}-${map_strategy}" \
    --output="${LOG_DIR}/h3-${gate_stage}-${source_kind}-${map_strategy}-%j.out" \
    --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' SOURCE_KIND='${source_kind}' MAP_STRATEGY='${map_strategy}' GATE_STAGE='${gate_stage}' /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_candidate_step_gate.sbatch'"
}

# Round 1: two Dense candidates, 32 GPUs total.
dense_activation_fresh="$(submit_gate dense activation fresh "${DENSE_ACTIVATION_JOB_ID}")"
dense_uniform_fresh="$(submit_gate dense uniform fresh "${DENSE_UNIFORM_JOB_ID}")"
dense_fresh_barrier="${dense_activation_fresh}:${dense_uniform_fresh}"

# Round 2: resume both Dense checkpoints for their second finite step.
dense_activation_resume="$(submit_gate dense activation resume "${dense_fresh_barrier}")"
dense_uniform_resume="$(submit_gate dense uniform resume "${dense_fresh_barrier}")"
dense_resume_barrier="${dense_activation_resume}:${dense_uniform_resume}"

# Round 3: two exact-backend VSA candidates, after Dense resume receipts exist.
vsa_activation_fresh="$(submit_gate vsa activation fresh "${dense_resume_barrier}:${VSA_ACTIVATION_JOB_ID}")"
vsa_uniform_fresh="$(submit_gate vsa uniform fresh "${dense_resume_barrier}:${VSA_UNIFORM_JOB_ID}")"
vsa_fresh_barrier="${vsa_activation_fresh}:${vsa_uniform_fresh}"

# Round 4: resume both VSA checkpoints for their second finite step.
vsa_activation_resume="$(submit_gate vsa activation resume "${vsa_fresh_barrier}")"
vsa_uniform_resume="$(submit_gate vsa uniform resume "${vsa_fresh_barrier}")"

printf 'dense_fresh_jobs=%s %s\n' "${dense_activation_fresh}" "${dense_uniform_fresh}"
printf 'dense_resume_jobs=%s %s\n' "${dense_activation_resume}" "${dense_uniform_resume}"
printf 'vsa_fresh_jobs=%s %s\n' "${vsa_activation_fresh}" "${vsa_uniform_fresh}"
printf 'vsa_resume_jobs=%s %s\n' "${vsa_activation_resume}" "${vsa_uniform_resume}"

