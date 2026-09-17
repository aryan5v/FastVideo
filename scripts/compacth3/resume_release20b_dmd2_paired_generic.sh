#!/bin/bash
# Launch one 32-GPU DMD2 world across two already-running four-node jobs.
# Usage: bash resume_release20b_dmd2_paired_generic.sh JOB_A JOB_B RESUME_STEP
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 JOB_A JOB_B RESUME_STEP" >&2
  exit 2
fi

JOB_A="$1"
JOB_B="$2"
RESUME_STEP="$3"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
VALIDATION_EVERY="${VALIDATION_EVERY:-200}"
SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
CODE_ROOT=${SPRINT_ROOT}/code/release20b-dmd2-v12-corrected-v17
SELECTED_PARENT=${SPRINT_ROOT}/runs/release20b-folded-long-4k-v3/dmd-parent-step750-complete-v1
TEACHER_PARENT=${SPRINT_ROOT}/release-candidates/base-h3-teacher-complete-v1
OUTPUT_BASE=${SPRINT_ROOT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3
RUN_ID=paired8972-8975-4000-v3
OUTPUT_ROOT=${OUTPUT_BASE}/job-${RUN_ID}
RESUME_PATH=${OUTPUT_ROOT}/checkpoint-${RESUME_STEP}

[[ "$(squeue -h -j "${JOB_A}" -o %T)" == "RUNNING" ]]
[[ "$(squeue -h -j "${JOB_B}" -o %T)" == "RUNNING" ]]
NODES_A="$(squeue -h -j "${JOB_A}" -o %N)"
NODES_B="$(squeue -h -j "${JOB_B}" -o %N)"
[[ "$(scontrol show hostnames "${NODES_A}" | wc -l)" -eq 4 ]]
[[ "$(scontrol show hostnames "${NODES_B}" | wc -l)" -eq 4 ]]

# A resumable checkpoint is committed only after all distributed state and all
# rank-local RNG snapshots exist. Never fall back to a partially written save.
test -s "${RESUME_PATH}/.complete"
test -s "${RESUME_PATH}/dcp/.metadata"
test -s "${RESUME_PATH}/metadata.json"
[[ "$(find "${RESUME_PATH}/dcp" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 32 ]]
[[ "$(find "${RESUME_PATH}" -maxdepth 1 -type f -name 'rng_state_rank*.pt' | wc -l)" -eq 32 ]]

MASTER_ADDR="$(scontrol show hostnames "${NODES_A}" | head -n 1)"
MASTER_PORT="$((30000 + JOB_A % 10000))"

launch_half() {
  local job_id="$1" node_list="$2" rank_base="$3"
  local log_path=${SPRINT_ROOT}/resume-dmd2-${RUN_ID}-job${job_id}-from${RESUME_STEP}.log
  nohup srun --overlap --jobid="${job_id}" --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gres=gpu:4 --nodelist="${node_list}" \
    --kill-on-bad-exit=1 \
    --container-image='nvcr.io#nvidia/pytorch:25.06-py3' \
    --container-mounts=/mnt/nfs:/mnt/nfs,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
    env CODE_ROOT="${CODE_ROOT}" SELECTED_PARENT="${SELECTED_PARENT}" \
      TEACHER_PARENT="${TEACHER_PARENT}" OUTPUT_BASE="${OUTPUT_BASE}" \
      SPRINT_ROOT="${SPRINT_ROOT}" MASTER_ADDR="${MASTER_ADDR}" \
      MASTER_PORT="${MASTER_PORT}" NNODES=8 NODE_RANK_BASE="${rank_base}" \
      RUN_ID="${RUN_ID}" PRODUCTION_TARGET=4000 RESUME_PATH="${RESUME_PATH}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" VALIDATION_EVERY="${VALIDATION_EVERY}" \
      bash "${CODE_ROOT}/scripts/run_release20b_dmd2_v12_16gpu.sh" \
      >"${log_path}" 2>&1 &
  echo "$!" > "${log_path}.pid"
}

launch_half "${JOB_A}" "${NODES_A}" 0
launch_half "${JOB_B}" "${NODES_B}" 4

printf 'job_a=%s\njob_b=%s\nresume_step=%s\ncheckpoint_every=%s\nvalidation_every=%s\nmaster=%s:%s\nlaunched_at=%s\n' \
  "${JOB_A}" "${JOB_B}" "${RESUME_STEP}" "${CHECKPOINT_EVERY}" \
  "${VALIDATION_EVERY}" "${MASTER_ADDR}" "${MASTER_PORT}" "$(date -u +%FT%TZ)" \
  > "${OUTPUT_ROOT}/paired-rollover-from-${RESUME_STEP}.receipt"

echo "Launched one 32-GPU world from checkpoint ${RESUME_STEP} across ${JOB_A}+${JOB_B}."
