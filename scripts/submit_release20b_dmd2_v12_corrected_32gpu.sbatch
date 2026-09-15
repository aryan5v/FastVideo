#!/bin/bash
#SBATCH --job-name=h3-dmd2-v12-32g
#SBATCH --partition=all
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release20b-dmd2-v12-32g-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release20b-dmd2-v12-32g-%j.log

set -uo pipefail

SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
CODE_ROOT=${SPRINT_ROOT}/code/release20b-dmd2-v12-corrected-v9
SELECTED_PARENT=${SPRINT_ROOT}/runs/release20b-folded-long-4k-v3/dmd-parent-step750-complete-v1
TEACHER_PARENT=${SPRINT_ROOT}/release-candidates/base-h3-teacher-complete-v1
OUTPUT_BASE=${SPRINT_ROOT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-v1
MASTER_ADDR="${SLURMD_NODENAME}"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

set +e
srun \
  --nodes=8 --ntasks=8 --ntasks-per-node=1 --gres=gpu:4 \
  --kill-on-bad-exit=1 \
  --container-image='nvcr.io#nvidia/pytorch:25.06-py3' \
  --container-mounts=/mnt/nfs:/mnt/nfs,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  env \
  CODE_ROOT="${CODE_ROOT}" SELECTED_PARENT="${SELECTED_PARENT}" \
  TEACHER_PARENT="${TEACHER_PARENT}" OUTPUT_BASE="${OUTPUT_BASE}" \
  SPRINT_ROOT="${SPRINT_ROOT}" MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" NNODES=8 PRODUCTION_TARGET=4000 \
  bash "${CODE_ROOT}/scripts/run_release20b_dmd2_v12_16gpu.sh"
payload_rc=$?
set -e

if [[ "${payload_rc}" -ne 0 ]]; then
  failure_receipt="${OUTPUT_BASE}/allocation-held-after-failure-job-${SLURM_JOB_ID}"
  printf 'exit_code=%s\nfailed_at=%s\n' "${payload_rc}" "$(date -u +%FT%TZ)" > "${failure_receipt}"
  echo "32-GPU corrected DMD2 failed with exit code ${payload_rc}; holding allocation for two hours."
  sleep 7200
fi
exit "${payload_rc}"
