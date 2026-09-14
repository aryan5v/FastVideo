#!/bin/bash
#SBATCH --job-name=h3-dmd2-repair2750
#SBATCH --partition=all
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release20b-dmd2-repair2750-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release20b-dmd2-repair2750-%j.log

set -uo pipefail

SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
CODE_ROOT=${SPRINT_ROOT}/code/release20b-dmd2-v12-corrected-v8
OLD_RUN=${SPRINT_ROOT}/runs/release20b-dmd2-v15-dense-c4-parent750-16gpu-cpu-fp32master-v1/job-8255
DMD_CHECKPOINT=${OLD_RUN}/checkpoint-2750
REPAIR_PARENT=${OLD_RUN}/repair-parent-step2750-student-only-v1
EXPORT_BASE_PARENT=${SPRINT_ROOT}/runs/release20b-folded-long-4k-v3/dmd-parent-step750-complete-v1
TEACHER_PARENT=${SPRINT_ROOT}/release-candidates/base-h3-teacher-complete-v1
OUTPUT_BASE=${SPRINT_ROOT}/runs/release20b-dmd2-v12-repair2750-c4-v1
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="$((22000 + SLURM_JOB_ID % 20000))"

set +e
srun \
  --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:4 \
  --kill-on-bad-exit=1 \
  --container-image='nvcr.io#nvidia/pytorch:25.06-py3' \
  --container-mounts=/mnt/nfs:/mnt/nfs,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  env \
  CODE_ROOT="${CODE_ROOT}" DMD_CHECKPOINT="${DMD_CHECKPOINT}" \
  REPAIR_PARENT="${REPAIR_PARENT}" TEACHER_PARENT="${TEACHER_PARENT}" \
  EXPORT_BASE_PARENT="${EXPORT_BASE_PARENT}" \
  OUTPUT_BASE="${OUTPUT_BASE}" SPRINT_ROOT="${SPRINT_ROOT}" \
  MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" NNODES=4 \
  bash "${CODE_ROOT}/scripts/run_release20b_dmd2_repair2750_16gpu.sh"
payload_rc=$?
set -e

if [[ "${payload_rc}" -ne 0 ]]; then
  echo "Repair-2750 failed with exit code ${payload_rc}; holding allocation for two hours."
  sleep 7200
fi
exit "${payload_rc}"
