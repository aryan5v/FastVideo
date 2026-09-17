#!/bin/bash
# 34-block recovery launcher, hardened after four failures in one day.
#
# Fixes vs submit_release14b_from1000.sbatch:
#   1. MASTER_PORT derived from SLURM_JOB_ID (was hardcoded 64621 -> invites collisions
#      with any other job on the master node; the 18:52 death was a C10d store failure).
#   2. No `sleep 7200` hold. It burned 4 GPU-hours across three incidents and never
#      helped. Fail fast so the slot can be reclaimed / resubmitted immediately.
#   3. NCCL fail-fast: TORCH_NCCL_ASYNC_ERROR_HANDLING=1 plus a heartbeat watchdog.
#      NOTE: timeouts must exceed the longest legitimate stall (a 20B checkpoint write can
#      take 20+ min). An earlier revision used 1800s and self-destructed a healthy run at
#      exactly 30:06 -- do not tighten these below 3600s.
#   4. RESUME_PATH auto-selects the NEWEST COMPLETE checkpoint instead of a hardcoded step.
#      A resubmit therefore always continues rather than replaying from a stale step.
#SBATCH --job-name=h3-14b-4k-wb
#SBATCH --partition=hpc-rack-2
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=21-00:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release14b-hardened-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-release14b-hardened-%j.log

set -u

SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
CODE_ROOT=${SPRINT_ROOT}/code/release-execution-2e7fa15f
OUTPUT_ROOT=${SPRINT_ROOT}/runs/release14b-folded-long-4k-v3

# MASTER_PORT derived from the job id -- never a fixed port.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# --- auto-select the newest COMPLETE checkpoint ---------------------------------
RESUME_PATH=""
for d in $(ls -1d "${OUTPUT_ROOT}"/checkpoint-* 2>/dev/null \
           | sed 's#.*/checkpoint-##' | grep -E '^[0-9]+$' | sort -n -r); do
  C="${OUTPUT_ROOT}/checkpoint-${d}"
  if [[ -s "${C}/metadata.json" && -s "${C}/dcp/.metadata" ]] \
     && [[ "$(find "${C}" -maxdepth 1 -name 'rng_state_rank*.pt' | wc -l)" -eq 16 ]]; then
    RESUME_PATH="${C}"
    break
  fi
done
if [[ -z "${RESUME_PATH}" ]]; then
  echo "FATAL: no complete checkpoint found under ${OUTPUT_ROOT}" >&2
  exit 2
fi
echo "=============================================================="
echo "RESUMING FROM : ${RESUME_PATH}"
echo "MASTER_ADDR   : ${MASTER_ADDR}"
echo "MASTER_PORT   : ${MASTER_PORT}"
echo "=============================================================="

set +e
srun --kill-on-bad-exit=1 \
  --container-image='nvcr.io#nvidia/pytorch:25.06-py3' \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" \
  env SPRINT_ROOT="${SPRINT_ROOT}" CODE_ROOT="${CODE_ROOT}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" RESUME_PATH="${RESUME_PATH}" \
      MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
  bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}/tracker"
export WANDB_RUN_ID=release14b-folded-long-4k-v3 WANDB_RESUME=allow
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- fail fast instead of hanging ------------------------------------------------
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/release14b-folded-long-4k-v3/node-${SLURM_PROCID}"
mkdir -p "${TRITON_CACHE_DIR}"
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
  --nnodes 4 --nproc_per_node 4 --node_rank "${SLURM_PROCID}" \
  --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_conf "timeout=7200,join_timeout=7200,read_timeout=7200" \
  -m fastvideo.train.entrypoint.train \
  --config "${SPRINT_ROOT}/job-scripts/release14b_recovery_wandb.yaml" \
  --training.distributed.num_gpus 16 \
  --training.distributed.hsdp_replicate_dim 4 \
  --training.loop.gradient_accumulation_steps 3 \
  --training.loop.max_train_steps 4000 \
  --method.student_state_probability 0.0 \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 100 \
  --training.tracker.run_name release14b-folded-long-4k-v3 \
  --training.checkpoint.resume_from_checkpoint "${RESUME_PATH}" \
  --callbacks.validation.dataset_file "${SPRINT_ROOT}/job-scripts/release14b_validation_five_wrapped.json"
'
rc=$?
echo "34-block recovery exited rc=${rc} (no hold; allocation released immediately)"
exit "${rc}"
