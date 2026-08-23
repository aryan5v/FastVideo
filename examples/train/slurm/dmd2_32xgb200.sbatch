#!/bin/bash
# Multi-node MiniMax-H3 DMD2 launch; defaults to 8 x 4-GPU trays.
#
#   sbatch examples/train/slurm/dmd2_32xgb200.sbatch
#   sbatch --nodes=2 examples/train/slurm/dmd2_32xgb200.sbatch          # smaller sweep
#   CONFIG=path/to/other.yaml sbatch examples/train/slurm/dmd2_32xgb200.sbatch
#
# The launcher derives the SP=1/full-shard mesh from the allocation. Smaller
# allocations still require an independent memory-capacity check.
#
# Cluster-specific knobs (partition/account/container) are marked CLUSTER.

#SBATCH --job-name=h3-dmd2-32xgb200
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<CLUSTER: partition>
##SBATCH --account=<CLUSTER: account>

set -euo pipefail

# ── paths (shared FS visible on every tray) ──────────────────────────────
REPO="${REPO:-/mnt/FastVideo}"
VENV="${VENV:-${REPO}/.venv}"
CONFIG="${CONFIG:-${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml}"

# ── topology derived from the allocation ─────────────────────────────────
NNODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"   # "4" or "gb200:4" -> 4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
# Topology overrides default to SP=1 and full-shard FSDP across the world.
# For SP across each tray with HSDP instead:
# SP_SIZE=4 HSDP_REPLICATE=$NNODES HSDP_SHARD=4.
SP_SIZE="${SP_SIZE:-1}"
HSDP_REPLICATE="${HSDP_REPLICATE:-1}"
HSDP_SHARD="${HSDP_SHARD:-${WORLD_SIZE}}"
if (( WORLD_SIZE % SP_SIZE != 0 )); then
  echo "TOPOLOGY GATE FAILED: world size ${WORLD_SIZE} is not divisible by SP ${SP_SIZE}" >&2
  exit 1
fi
if (( HSDP_REPLICATE * HSDP_SHARD != WORLD_SIZE )); then
  echo "TOPOLOGY GATE FAILED: HSDP ${HSDP_REPLICATE}x${HSDP_SHARD} != world size ${WORLD_SIZE}" >&2
  exit 1
fi
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "=== H3 DMD2 multi-node launch ==="
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "World size:   ${WORLD_SIZE}"
echo "Topology:     sp=${SP_SIZE} hsdp=(replicate=${HSDP_REPLICATE}, shard=${HSDP_SHARD})"
echo "Rendezvous:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"

# A preflighted v10 launch pins the execution checkout in its wrapper. This
# catches a mutable shared checkout before any model or data state is loaded.
if [ -n "${EXPECTED_V10_COMMIT:-}" ]; then
  observed_commit="$(git -C "${REPO}" rev-parse HEAD)"
  if [ "${observed_commit}" != "${EXPECTED_V10_COMMIT}" ]; then
    echo "V10 SOURCE GATE FAILED: execution HEAD ${observed_commit} != ${EXPECTED_V10_COMMIT}" >&2
    exit 1
  fi
  if [ -n "$(git -C "${REPO}" status --porcelain)" ]; then
    echo "V10 SOURCE GATE FAILED: execution checkout is dirty" >&2
    git -C "${REPO}" status --short >&2
    exit 1
  fi
  echo "Execution:    ${observed_commit} (pinned)"
fi

# ── per-rank environment (srun propagates exported vars) ─────────────────
# CLUSTER: Slinky worker pods do not mount the login-node home. Point every
# runtime consumer at a dedicated writable location instead of changing HOME,
# which is process-global and can redirect unrelated libraries unexpectedly.
if [ -n "${LUSTRE_HOME:-}" ]; then
  FASTVIDEO_SHARED_RUNTIME_ROOT="${LUSTRE_HOME}"
  FASTVIDEO_JOB_CACHE_ROOT="/tmp/fastvideo-${SLURM_JOB_ID}"
  export FASTVIDEO_SHARED_RUNTIME_ROOT FASTVIDEO_JOB_CACHE_ROOT
  export HF_HOME="${HF_HOME:-${FASTVIDEO_SHARED_RUNTIME_ROOT}/.cache/huggingface}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FASTVIDEO_JOB_CACHE_ROOT}/xdg-cache}"
  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${FASTVIDEO_JOB_CACHE_ROOT}/xdg-config}"
  export TORCH_HOME="${TORCH_HOME:-${FASTVIDEO_JOB_CACHE_ROOT}/torch}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${FASTVIDEO_JOB_CACHE_ROOT}/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${FASTVIDEO_JOB_CACHE_ROOT}/torchinductor}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${FASTVIDEO_JOB_CACHE_ROOT}/torch-extensions}"
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${FASTVIDEO_JOB_CACHE_ROOT}/flashinfer}"
  export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${FASTVIDEO_JOB_CACHE_ROOT}/cuda}"
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${FASTVIDEO_JOB_CACHE_ROOT}/numba}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${FASTVIDEO_SHARED_RUNTIME_ROOT}/.config/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${FASTVIDEO_SHARED_RUNTIME_ROOT}/.cache/wandb}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${FASTVIDEO_SHARED_RUNTIME_ROOT}/.local/share/wandb}"
  export NETRC="${NETRC:-${FASTVIDEO_SHARED_RUNTIME_ROOT}/.netrc}"
  mkdir -p "${WANDB_CONFIG_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
fi
export FASTVIDEO_FA4=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
# Node-local weight cache: relaunches rebuild all three 33B roles from tmpfs
# in seconds. PER_NODE=1 makes every tray write its own /dev/shm copy
# (single-writer semantics would leave 7 of 8 trays permanently cold).
export FASTVIDEO_WEIGHT_SHARD_CACHE=/dev/shm/fastvideo-wcache
export FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE=1
# W&B: online v10 uses the explicit shared NETRC above; flip to offline if
# compute nodes have no egress.
export WANDB_MODE="${WANDB_MODE:-online}"
# CLUSTER: NCCL fabric tuning. These three come from the cluster onboarding
# guide's multi-node NCCL-test recipe (GB200 NVL72 + CX7 IB cross-rack):
# MNNVL is multi-node NVLink within a rack's NVL72 domain.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-2}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-eth0}"
# WARN costs nothing and turns silent transport hangs at communicator init
# into named errors in the log.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export NNODES MASTER_ADDR MASTER_PORT
export NUM_GPUS="${GPUS_PER_NODE}"

# V10-only hardware gate. Generic launches are unchanged when the opt-in is
# unset. The new kernel prefix must precede the FA4 overlay because that
# overlay contains an older fastvideo_kernel copy; reversing the order would
# silently resurrect the pre-#1719/#1730 implementation.
H3_V10_KERNEL_GATE="${H3_V10_KERNEL_GATE:-0}"
if [ "${H3_V10_KERNEL_GATE}" = "1" ]; then
  H3_V10_KERNEL_PREFIX="${H3_V10_KERNEL_PREFIX:-/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix}"
  H3_V10_FA4_OVERLAY="${H3_V10_FA4_OVERLAY:-/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay}"
  H3_V10_CUTLASS_PACKAGES="${H3_V10_CUTLASS_PACKAGES:-${H3_V10_FA4_OVERLAY}/nvidia_cutlass_dsl/python_packages}"
  EXPECTED_V10_PYTHONPATH="${H3_V10_KERNEL_PREFIX}:${H3_V10_FA4_OVERLAY}:${H3_V10_CUTLASS_PACKAGES}"
  if [ "${PYTHONPATH:-}" != "${EXPECTED_V10_PYTHONPATH}" ]; then
    echo "V10 KERNEL GATE FAILED: PYTHONPATH must be exactly ${EXPECTED_V10_PYTHONPATH}" >&2
    echo "Observed: ${PYTHONPATH:-<unset>}" >&2
    exit 1
  fi
  export REPO VENV H3_V10_KERNEL_PREFIX H3_V10_FA4_OVERLAY H3_V10_CUTLASS_PACKAGES
  echo "=== v10 kernel gate: real Triton gradients + sm_100a route (head tray) ==="
  if ! srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)" \
    bash "${REPO}/scripts/train/gate_h3_v10_kernel.sh"; then
    echo "V10 KERNEL GATE FAILED — aborting before training."
    exit 1
  fi
  echo "=== v10 kernel gate passed ==="
fi

# Boot watchdog: NCCL communicator init can wedge silently (all ranks parked
# in ib_uverbs_event_read/poll, no WARN output, GPUs at 0%). If no weight
# loading has been logged within BOOT_TIMEOUT_MIN minutes, abort the job so
# the allocation isn't squatted on. 0 disables.
BOOT_TIMEOUT_MIN="${BOOT_TIMEOUT_MIN:-30}"
if [ "${BOOT_TIMEOUT_MIN}" -gt 0 ]; then
  (
    sleep $((BOOT_TIMEOUT_MIN * 60))
    # Either sentinel proves the boot is alive: the lustre load path logs
    # "Loading weights took"; the warm /dev/shm cache path skips it, so also
    # accept visible training progress (job 2373 was killed at step 2).
    if ! grep -qsE "Loading weights took|Steps: " "${REPO}/examples/train/logs/${SLURM_JOB_ID}-node0"/*.log; then
      echo "BOOT WATCHDOG: no weight loading after ${BOOT_TIMEOUT_MIN} min — aborting job ${SLURM_JOB_ID}"
      scancel "${SLURM_JOB_ID}"
    fi
  ) &
  WATCHDOG_PID=$!
fi

# Launch gate (HANDOFF-h3-dmd2-vsa.md): run the two-GPU grouped-FSDP
# mixed-dtype test on the head tray before committing the allocation to
# training. GATE_TEST=0 skips (e.g. requeues of an already-gated lineage).
GATE_TEST="${GATE_TEST:-1}"
if [ "${GATE_TEST}" = "1" ]; then
  echo "=== launch gate: grouped-FSDP mixed-dtype test (head tray) ==="
  if ! srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)" bash -c "
    set -euo pipefail
    source '${VENV}/bin/activate'
    cd '${REPO}'
    '${VENV}/bin/python' -m pytest -q tests/local_tests/models/test_fsdp_load_mixed_dtype.py \
      -k declared_fp32_group_distributed_forward_backward
    '${VENV}/bin/python' -m pytest -q fastvideo/tests/train/methods/test_dmd2_fake_score_loss_space.py \
      fastvideo/tests/train/methods/test_dmd2_rollout_carry.py \
      fastvideo/tests/train/methods/test_dmd2_data_forcing.py \
      fastvideo/tests/train/methods/test_minimax_h3_dmd2.py \
      fastvideo/tests/attention/test_vsa_h3_metadata.py \
      fastvideo/tests/attention/test_flash_attn_h3_packed_varlen.py \
      fastvideo/tests/dataset/test_exact_shape_bucket_sampler.py \
      fastvideo/tests/train/callbacks/test_latent_vis_shape_context.py \
      fastvideo/tests/train/callbacks/test_validation.py \
      fastvideo/tests/train/utils/test_checkpoint.py \
      fastvideo/tests/train/utils/test_config.py \
      fastvideo/tests/train/utils/test_inference_checkpoint.py
  "; then
    echo 'LAUNCH GATE FAILED — aborting before training.'
    exit 1
  fi
  echo "=== launch gate passed ==="
fi

# CLUSTER: if trays require a container, wrap the srun payload accordingly,
# e.g. add `--container-image=... --container-mounts=/mnt:/mnt` (pyxis) so
# /mnt (repo, models, data) and /dev/shm are available inside.
srun --kill-on-bad-exit=1 --ntasks-per-node=1 bash -c "
  set -euo pipefail
  export NODE_RANK=\${SLURM_NODEID}
  if [ "\${H3_V10_COMPILE_LOGS:-0}" = "1" ] && [ "\${SLURM_NODEID}" = "0" ]; then
    export TORCH_LOGS="recompiles,graph_breaks"
  fi
  # One log dir per node — parallel tee writers on one NFS file garble it.
  export LOG_DIR=examples/train/logs/\${SLURM_JOB_ID}-node\${SLURM_NODEID}
  source '${VENV}/bin/activate'
  cd '${REPO}'
  bash examples/train/run.sh '${CONFIG}' \
    --training.distributed.num_gpus ${WORLD_SIZE} \
    --training.distributed.hsdp_replicate_dim ${HSDP_REPLICATE} \
    --training.distributed.hsdp_shard_dim ${HSDP_SHARD} \
    --training.distributed.sp_size ${SP_SIZE}
"
SRUN_EXIT=$?
[ -n "${WATCHDOG_PID:-}" ] && kill "${WATCHDOG_PID}" 2>/dev/null
exit "${SRUN_EXIT}"
