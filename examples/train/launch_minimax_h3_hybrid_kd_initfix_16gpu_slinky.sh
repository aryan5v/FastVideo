#!/usr/bin/env bash
# Submit a fresh 16-GPU validation of the live hybrid-branch initialization.

set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
PREVIEW_ROOT=/mnt/lustre/vlm-aryan/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-b65818d
IMAGE=/mnt/lustre/vlm-wlsaidhi/fastvideo/images/fastvideo-dev-sm100-2f28adad2e05.sqsh
KERNEL_TARGET=/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/sm100a_main/prefix
CONFIG_FILE=${WORKTREE_ROOT}/examples/train/configs/overfit_minimax_h3_hybrid_kd_initfix_16gpu.yaml
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_v1_initfix_16gpu_overfit
SECRET_FILE=/mnt/lustre/vlm-aryan/.secrets/wandb_api_key

[[ -f "${PREVIEW_ROOT}/modular_model_index.json" ]] || { echo "Missing Preview model: ${PREVIEW_ROOT}" >&2; exit 1; }
[[ -f "${SECRET_FILE}" ]] || { echo "Missing W&B secret: ${SECRET_FILE}" >&2; exit 1; }
[[ -f "${IMAGE}" ]] || { echo "Missing runtime image: ${IMAGE}" >&2; exit 1; }
[[ -d "${KERNEL_TARGET}" ]] || { echo "Missing VSA kernel prefix: ${KERNEL_TARGET}" >&2; exit 1; }
[[ ! -e "${RESULT_DIR}" ]] || { echo "Refusing to overwrite existing run: ${RESULT_DIR}" >&2; exit 1; }

mkdir -p "${RESULT_DIR}/slurm"

sbatch \
  --job-name=fasth3-initfix-16g \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive \
  --chdir="${WORKTREE_ROOT}" \
  --output="${RESULT_DIR}/slurm/%x_%j.out" \
  --error="${RESULT_DIR}/slurm/%x_%j.err" \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive --chdir="${WORKTREE_ROOT}" \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive --chdir="${WORKTREE_ROOT}" \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive --chdir="${WORKTREE_ROOT}" <<'SBATCH_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
PREVIEW_ROOT=/mnt/lustre/vlm-aryan/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-b65818d
IMAGE=/mnt/lustre/vlm-wlsaidhi/fastvideo/images/fastvideo-dev-sm100-2f28adad2e05.sqsh
KERNEL_TARGET=/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/sm100a_main/prefix
CONFIG_FILE=${WORKTREE_ROOT}/examples/train/configs/overfit_minimax_h3_hybrid_kd_initfix_16gpu.yaml
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_v1_initfix_16gpu_overfit
PARQUET=${WORKTREE_ROOT}/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet

export PYTHONPATH="${KERNEL_TARGET}:${WORKTREE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH=/opt/venv/bin:${PATH}
export HF_HOME=${WORKTREE_ROOT}/.cache/huggingface
export WANDB_API_KEY="$(< /mnt/lustre/vlm-aryan/.secrets/wandb_api_key)"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FASTVIDEO_VSA_SM100A=1
export FASTVIDEO_VSA_CUTEDSL=0
export FASTVIDEO_MINIMAX_H3_FUSIONS=0

[[ -s "${PARQUET}" ]] || { echo "Missing preprocessed training sample: ${PARQUET}" >&2; exit 1; }
cd "${WORKTREE_ROOT}"

# Validate the exact loader/init/liveness path in the runtime image before the
# costly multi-node process begins.
srun --het-group=0 --nodes=1 --ntasks=1 --gres=gpu:4 \
  --container-image="${IMAGE}" --container-mounts=/mnt/lustre:/mnt/lustre \
  --container-workdir="${WORKTREE_ROOT}" \
  /opt/venv/bin/python -m pytest -q \
    fastvideo/tests/loader/test_fsdp_load_releases_checkpoint.py \
    fastvideo/tests/train/callbacks/test_minimax_h3_hybrid_liveness.py \
    fastvideo/tests/train/methods/test_minimax_h3_hybrid_kd.py \
    fastvideo/tests/transformers/test_minimax_h3_hybrid_linear.py

export MASTER_ADDR=${SLURM_JOB_NODELIST_HET_GROUP_0}
export MASTER_PORT=29543
export CONFIG_FILE RESULT_DIR
CONTROL_DIR=${RESULT_DIR}/control
RETRY_FILE=${CONTROL_DIR}/retry-${SLURM_JOB_ID}
STOP_FILE=${CONTROL_DIR}/stop-${SLURM_JOB_ID}
FAILURE_FILE=${CONTROL_DIR}/failure-${SLURM_JOB_ID}.txt
MAX_RESTART_WAIT_SECONDS=${FASTVIDEO_RESTART_WAIT_SECONDS:-7200}
mkdir -p "${CONTROL_DIR}"
rm -f "${RETRY_FILE}" "${STOP_FILE}" "${FAILURE_FILE}"

training_attempt=0
while true; do
  training_attempt=$((training_attempt + 1))
  echo "Starting training attempt ${training_attempt} in allocation ${SLURM_JOB_ID}"
  resume_value=""
  if compgen -G "${RESULT_DIR}/checkpoints/checkpoint-*" >/dev/null; then
    resume_value=latest
    echo "Resuming the corrected run from its latest checkpoint"
  fi
  export RESUME_VALUE=${resume_value}

  set +e
  srun --kill-on-bad-exit=1 --het-group=0-3 \
    --container-image="${IMAGE}" --container-mounts=/mnt/lustre:/mnt/lustre \
    --container-workdir="${WORKTREE_ROOT}" \
    bash -c '
      export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_PROCID}"
      exec /opt/venv/bin/torchrun \
        --nnodes 4 --nproc_per_node 4 --node_rank "${SLURM_PROCID}" \
        --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        fastvideo/train/entrypoint/train.py \
        --config "${CONFIG_FILE}" \
        --training.checkpoint.resume_from_checkpoint "${RESUME_VALUE}"
    '
  training_status=$?
  set -e
  if (( training_status == 0 )); then
    rm -f "${FAILURE_FILE}"
    break
  fi

  printf 'attempt=%d\nexit_code=%d\nfailed_at=%s\n' \
    "${training_attempt}" "${training_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${FAILURE_FILE}"
  echo "Training failed with exit ${training_status}; holding this allocation for an in-place fix."
  echo "Retry with: touch ${RETRY_FILE}"
  waited=0
  while (( waited < MAX_RESTART_WAIT_SECONDS )); do
    [[ ! -e "${STOP_FILE}" ]] || exit "${training_status}"
    if [[ -e "${RETRY_FILE}" ]]; then
      rm -f "${RETRY_FILE}"
      continue 2
    fi
    sleep 10
    waited=$((waited + 10))
  done
  exit "${training_status}"
done
SBATCH_SCRIPT
