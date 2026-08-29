#!/usr/bin/env bash
#SBATCH --job-name=h3-candidate-step-gate
#SBATCH --partition=all
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -euo pipefail
# Jobs are intentionally submitted with --export=NIL so login-node state
# cannot leak into training. Re-enable propagation only for variables that
# this batch script validates and exports below.
export SLURM_EXPORT_ENV=ALL

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${SOURCE_KIND:?SOURCE_KIND is required}"
: "${MAP_STRATEGY:?MAP_STRATEGY is required}"
: "${GATE_STAGE:?GATE_STAGE is required}"

case "${SOURCE_KIND}" in
  dense)
    ATTENTION_BACKEND=TORCH_SDPA
    VSA_SPARSITY=0.0
    ;;
  vsa)
    ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3
    VSA_SPARSITY=0.9
    ;;
  *)
    echo "SOURCE_KIND must be dense or vsa, got ${SOURCE_KIND}" >&2
    exit 2
    ;;
esac
case "${MAP_STRATEGY}" in
  activation|uniform) ;;
  *)
    echo "MAP_STRATEGY must be activation or uniform, got ${MAP_STRATEGY}" >&2
    exit 2
    ;;
esac
case "${GATE_STAGE}" in
  fresh)
    MAX_STEPS=1
    RESUME_ARGS=""
    ;;
  resume)
    MAX_STEPS=2
    RESUME_ARGS="--training.checkpoint.resume_from_checkpoint latest"
    ;;
  *)
    echo "GATE_STAGE must be fresh or resume, got ${GATE_STAGE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="${SPRINT_ROOT}/repo"
CONFIG_PATH="${REPO_ROOT}/examples/train/configs/fasth3_14b_candidate_step_gate.yaml"
CANDIDATE_ROOT="${SPRINT_ROOT}/checkpoints/h18-candidates/${SOURCE_KIND}-${MAP_STRATEGY}"
OUTPUT_ROOT="${SPRINT_ROOT}/runs/h18-step-gates/${SOURCE_KIND}-${MAP_STRATEGY}"
RUN_ID="h18-${GATE_STAGE}-${SOURCE_KIND}-${MAP_STRATEGY}-20b"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"
test -s "${CANDIDATE_ROOT}/transformer/block_map_manifest.json"
test -s "${CANDIDATE_ROOT}/model_index.json" -o -s "${CANDIDATE_ROOT}/modular_model_index.json"
srun \
  --export=ALL \
  --nodes=1 \
  --ntasks=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='${REPO_ROOT}:${SPRINT_ROOT}/python-packages'
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/fasth3_sprint/validate_h3_candidate.py \
      --transformer '${CANDIDATE_ROOT}/transformer' \
      --expected-source-kind '${SOURCE_KIND}' \
      > '${LOG_DIR}/candidate-validation.json'
  "
if [[ "${GATE_STAGE}" == "resume" ]]; then
  test -d "${OUTPUT_ROOT}/checkpoint-1/dcp"
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
export REPO_ROOT CONFIG_PATH CANDIDATE_ROOT OUTPUT_ROOT RUN_ID LOG_DIR
export ATTENTION_BACKEND VSA_SPARSITY MAX_STEPS RESUME_ARGS SPRINT_ROOT

srun \
  --export=ALL \
  --nodes="${SLURM_JOB_NUM_NODES}" \
  --ntasks="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc '
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
    export PYTHONDONTWRITEBYTECODE=1
    export WANDB_MODE=online
    export WANDB_DIR="${LOG_DIR}"
    export WANDB_RUN_ID="${RUN_ID}"
    export WANDB_RESUME=allow
    export FASTVIDEO_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
    export FASTVIDEO_MINIMAX_H3_FUSIONS=0
    export TOKENIZERS_PARALLELISM=false
    export TORCH_NCCL_ENABLE_MONITORING=0
    export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}/node-${SLURM_PROCID}"
    mkdir -p "${TRITON_CACHE_DIR}"
    echo "$(date -Is) job=${SLURM_JOB_ID} node_rank=${SLURM_PROCID} commit=$(git -C "${REPO_ROOT}" rev-parse HEAD) backend=${ATTENTION_BACKEND} stage=${RUN_ID}"
    exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/torchrun \
      --nnodes "${SLURM_JOB_NUM_NODES}" \
      --nproc_per_node 4 \
      --node_rank "${SLURM_PROCID}" \
      --rdzv_backend c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      -m fastvideo.train.entrypoint.train \
      --config "${CONFIG_PATH}" \
      --models.student.init_from "${CANDIDATE_ROOT}" \
      --models.student.attention_backend "${ATTENTION_BACKEND}" \
      --training.loop.max_train_steps "${MAX_STEPS}" \
      --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
      --training.tracker.run_name "${RUN_ID}" \
      --training.vsa.sparsity "${VSA_SPARSITY}" \
      ${RESUME_ARGS}
  '
