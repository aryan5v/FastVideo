#!/usr/bin/env bash
#SBATCH --job-name=h3-bf16-recovery
#SBATCH --partition=all
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${SOURCE_KIND:?SOURCE_KIND is required}"
: "${MAP_STRATEGY:?MAP_STRATEGY is required}"
: "${TARGET_STEPS:?TARGET_STEPS is required}"
RUN_MODE="${RUN_MODE:-long}"

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
    echo "SOURCE_KIND must be dense or vsa" >&2
    exit 2
    ;;
esac
case "${MAP_STRATEGY}" in
  activation|uniform) ;;
  *)
    echo "MAP_STRATEGY must be activation or uniform" >&2
    exit 2
    ;;
esac
case "${RUN_MODE}" in
  finite)
    case "${TARGET_STEPS}" in
      1)
        RESUME_ARGS=""
        ;;
      2)
        RESUME_ARGS="--training.checkpoint.resume_from_checkpoint latest"
        ;;
      *)
        echo "finite recovery TARGET_STEPS must be 1 or 2" >&2
        exit 2
        ;;
    esac
    ;;
  long)
    case "${TARGET_STEPS}" in
      200)
        RESUME_ARGS=""
        ;;
      400|600|800|1000)
        RESUME_ARGS="--training.checkpoint.resume_from_checkpoint latest"
        ;;
      *)
        echo "long recovery TARGET_STEPS must be 200, 400, 600, 800, or 1000" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "RUN_MODE must be finite or long" >&2
    exit 2
    ;;
esac

REPO_ROOT="${SPRINT_ROOT}/repo"
CONFIG_PATH="${REPO_ROOT}/examples/train/configs/fasth3_14b_recovery.yaml"
CANDIDATE_ROOT="${SPRINT_ROOT}/checkpoints/h18-candidates/${SOURCE_KIND}-${MAP_STRATEGY}"
STEP_GATE_ROOT="${SPRINT_ROOT}/runs/h18-step-gates/${SOURCE_KIND}-${MAP_STRATEGY}"
if [[ "${RUN_MODE}" == "finite" ]]; then
  OUTPUT_ROOT="${SPRINT_ROOT}/runs/h18-recovery-gates/${SOURCE_KIND}-${MAP_STRATEGY}"
  RUN_ID="h18-recovery-gate-${SOURCE_KIND}-${MAP_STRATEGY}-step${TARGET_STEPS}"
  CHECKPOINT_ARGS="--training.checkpoint.training_state_checkpointing_steps 1"
else
  OUTPUT_ROOT="${SPRINT_ROOT}/runs/h36-recovery/${SOURCE_KIND}-${MAP_STRATEGY}"
  RUN_ID="h36-recovery-${SOURCE_KIND}-${MAP_STRATEGY}-step${TARGET_STEPS}"
  CHECKPOINT_ARGS=""
fi
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
SYNC_REVIEW="${SPRINT_ROOT}/eval/h0-vsa-sync-gate/00_speech_exact_presenter_review_manifest.json"
ASR_REVIEW="${SPRINT_ROOT}/eval/h0-vsa-sync-gate/asr_review.json"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"
test -s "${CANDIDATE_ROOT}/transformer/candidate_validation_receipt.json"
test -s "${STEP_GATE_ROOT}/checkpoint-2/metadata.json"
test -d "${STEP_GATE_ROOT}/checkpoint-2/dcp"
test -s "${SYNC_REVIEW}"
test -s "${ASR_REVIEW}"
python3 - "${SYNC_REVIEW}" "${ASR_REVIEW}" <<'PY'
import json
import sys

media = json.load(open(sys.argv[1], encoding="utf-8"))
asr = json.load(open(sys.argv[2], encoding="utf-8"))
assert media["decoded_video_frames"] == 124
assert media["channels"] == 2 and media["sample_rate"] == 32000
assert media["duration_seconds"] > 5.0
assert asr["passed"] is True and asr["word_error_rate"] <= asr["max_word_error_rate"]
PY

if [[ "${RUN_MODE}" == "finite" && "${TARGET_STEPS}" == "2" ]]; then
  test -s "${OUTPUT_ROOT}/checkpoint-1/metadata.json"
  test -d "${OUTPUT_ROOT}/checkpoint-1/dcp"
elif [[ "${RUN_MODE}" == "long" && "${TARGET_STEPS}" != "200" ]]; then
  previous="$((TARGET_STEPS - 200))"
  test -s "${OUTPUT_ROOT}/checkpoint-${previous}/metadata.json"
  test -d "${OUTPUT_ROOT}/checkpoint-${previous}/dcp"
fi
if [[ "${RUN_MODE}" == "long" ]]; then
  # The activation candidate is the backend-specific recovery canary. Every
  # map-specific candidate already passed its own optimizer/resume gate; the
  # recovery method path varies by Dense versus VSA, not by block indices.
  RECOVERY_GATE_ROOT="${SPRINT_ROOT}/runs/h18-recovery-gates/${SOURCE_KIND}-activation"
  test -s "${RECOVERY_GATE_ROOT}/checkpoint-2/metadata.json"
  test -d "${RECOVERY_GATE_ROOT}/checkpoint-2/dcp"
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
FASTH3_TOTAL_GPUS="$((SLURM_JOB_NUM_NODES * 4))"
if [[ "${RUN_MODE}" == "finite" ]]; then
  if (( FASTH3_TOTAL_GPUS != 4 )); then
    echo "Finite FastH3 recovery gate requires exactly 4 GPUs" >&2
    exit 4
  fi
elif (( FASTH3_TOTAL_GPUS < 12 || FASTH3_TOTAL_GPUS > 32 || FASTH3_TOTAL_GPUS % 4 != 0 )); then
  echo "Long FastH3 recovery requires 12 through 32 GPUs and SP=4 divisibility" >&2
  exit 4
fi
export REPO_ROOT CONFIG_PATH CANDIDATE_ROOT OUTPUT_ROOT RUN_ID LOG_DIR SPRINT_ROOT
export ATTENTION_BACKEND VSA_SPARSITY TARGET_STEPS RESUME_ARGS FASTH3_TOTAL_GPUS RUN_MODE CHECKPOINT_ARGS

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
    echo "$(date -Is) job=${SLURM_JOB_ID} node_rank=${SLURM_PROCID} total_gpus=${FASTH3_TOTAL_GPUS} commit=$(git -C "${REPO_ROOT}" rev-parse HEAD) backend=${ATTENTION_BACKEND} run_mode=${RUN_MODE} target_steps=${TARGET_STEPS}"
    exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      -m torch.distributed.run \
      --nnodes "${SLURM_JOB_NUM_NODES}" \
      --nproc_per_node 4 \
      --node_rank "${SLURM_PROCID}" \
      --rdzv_backend c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      -m fastvideo.train.entrypoint.train \
      --config "${CONFIG_PATH}" \
      --models.student.init_from "${CANDIDATE_ROOT}" \
      --models.student.attention_backend "${ATTENTION_BACKEND}" \
      --training.distributed.num_gpus "${FASTH3_TOTAL_GPUS}" \
      --training.distributed.hsdp_replicate_dim 1 \
      --training.distributed.hsdp_shard_dim "${FASTH3_TOTAL_GPUS}" \
      --training.loop.max_train_steps "${TARGET_STEPS}" \
      --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
      --training.tracker.run_name "${RUN_ID}" \
      --training.vsa.sparsity "${VSA_SPARSITY}" \
      ${CHECKPOINT_ARGS} \
      ${RESUME_ARGS}
  '
