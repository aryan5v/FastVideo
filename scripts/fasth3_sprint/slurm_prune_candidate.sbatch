#!/usr/bin/env bash
#SBATCH --job-name=h3-prune-candidate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${SOURCE_KIND:?SOURCE_KIND is required}"
: "${MAP_STRATEGY:?MAP_STRATEGY is required}"

case "${SOURCE_KIND}" in
  dense)
    MODEL_ID="FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
    SOURCE_REVISION="f624f08c6c279ab43534c003e556fc5b295b6558"
    SOURCE_ROOT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/${SOURCE_REVISION}/transformer"
    ;;
  vsa)
    MODEL_ID="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
    SOURCE_REVISION="b65818d41939b5085451074fe8ca8b799f8d4921"
    SOURCE_ROOT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/snapshots/${SOURCE_REVISION}/transformer"
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

REPO_ROOT="${SPRINT_ROOT}/repo"
MAP_PATH="${SPRINT_ROOT}/manifests/h6-block-score/${MAP_STRATEGY}_20_block_map.json"
OUTPUT_ROOT="${SPRINT_ROOT}/checkpoints/h18-candidates/${SOURCE_KIND}-${MAP_STRATEGY}"
OUTPUT_TRANSFORMER="${OUTPUT_ROOT}/transformer"
RUN_ID="h18-prune-${SOURCE_KIND}-${MAP_STRATEGY}-20b"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

test -s "${MAP_PATH}"
test -s "${SOURCE_ROOT}/diffusion_pytorch_model.safetensors.index.json"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export PYTHONPATH=${REPO_ROOT}:${SPRINT_ROOT}/python-packages
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_DIR}
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      scripts/checkpoint_conversion/prune_minimax_h3_blocks.py \
      --src '${SOURCE_ROOT}' \
      --dst '${OUTPUT_TRANSFORMER}' \
      --block-map '${MAP_PATH}' \
      --strategy '${MAP_STRATEGY}_20_block_${SOURCE_KIND}_v1' \
      --source-model '${MODEL_ID}' \
      --source-revision '${SOURCE_REVISION}' \
      --wandb-project fasth3-14b-2step-qad-sprint \
      --wandb-run-id '${RUN_ID}'
  "

