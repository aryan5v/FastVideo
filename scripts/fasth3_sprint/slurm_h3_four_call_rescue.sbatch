#!/usr/bin/env bash
#SBATCH --job-name=h3-4call-rescue
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail

: "${SOURCE_KIND:?SOURCE_KIND must be dense or vsa}"
: "${WARM_START_PATH:?WARM_START_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
SYNTH_DATA_ROOT="${SYNTH_DATA_ROOT:-${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}/data/h3-rescue-synthetic-train64-124f-207a}"
TARGET_STEPS="${TARGET_STEPS:-25}"
SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
CONFIG="${REPO_ROOT}/examples/train/configs/fasth3_14b_four_call_rescue.yaml"

case "${TARGET_STEPS}" in
  25) RESUME="" ;;
  50|75|100|200) RESUME="latest" ;;
  *) echo "TARGET_STEPS must stop at a 25-step rescue gate" >&2; exit 2 ;;
esac
case "${SOURCE_KIND}" in
  dense)
    TEACHER_PATH="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
    ATTENTION_BACKEND=TORCH_SDPA
    VSA_SPARSITY=0.0
    ;;
  vsa)
    TEACHER_PATH="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/snapshots/b65818d41939b5085451074fe8ca8b799f8d4921"
    ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3
    VSA_SPARSITY=0.9
    ;;
  *) echo "SOURCE_KIND must be dense or vsa" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${SPRINT_ROOT}/runs/h36-four-call-rescue/${SOURCE_KIND}"
RUN_ID="h36-four-call-rescue-${SOURCE_KIND}-step${TARGET_STEPS}"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"
test -s "${WARM_START_PATH}/transformer/config.json"
test -d "${DATA_ROOT}"
test -d "${SYNTH_DATA_ROOT}"

python3 - "${DATA_ROOT}" "${SYNTH_DATA_ROOT}" <<'PY'
import glob
import json
import sys

records = []
for root in sys.argv[1:]:
    for path in glob.glob(root + "/manifest_rank*.jsonl"):
        with open(path, encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
assert records, "rescue data has no manifest records"
for record in records:
    assert record["num_latent_frames"] == 37, record
    assert record["num_audio_latents"] == 207, record
    assert record["latent_height"] > 0 and record["latent_width"] > 0, record
print(f"validated {len(records)} aligned rescue records")
PY

if [[ -n "${RESUME}" ]]; then
  previous="$((TARGET_STEPS - 25))"
  test -s "${OUTPUT_ROOT}/checkpoint-${previous}/metadata.json"
  test -s "${OUTPUT_ROOT}/checkpoint-${previous}/dcp/.metadata"
fi

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export WANDB_DIR="${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"

echo "$(date -Is) run=${RUN_ID} node=${SLURM_JOB_NODELIST} warm_start=${WARM_START_PATH}"
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m fastvideo.train.entrypoint.train --config "${CONFIG}" \
    --models.student.init_from "${WARM_START_PATH}" \
    --models.student.attention_backend "${ATTENTION_BACKEND}" \
    --models.teacher.init_from "${TEACHER_PATH}" \
    --models.teacher.attention_backend "${ATTENTION_BACKEND}" \
    --training.data.data_path "[${DATA_ROOT}, ${SYNTH_DATA_ROOT}]" \
    --training.loop.max_train_steps "${TARGET_STEPS}" \
    --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
    --training.checkpoint.resume_from_checkpoint "${RESUME}" \
    --training.tracker.run_name "${RUN_ID}" \
    --training.vsa.sparsity "${VSA_SPARSITY}"
