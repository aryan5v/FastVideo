#!/usr/bin/env bash
#SBATCH --job-name=h3-mask-pilot
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
: "${SPRINT_ROOT:?}"
: "${CODE_ROOT:?Immutable code snapshot required}"
: "${PRUNING_MODE:?hard or annealed_skip}"
: "${EXPERIMENT_ID:?Unique experiment identifier required}"
TARGET_STEPS="${TARGET_STEPS:-2}"
[[ "${EXPERIMENT_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case "${PRUNING_MODE}" in hard|annealed_skip) ;; *) exit 2 ;; esac
case "${TARGET_STEPS}" in 2|25|200) ;; *) echo 'Bounded gates: 2, 25 or 200 steps only' >&2; exit 2 ;; esac
[[ "${SLURM_JOB_NUM_NODES}" == 1 ]] || exit 2
CONFIG_PATH="${CODE_ROOT}/examples/train/configs/fasth3_mask_recovery_pilot.yaml"
SPLIT_ROOT="${SPRINT_ROOT}/data/day1-mask-split"
OUTPUT_ROOT="${SPRINT_ROOT}/runs/day1-mask/${EXPERIMENT_ID}"
RUN_ID="day1-mask-${EXPERIMENT_ID}"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
export CONFIG_PATH SPLIT_ROOT OUTPUT_ROOT RUN_ID LOG_DIR
if [[ -e "${OUTPUT_ROOT}" ]]; then echo 'Output already exists; use a new experiment ID' >&2; exit 2; fi
if (( TARGET_STEPS > 2 )); then
  : "${FINITE_GATE_RECEIPT:?A successful finite gate is required}"
  python3 - "${FINITE_GATE_RECEIPT}" "${PRUNING_MODE}" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
assert receipt['passed'] is True and receipt['mode'] == sys.argv[2]
assert receipt['fp32_optimizer_update'] is True and receipt['finite_backward'] is True
PY
fi
python3 - "${SPLIT_ROOT}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); receipt = json.loads((root / 'split_receipt.json').read_text())
assert receipt['normalized_caption_id_text_overlap'] == 0
for split, view in receipt['views'].items():
    assert hashlib.sha256((root / split / 'manifest_rank0.jsonl').read_bytes()).hexdigest() == view['manifest_sha256']
PY
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
cp "${CODE_ROOT}/CODE_COMMIT" "${OUTPUT_ROOT}/CODE_COMMIT"
cp "${SPLIT_ROOT}/split_receipt.json" "${OUTPUT_ROOT}/split_receipt.json"
export PRUNING_MODE TARGET_STEPS CODE_ROOT SPRINT_ROOT
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="${CODE_ROOT}" bash -lc '
  set -euo pipefail
  source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
  export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
  export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
  export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${LOG_DIR}" WANDB_RUN_ID="${RUN_ID}"
  export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0 TOKENIZERS_PARALLELISM=false
  export TORCH_NCCL_ENABLE_MONITORING=0
  export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}"
  mkdir -p "${TRITON_CACHE_DIR}"
  SAVE_EVERY=25; VALIDATE_EVERY=25; ANNEAL_STEPS=100
  if [[ "${TARGET_STEPS}" == 2 ]]; then SAVE_EVERY=2; VALIDATE_EVERY=2; ANNEAL_STEPS=1; fi
  exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m fastvideo.train.entrypoint.train --config "${CONFIG_PATH}" \
    --method.pruning_mode "${PRUNING_MODE}" --method.mask_anneal_steps "${ANNEAL_STEPS}" \
    --method.validation_every "${VALIDATE_EVERY}" \
    --training.loop.max_train_steps "${TARGET_STEPS}" \
    --training.checkpoint.training_state_checkpointing_steps "${SAVE_EVERY}" \
    --training.checkpoint.output_dir "${OUTPUT_ROOT}" --training.tracker.run_name "${RUN_ID}"
'
