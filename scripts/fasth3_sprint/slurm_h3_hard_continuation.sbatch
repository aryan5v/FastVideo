#!/usr/bin/env bash
#SBATCH --job-name=h3-hard-recovery
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
: "${SPRINT_ROOT:?}"
: "${CODE_ROOT:?Immutable tested snapshot required}"
: "${RESUME_CHECKPOINT:?Explicit checkpoint required}"
: "${REVIEW_RECEIPT:?Decoded review required before continuation}"
: "${EXPERIMENT_ID:?Fresh run name required}"
TARGET_STEPS="${TARGET_STEPS:-500}"
[[ "${EXPERIMENT_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case "${TARGET_STEPS}" in 500|1000|2000|4000) ;; *) exit 2 ;; esac
export TARGET_STEPS RESUME_CHECKPOINT REVIEW_RECEIPT
python3 - <<'PY'
import json, os
from pathlib import Path
c = Path(os.environ['RESUME_CHECKPOINT']).resolve()
r = json.loads(Path(os.environ['REVIEW_RECEIPT']).read_text())
m = json.loads((c / 'metadata.json').read_text())
expected = {500: 200, 1000: 500, 2000: 1000, 4000: 2000}[int(os.environ['TARGET_STEPS'])]
assert m['step'] == expected, 'resume from the preceding reviewed milestone'
assert Path(r['checkpoint']).resolve() == c
assert r['continue_recovery'] is True and r['media_review_completed'] is True
assert (c / 'dcp/.metadata').is_file() and len(list(c.glob('rng_state_rank*.pt'))) == 4
assert m['config']['method']['pruning_mode'] == 'hard'
assert m['config']['method']['denoising_weight'] == 0
assert m['config']['training']['dit_precision'] == 'fp32'
assert m['config']['training']['data']['num_frames'] == 124
PY
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/day1-mask/${EXPERIMENT_ID}"
export RUN_ID="hard-continue-${EXPERIMENT_ID}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
cp "${CODE_ROOT}/CODE_COMMIT" "${OUTPUT_ROOT}/CODE_COMMIT"
cp "${REVIEW_RECEIPT}" "${OUTPUT_ROOT}/parent_review.json"
cp "${SPRINT_ROOT}/data/day1-mask-split/split_receipt.json" "${OUTPUT_ROOT}/split_receipt.json"
export SPRINT_ROOT CODE_ROOT
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}" WANDB_RUN_ID="${RUN_ID}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m fastvideo.train.entrypoint.train --config "${CODE_ROOT}/examples/train/configs/fasth3_mask_recovery_pilot.yaml" \
  --method.pruning_mode hard --method.validation_every 50 \
  --training.loop.max_train_steps "${TARGET_STEPS}" \
  --training.checkpoint.resume_from_checkpoint "${RESUME_CHECKPOINT}" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 250 \
  --training.checkpoint.preserve_every_steps 250 \
  --training.tracker.run_name "${RUN_ID}"
'
