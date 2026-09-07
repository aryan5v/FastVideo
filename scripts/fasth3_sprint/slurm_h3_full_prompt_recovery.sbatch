#!/usr/bin/env bash
#SBATCH --job-name=h3-full-prompt-recovery
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
: "${CODE_ROOT:?Immutable snapshot required}"
export CODE_ROOT
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export INDEX_ROOT=${SPRINT_ROOT}/data/user-prompts-20260906/index-v1
export RUN_ROOT=${SPRINT_ROOT}/runs/full-prompts/hard200-to500-v1
export PARENT=${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200
python3 - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['INDEX_ROOT']) / 'receipt.json').read_text())
assert r['requested'] == r['matched'] == r['validated_embeddings'] == 58669
assert r['metadata_match_complete'] and r['embedding_values_validated']
assert r.get('encoder_provenance_verified') is True, 'Verify cached embeddings against pinned H3 encoder first' 
assert not Path(os.environ['RUN_ROOT']).exists(), 'Fresh run destination required'
PY
mkdir -p "${RUN_ROOT}"
cp "${CODE_ROOT}/CODE_COMMIT" "${RUN_ROOT}/CODE_COMMIT"
cp "${INDEX_ROOT}/receipt.json" "${RUN_ROOT}/data_receipt.json"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${RUN_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
RESUME="$PARENT"
# Two resumed updates prove real checkpoint/optimizer loading and native-shape training.
# Only a successful preflight permits the bounded continuation in the second process.
for TARGET in 202 500; do
 if [ "$TARGET" = 202 ]; then RESET=true; SAVE=2; else RESET=false; SAVE=50; fi
 export WANDB_RUN_ID="h3-full-prompts-${SLURM_JOB_ID}-${TARGET}"
 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${CODE_ROOT}/examples/train/configs/fasth3_mask_recovery_pilot.yaml" \
  --method.pruning_mode hard --method.validation_every 50 \
  --training.data.data_path "$INDEX_ROOT" \
  --training.loop.max_train_steps "$TARGET" \
  --training.checkpoint.resume_from_checkpoint "$RESUME" \
  --training.checkpoint.reset_dataloader_on_resume "$RESET" \
  --training.checkpoint.output_dir "${RUN_ROOT}/phase-${TARGET}" \
  --training.checkpoint.training_state_checkpointing_steps "$SAVE" \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "$WANDB_RUN_ID"
 RESUME="${RUN_ROOT}/phase-${TARGET}/checkpoint-${TARGET}"
 test -s "$RESUME/dcp/.metadata"
 done
date -Is > "${RUN_ROOT}/training_completed_at.txt"
'
