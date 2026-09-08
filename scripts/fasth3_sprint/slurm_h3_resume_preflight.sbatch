#!/bin/bash
#SBATCH --job-name=h3-resume-proof
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export CODE_ROOT="${PWD:?batch working directory is required}"
export PARENT=${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200
export OUTPUT_ROOT=${SPRINT_ROOT}/runs/resume-proof/job-${SLURM_JOB_ID}
for required in "${CODE_ROOT}/CODE_COMMIT" "${PARENT}/dcp/.metadata" "${PARENT}/metadata.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "required file is missing or empty: ${required}" >&2
    exit 1
  fi
done
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
cp "${CODE_ROOT}/CODE_COMMIT" "${OUTPUT_ROOT}/CODE_COMMIT"

srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m fastvideo.train.entrypoint.train \
  --config "${CODE_ROOT}/examples/train/configs/fasth3_mask_recovery_pilot.yaml" \
  --method.pruning_mode hard --method.validation_every 50 \
  --training.optimizer.learning_rate 3e-5 \
  --training.loop.max_train_steps 202 \
  --training.checkpoint.resume_from_checkpoint "${PARENT}" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 2 \
  --training.checkpoint.preserve_every_steps 2 \
  --training.tracker.run_name "h3-resume-proof-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-202/dcp/.metadata"
test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name "resume_optimizer_audit_step200_rank*.json" | wc -l)" -eq 4
"${PY}" - <<PY
import json
from pathlib import Path
root = Path("${OUTPUT_ROOT}")
receipts = [json.loads(path.read_text()) for path in sorted(root.glob("resume_optimizer_audit_step200_rank*.json"))]
assert len(receipts) == 4 and all(item["passed"] for item in receipts)
assert all(item["actual_learning_rates"] == [3e-5] for item in receipts)
assert all(item["sampled_nonzero_moment_values"] > 0 for item in receipts)
(root / "resume_proof_passed.json").write_text(json.dumps({"passed": True, "receipts": receipts}, indent=2) + "\n")
PY
date -Is > "${OUTPUT_ROOT}/completed_at.txt"
'
