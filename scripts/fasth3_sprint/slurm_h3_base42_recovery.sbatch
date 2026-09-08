#!/bin/bash
#SBATCH --job-name=h3-base42-recover
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base42-full-schedule/job-${SLURM_JOB_ID}"
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export STUDENT="${SPRINT_ROOT}/checkpoints/base42-uniform-v1"
export BASE_GATE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${STUDENT}/transformer/block_map_manifest.json"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/assemble_h3_stage.py" --base "${BASE}" --student "${STUDENT}"
# The step-zero panel is preserved from6869; do not regenerate it here.
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/check_h3_base_gate.py" "${BASE_GATE}"
# Save step 2 before committing further compute; resume exact model and Adam state.
for TARGET in 2 200; do
 RESUME=()
 if [[ "${TARGET}" == 200 ]]; then
  "${PY}" -c "import json,pathlib; r=pathlib.Path(\"${OUTPUT_ROOT}\"); a=[json.loads((r/f\"base_update_rank{i}.json\").read_text()) for i in range(4)]; assert all(x[\"passed\"] for x in a); assert (r/\"checkpoint-2/dcp/.metadata\").is_file()"
  RESUME=(--training.checkpoint.resume_from_checkpoint "${OUTPUT_ROOT}/checkpoint-2")
 fi
 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${CODE_ROOT}/examples/train/configs/fasth3_base42_recovery.yaml" \
  --training.loop.max_train_steps "${TARGET}" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "base42-full-schedule-${SLURM_JOB_ID}-${TARGET}" "${RESUME[@]}"
 test -s "${OUTPUT_ROOT}/checkpoint-${TARGET}/dcp/.metadata"
done
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-200" \
 --output-dir "${OUTPUT_ROOT}/export-200" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-200" --checkpoint-role base42-recovered200 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-200" --run-id "base42-200-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
