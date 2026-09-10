#!/bin/bash
#SBATCH --job-name=h3-b34-exp25
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
# Student-state exposure A/B on the coherent candidate: 200 prompt-only
# updates from job-7048 export-200 with student_state_probability 0.25.
# Arm B (job 7046) tried this from the half-coherent 6972 and crushed audio;
# from 7048 (WER 0, motion 0.94x) the mechanism should insure against rollout
# drift instead of teaching self-imitation. Everything else unchanged so the
# comparison against 7048/cont100 is clean. Panels at 100 and 200.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export STUDENT="${SPRINT_ROOT}/runs/base34-recut-detail-band/job-7048/export-200"
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base34-exposure25/job-${SLURM_JOB_ID}"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${STUDENT}/transformer/config.json"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" \
 --template "${CODE_ROOT}/examples/train/configs/fasth3_detail_band_recovery34.yaml" \
 --student "${STUDENT}" --depth 34 --output "${OUTPUT_ROOT}/config.yaml"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
 --config "${OUTPUT_ROOT}/config.yaml" \
 --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
 --training.loop.max_train_steps 200 \
 --training.checkpoint.training_state_checkpointing_steps 100 \
 --training.checkpoint.preserve_every_steps 100 \
 --method.allow_prompt_only true --method.denoising_weight 0.0 \
 --method.student_state_probability 0.25 \
 --training.data.data_path "${SPRINT_ROOT}/data/user-prompts-20260906/index-v2" \
 --training.tracker.run_name "base34-exposure25-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-200/dcp/.metadata"
for step in 100 200; do
 "${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-${step}" \
  --output-dir "${OUTPUT_ROOT}/export-${step}" --role student --load-precision bf16 --verify
 "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "${OUTPUT_ROOT}/export-${step}" --checkpoint-role "base34-exposure25-${step}" \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
  --output-dir "${OUTPUT_ROOT}/media-${step}" --run-id "base34-exposure25-${step}-${SLURM_JOB_ID}" \
  --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
  --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
  --profile strict --no-fa4 --no-compile --no-upload-videos
done
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
