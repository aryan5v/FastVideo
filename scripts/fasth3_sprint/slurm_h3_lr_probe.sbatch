#!/bin/bash
#SBATCH --job-name=h3-lr-probe
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
LEARNING_RATE="${1:?first argument must be 3e-5 or 1e-4}"
ARM="${2:?second argument must be lr3e5 or lr1e4}"
export LEARNING_RATE ARM
case "${ARM}:${LEARNING_RATE}" in
  lr3e5:3e-5|lr1e4:1e-4) ;;
  *) echo "invalid LR arm: ${ARM}:${LEARNING_RATE}" >&2; exit 2 ;;
esac
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export CODE_ROOT="${PWD:?batch working directory is required}"
export PARENT=${SPRINT_ROOT}/runs/day1-mask/dense24-hard-pilot200-v1/checkpoint-200
export RUN_ROOT=${SPRINT_ROOT}/runs/lr-probes/${ARM}-job-${SLURM_JOB_ID}
export FULL_EXPORT=${SPRINT_ROOT}/exports/lr-probes/${ARM}-job-${SLURM_JOB_ID}-full-bf16
export COMPACT_EXPORT=${SPRINT_ROOT}/exports/lr-probes/${ARM}-job-${SLURM_JOB_ID}-compact-bf16
export AUDIT_ROOT=${SPRINT_ROOT}/diagnostics/lr-probes/${ARM}-job-${SLURM_JOB_ID}
for required in "${CODE_ROOT}/CODE_COMMIT" "${PARENT}/dcp/.metadata" "${PARENT}/metadata.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "required file is missing or empty: ${required}" >&2
    exit 1
  fi
done
for fresh in "${RUN_ROOT}" "${FULL_EXPORT}" "${COMPACT_EXPORT}" "${AUDIT_ROOT}"; do
  test ! -e "${fresh}"
done
mkdir -p "${RUN_ROOT}" "${AUDIT_ROOT}"
cp "${CODE_ROOT}/CODE_COMMIT" "${RUN_ROOT}/CODE_COMMIT"

srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${RUN_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m fastvideo.train.entrypoint.train \
  --config "${CODE_ROOT}/examples/train/configs/fasth3_mask_recovery_pilot.yaml" \
  --method.pruning_mode hard --method.validation_every 50 \
  --training.optimizer.learning_rate "${LEARNING_RATE}" \
  --training.loop.max_train_steps 500 \
  --training.checkpoint.resume_from_checkpoint "${PARENT}" \
  --training.checkpoint.output_dir "${RUN_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "h3-${ARM}-${SLURM_JOB_ID}"
CHECKPOINT="${RUN_ROOT}/checkpoint-500"
test -s "${CHECKPOINT}/dcp/.metadata"
test "$(find "${RUN_ROOT}" -maxdepth 1 -name "resume_optimizer_audit_step200_rank*.json" | wc -l)" -eq 4

CUDA_VISIBLE_DEVICES=0 "${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "${CHECKPOINT}" --output-dir "${FULL_EXPORT}" --role student \
  --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/extract_h3_mask_export.py" \
  --checkpoint "${CHECKPOINT}" --full-export "${FULL_EXPORT}" --compact-export "${COMPACT_EXPORT}"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/validate_h3_candidate.py" \
  --transformer "${COMPACT_EXPORT}/transformer" --expected-source-kind dense --expected-layers 24
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "${COMPACT_EXPORT}" --checkpoint-role "hard24-${ARM}-step500" \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
  --output-dir "${AUDIT_ROOT}/media-bf16" --run-id "hard24-${ARM}-media-${SLURM_JOB_ID}" \
  --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 \
  --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 \
  --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
ffmpeg -hide_banner -loglevel error -y \
  -i "${AUDIT_ROOT}/media-bf16/00_speech_exact_presenter.mp4" \
  -vn -ac 2 -ar 32000 -c:a pcm_s16le "${AUDIT_ROOT}/speech.wav"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/verify_speech_asr.py" \
  --wav "${AUDIT_ROOT}/speech.wav" \
  --expected "Fast video and clear audio arrive together." \
  --output "${AUDIT_ROOT}/speech_wer.json" \
  --run-id "hard24-${ARM}-asr-${SLURM_JOB_ID}" --max-wer 100
date -Is > "${AUDIT_ROOT}/completed_at.txt"
'
