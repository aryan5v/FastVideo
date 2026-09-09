#!/bin/bash
#SBATCH --job-name=h3-detail-band
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
# Detail-band recovery (audit 2026-09-08). Usage: sbatch ... sbatch ARM STAGE
#   STAGE=34  release-candidate track: job-6972 export (activation-34 step0 + 200 prompt-58k)
#   STAGE=42  quality-tier track: preserved activation-42 step-500 export
#   ARM=A  teacher-prefix states, real paired data anchor (control)
#   ARM=B  A + student_state_probability 0.25 (exposure fix)
#   ARM=C  prompt-only 58k index with the same detail-band controls (winning-recipe + fixes)
# Trains 300 updates with low-sigma-biased interval sampling, audio velocity
# weight 4, audio seam weight 2 and (A/B) the real paired denoising branch,
# then exports, decodes the five-prompt panel under TORCH_SDPA and runs the
# fidelity + ASR gates before declaring done.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export ARM="${1:-A}"
export STAGE="${2:-34}"
case "${ARM}" in A|B|C) ;; *) echo "ARM must be A, B or C" >&2; exit 2 ;; esac
case "${STAGE}" in 34|42) ;; *) echo "STAGE must be 34 or 42" >&2; exit 2 ;; esac
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base${STAGE}-detail-band/job-${SLURM_JOB_ID}"
export REFERENCE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
if [[ "${STAGE}" == "34" ]]; then
  export CONFIG="${CODE_ROOT}/examples/train/configs/fasth3_detail_band_recovery34.yaml"
  export STUDENT="${STUDENT_OVERRIDE:-${SPRINT_ROOT}/runs/activation34-prompt58k-recovery/job-6972/export-200}"
else
  export CONFIG="${CODE_ROOT}/examples/train/configs/fasth3_detail_band_recovery.yaml"
  export STUDENT="${STUDENT_OVERRIDE:-${SPRINT_ROOT}/preserved/activation42-step500-job6878/export-500}"
fi
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${STUDENT}/transformer/block_map_manifest.json"
test -d "${REFERENCE}"
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
EXTRA_ARGS=()
case "${ARM}" in
  B) EXTRA_ARGS+=(--method.student_state_probability 0.25) ;;
  C) EXTRA_ARGS+=(--method.allow_prompt_only true --method.denoising_weight 0.0
                 --training.data.data_path "${SPRINT_ROOT}/data/user-prompts-20260906/index-v2") ;;
esac
# Re-derive seams from the actual student block map so a re-cut checkpoint can
# never train against another map's seam indices.
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" \
 --template "${CONFIG}" --student "${STUDENT}" --depth "${STAGE}" \
 --output "${OUTPUT_ROOT}/config.yaml"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "${OUTPUT_ROOT}/config.yaml" \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.tracker.run_name "base${STAGE}-detail-band-${ARM}-${SLURM_JOB_ID}" \
  "${EXTRA_ARGS[@]}"
test -s "${OUTPUT_ROOT}/checkpoint-300/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-300" \
 --output-dir "${OUTPUT_ROOT}/export-300" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-300" --checkpoint-role "base${STAGE}-detail-band-${ARM}" \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-300" --run-id "detail-band-${STAGE}-${ARM}-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
ffmpeg -v error -y -i "${OUTPUT_ROOT}/media-300/00_speech_exact_presenter.mp4" -vn -ac 2 -ar 32000 \
 -c:a pcm_s16le "${OUTPUT_ROOT}/media-300/00_speech_exact_presenter.wav"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/audio_fidelity_gate.py" \
 --candidate-dir "${OUTPUT_ROOT}/media-300" --reference-dir "${REFERENCE}" \
 --receipt-out "${OUTPUT_ROOT}/fidelity-gate.json" || true
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/verify_speech_asr.py" \
 --wav "${OUTPUT_ROOT}/media-300/00_speech_exact_presenter.wav" \
 --expected "Fast video and clear audio arrive together." \
 --output "${OUTPUT_ROOT}/asr-gate.json" \
 --run-id "detail-band-${STAGE}-${ARM}-${SLURM_JOB_ID}" || true
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
