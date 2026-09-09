#!/bin/bash
#SBATCH --job-name=h3-b34-soup
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
# No-training quality knob: interpolate 7048 (audio-bright) with 7101
# (contrast/motion-max) at three alphas, decode the five-prompt panel for each,
# so the joint fidelity gate can pick a mix without spending training time.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export LEFT="${SPRINT_ROOT}/runs/base34-recut-detail-band/job-7048/export-200/transformer"
export RIGHT="${SPRINT_ROOT}/runs/base34-speech-lift/job-7101/export-100/transformer"
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base34-soup/job-${SLURM_JOB_ID}"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${LEFT}/model.safetensors"
test -s "${RIGHT}/model.safetensors"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/soup_h3_exports.py" \
 --left "${LEFT}" --right "${RIGHT}" --alphas 0.3,0.5,0.7 --out-dir "${OUTPUT_ROOT}/models"
SRC_ROOT="$(dirname "${LEFT}")"
for alpha in 0.30 0.50 0.70; do
 model_root="${OUTPUT_ROOT}/models/alpha-${alpha}"
 mkdir -p "${model_root}"
 for component in "${SRC_ROOT}"/*; do
   name="$(basename "${component}")"
   [[ "${name}" == "transformer" ]] && continue
   ln -sfn "${component}" "${model_root}/${name}"
 done
 "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "${model_root}" --checkpoint-role "base34-soup-${alpha}" \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
  --output-dir "${OUTPUT_ROOT}/media-${alpha}" --run-id "base34-soup-${alpha}-${SLURM_JOB_ID}" \
  --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
  --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
  --profile strict --no-fa4 --no-compile --no-upload-videos
done
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
