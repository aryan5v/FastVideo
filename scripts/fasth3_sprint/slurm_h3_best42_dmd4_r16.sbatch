#!/bin/bash
#SBATCH --job-name=h3-b42dmd-r16
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
# Rank-16 AdaLN factorization of the 42-block DMD2 4-call turbo SKU
# (job-7903 export-500), then the five-prompt SDPA panel so r16 can be
# compared against its full-rank parent.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export SRC="${SPRINT_ROOT}/runs/best42-dmd4-resume/job-7903/export-500"
export DEST="${SPRINT_ROOT}/checkpoints/best42-dmd4-r16-v1"
export OUTPUT_ROOT="${SPRINT_ROOT}/diagnostics/best42-dmd4-r16-five-${SLURM_JOB_ID}"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${SRC}/transformer/config.json"
test ! -e "${DEST}"
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
"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py" \
 --src "${SRC}/transformer" --dst "${DEST}/transformer" --rank 16
for component in "${SRC}"/*; do
  name="$(basename "${component}")"
  [[ "${name}" == "transformer" ]] && continue
  ln -sfn "${component}" "${DEST}/${name}"
done
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${DEST}" --checkpoint-role best42-dmd4-r16 \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}" --run-id "best42-dmd4-r16-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
'
