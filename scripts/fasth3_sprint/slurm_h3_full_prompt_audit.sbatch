#!/usr/bin/env bash
#SBATCH --job-name=h3-full-prompt-audit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
: "${CODE_ROOT:?Immutable snapshot required}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export CODE_ROOT
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
CACHE=/mnt/lustre/vlm-shared/h3_t2av_prompt/t2av_dataset/nuva_lab/preprocessed/prompt_only_t2av_v1
OUT=${SPRINT_ROOT}/data/user-prompts-20260906/index-v1
"$PY" scripts/fasth3_sprint/index_h3_cached_prompts.py \
 --prompts "${SPRINT_ROOT}/data/user-prompts-20260906/train.jsonl" \
 --cache-root "${CACHE}/nuva_49p7k/data" --cache-root "${CACHE}/nuva_10k/data" --output "$OUT"
"$PY" scripts/fasth3_sprint/audit_h3_index_values.py --root "$OUT"
'
