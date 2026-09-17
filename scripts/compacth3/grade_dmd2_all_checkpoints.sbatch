#!/bin/bash
#SBATCH --job-name=h3-dmd2-grade-all
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/sbatch-h3-dmd2-grade-all-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/sbatch-h3-dmd2-grade-all-%j.log

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
EVAL=/mnt/nfs/vlm-aryan/fasth3-eval
RUN=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3
OUT="${RUN}/dmd-native-export-grade-all-v1"

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  bash -lc '
set -euo pipefail
SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
EVAL=/mnt/nfs/vlm-aryan/fasth3-eval
RUN=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3
OUT="${RUN}/dmd-native-export-grade-all-v1"
mkdir -p "${OUT}"

for step in 100 200 400 600 800 1000 1200; do
  media="${RUN}/eval-exported-checkpoint-${step}-bench-five-exact4-v1/media"
  [[ $(find "${media}" -maxdepth 1 -name "*.mp4" | wc -l) -eq 5 ]] || {
    echo "FATAL: step ${step} does not have five videos" >&2
    exit 2
  }
done

bash "${EVAL}/probe_deps.sh" | tee "${OUT}/dependency-probe.txt"
"${EVAL}/venv/bin/python" "${EVAL}/score_34block.py" \
  --runs "${EVAL}/dmd2_all_checkpoints_runs.json" \
  --reference-media "${RUN}/eval-parent-750-bench-five-correct4-v1/media" \
  --prompts "${SPRINT}/job-scripts/bench_five_new_prompts.json" \
  --out "${OUT}/scores.json" | tee "${OUT}/scorer.log"
"${EVAL}/venv/bin/python" "${EVAL}/summarize_scores.py" "${OUT}/scores.json" | tee "${OUT}/summary.txt"
date -Is > "${OUT}/completed_at.txt"
'
