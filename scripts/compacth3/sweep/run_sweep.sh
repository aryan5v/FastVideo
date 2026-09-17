#!/bin/bash
# Container entrypoint for the behavioral AdaLN rank sweep.
#
# usage: sbatch <SPRINT>/adaln_rank_analysis/sweep/sweep.sbatch \
#                <SPRINT>/adaln_rank_analysis/sweep/run_sweep.sh
# env (all optional):
#   RANKS=768,64,16,8      ranks to render, in order (one generator per rank)
#   SEEDS=20260917,424242  seeds; SEED is the OUTER loop, so every rank finishes
#                          seed A before any rank starts seed B.  A truncated run
#                          therefore still yields a complete, matched r64-vs-r16
#                          comparison rather than a partial one.
#   CASE_IDS=id1,id2       subset of the hard-motion set
#   TIME_BUDGET_S=81000    stop cleanly before the SLURM wall clock
set -uo pipefail

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
SWEEP="${SPRINT}/adaln_rank_analysis/sweep"
DRIVER="${SWEEP}/sweep_driver.py"
CKPT="${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3/inference/checkpoint-1400"

# Per-run settings live in a file rather than in `sbatch --export=...`: this
# cluster's SLURM rejects the submission with
# "user_env_retrieval_failed_requeued_held" when the submitting environment is
# exported (it cannot chdir to the user's home).  Write sweep_env.sh and submit
# plainly.
[[ -f "${SWEEP}/sweep_env.sh" ]] && source "${SWEEP}/sweep_env.sh"

RANKS="${RANKS:-768,64,16,8}"
SEEDS="${SEEDS:-20260917}"
TIME_BUDGET_S="${TIME_BUDGET_S:-81000}"
CASE_IDS="${CASE_IDS:-}"

export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
# M first: the venv's editable fastvideo finder is appended to sys.meta_path, so
# the path-based finder (sys.path) still wins and M is the code that runs.
export PYTHONPATH="${M}:${SWEEP}:${SPRINT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0
export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250
export FASTVIDEO_ADALN_FOLD_DIR="${SWEEP}/folds"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29611
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env >/dev/null 2>&1 || true

mkdir -p "$SWEEP"
cd "$M"

echo "=== host=$(hostname) gpus=$(nvidia-smi -L | wc -l) ranks=${RANKS} seeds=${SEEDS} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
date -u

START_EPOCH=$(date +%s)
rc=0
stop=0

for seed in $(echo "$SEEDS" | tr ',' ' '); do
  for rank in $(echo "$RANKS" | tr ',' ' '); do
    elapsed=$(( $(date +%s) - START_EPOCH ))
    if [[ $elapsed -ge $TIME_BUDGET_S ]]; then
      echo "=== budget exhausted (${elapsed}s >= ${TIME_BUDGET_S}s); stopping before r${rank}/seed${seed} ==="
      stop=1
      break
    fi
    OUT="${SWEEP}/sweep_r${rank}"
    mkdir -p "$OUT"
    echo "=== RANK ${rank} SEED ${seed} start $(date -u)  budget_left=$(( TIME_BUDGET_S - elapsed ))s ==="
    ARGS=(--model-path "$CKPT" --output-dir "$OUT" --rank "$rank" --seeds "$seed" --num-gpus 4)
    if [[ -n "$CASE_IDS" ]]; then ARGS+=(--case-ids "$CASE_IDS"); fi
    "${PY}" "$DRIVER" "${ARGS[@]}"
    status=$?
    echo "=== RANK ${rank} SEED ${seed} exit=${status} $(date -u) ==="
    if [[ $status -ne 0 ]]; then rc=$status; fi
  done
  [[ $stop -eq 1 ]] && break
done

echo "=== sweep finished rc=${rc} elapsed=$(( $(date +%s) - START_EPOCH ))s ==="
exit $rc
