#!/bin/bash
# Container entrypoint for the AdaLN timestep-rank spectral analysis.
#
# usage:
#   sbatch <SPRINT>/exp.sbatch <SPRINT>/adaln_rank_analysis/run_adaln_rank.sh
# env:
#   STAGE=both|parent|dmd2   (default both) -- which checkpoints to analyse
#   SMOKE=1                  (default 0)    -- only the first 2 blocks, load check
set -uo pipefail

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
SCRIPT="${SPRINT}/adaln_rank_analysis/analyze_adaln_rank.py"
OUT="${SPRINT}/adaln_rank_analysis"

PARENT_CKPT="${SPRINT}/runs/release20b-folded-long-4k-v3/dmd-parent-step750-complete-v1"
DMD2_CKPT="${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3/inference/checkpoint-1400"

STAGE="${STAGE:-both}"
SMOKE="${SMOKE:-0}"

export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
# M first: the venv's editable fastvideo finder is appended to sys.meta_path, so
# the path-based finder (sys.path) still wins and M is the code that runs.
export PYTHONPATH="${M}:${SPRINT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export FASTVIDEO_MINIMAX_H3_FUSIONS=0
export FASTVIDEO_DISABLE_ATTENTION_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env >/dev/null 2>&1 || true

mkdir -p "$OUT"
cd "$M"

echo "=== host=$(hostname) gpus=$(nvidia-smi -L | wc -l) stage=${STAGE} smoke=${SMOKE} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then EXTRA+=(--max-blocks 2); fi

rc=0
run_one() {
  local tag="$1" ckpt="$2"
  echo "=== ${tag}: ${ckpt} ==="
  "${PY}" "$SCRIPT" --tag "$tag" --model-path "$ckpt" --out-dir "$OUT" "${EXTRA[@]}"
  local status=$?
  echo "=== ${tag} exit=${status} ==="
  if [[ $status -ne 0 ]]; then rc=$status; fi
}

if [[ "$STAGE" == "both" || "$STAGE" == "parent" ]]; then run_one parent "$PARENT_CKPT"; fi
if [[ "$STAGE" == "both" || "$STAGE" == "dmd2" ]]; then run_one dmd2 "$DMD2_CKPT"; fi

echo "=== analysis finished rc=${rc} ==="
ls -la "$OUT"
exit $rc
