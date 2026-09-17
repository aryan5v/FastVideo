#!/bin/bash
# Container entrypoint: AdaLN rank patch smoke test.
#
# Three arms on ONE small case and ONE seed:
#   r16   patched with a real folded basis   -> plumbing works, clip produced
#   r768  patched with the identity fold     -> "patch present but identity"
#   -1    NOT patched (FASTVIDEO_ADALN_RANK unset) -> the reference
#
# r768 vs -1 is the end-to-end identity gate: if the fold swap were doing
# anything to the function, these two clips would differ.
set -uo pipefail

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
SWEEP="${SPRINT}/adaln_rank_analysis/sweep"
DRIVER="${SWEEP}/sweep_driver.py"
CKPT="${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3/inference/checkpoint-1400"
CASES="${SMOKE_CASES:-t2va-2026082050-007339}"
SEEDS="${SMOKE_SEEDS:-20260917}"
ARMS="${SMOKE_ARMS:-16,768,-1}"

export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${M}:${SWEEP}:${SPRINT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0
export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250
export FASTVIDEO_ADALN_FOLD_DIR="${SWEEP}/folds"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29613
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env >/dev/null 2>&1 || true

mkdir -p "$SWEEP/smoke"
cd "$M"
echo "=== smoke host=$(hostname) arms=${ARMS} cases=${CASES} seeds=${SEEDS} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
date -u

rc=0
for arm in $(echo "$ARMS" | tr ',' ' '); do
  OUT="${SWEEP}/smoke/arm_r${arm}"
  mkdir -p "$OUT"
  echo "=== SMOKE ARM ${arm} start $(date -u) ==="
  "${PY}" "$DRIVER" --model-path "$CKPT" --output-dir "$OUT" --rank="$arm" \
      --seeds "$SEEDS" --num-gpus 4 --case-ids "$CASES"
  status=$?
  echo "=== SMOKE ARM ${arm} exit=${status} ==="
  if [[ $status -ne 0 ]]; then rc=$status; fi
done

echo "=== patch receipts ==="
grep -h "ADALN_RANK_PATCH_OK" "${SWEEP}/slurm-"*.log 2>/dev/null | tail -8 || true
grep -h "ADALN_RANK_PATCH_OK" /mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-*.log 2>/dev/null | tail -8 || true

echo "=== ladder check (expect 5 grid points / 4 DiT forwards) ==="
case_id=$(echo "$CASES" | cut -d, -f1)
A="${SWEEP}/smoke/arm_r768/${case_id}__seed${SEEDS}.mp4"
B="${SWEEP}/smoke/arm_r-1/${case_id}__seed${SEEDS}.mp4"
if [[ -f "$A" && -f "$B" ]]; then
  "${PY}" "${SWEEP}/check_identity_gate.py" --a "$A" --b "$B"
  gate=$?
else
  echo "identity gate skipped: missing $A or $B"
  gate=3
fi

echo "=== smoke finished rc=${rc} gate=${gate} $(date -u) ==="
[[ $rc -ne 0 ]] && exit $rc
exit $gate
