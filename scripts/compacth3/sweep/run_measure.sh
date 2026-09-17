#!/bin/bash
# Container entrypoint: CPU-side behavioral metrics + paired rank comparison.
#
# Uses the fasth3-eval venv (numpy 1.26.4 -- numpy 2.x silently kills CLIP
# alignment and the VLM adherence check).  Note that this OpenCV (cv2 5.0.0)
# has no CascadeClassifier, so the Haar face count is unavailable; the crude
# permanence detector is the blob counter only.
#
# The per-rank passes run concurrently: measurement is CPU-bound and each rank
# is an independent directory.
set -uo pipefail

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
SWEEP="${SPRINT}/adaln_rank_analysis/sweep"
PY=/mnt/nfs/vlm-aryan/fasth3-eval/venv/bin/python

export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env >/dev/null 2>&1 || true

cd "$SWEEP"
echo "=== measure host=$(hostname) $(date -u) ==="
"${PY}" -c "import numpy, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__); print('CascadeClassifier', hasattr(cv2,'CascadeClassifier'))"

RANKS="${RANKS:-$(ls -d ${SWEEP}/sweep_r* 2>/dev/null | sed 's/.*sweep_r//' | tr '\n' ',' | sed 's/,$//')}"
echo "ranks to measure: ${RANKS}"

pids=()
for r in $(echo "$RANKS" | tr ',' ' '); do
  echo "--- launching measure for r${r} ---"
  "${PY}" "${SWEEP}/measure_rank.py" --ranks "$r" > "${SWEEP}/measure_r${r}.log" 2>&1 &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=$?
done
for r in $(echo "$RANKS" | tr ',' ' '); do
  echo "=== measure_r${r}.log tail ==="
  tail -6 "${SWEEP}/measure_r${r}.log" 2>/dev/null
done

echo "=== measure rc=${rc} $(date -u) ==="
if [[ $rc -eq 0 ]]; then
  echo "=== comparison ==="
  "${PY}" "${SWEEP}/compare_ranks.py" 2>&1 | tail -220
  echo "=== compare exit=${PIPESTATUS[0]} ==="
fi
date -u
exit $rc
