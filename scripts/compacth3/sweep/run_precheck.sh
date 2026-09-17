#!/bin/bash
# Container entrypoint: pre-sweep correctness checks that need no GPU.
#   1. identity gate -- patched r=768 vs unpatched, same case and seed
#   2. dependency probe -- confirm the measurement venv can do the metrics
#      (numpy<2, headless cv2 with Farneback/remap/connectedComponents, av)
set -uo pipefail

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
SWEEP="${SPRINT}/adaln_rank_analysis/sweep"
PY_WAN=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
PY_EVAL=/mnt/nfs/vlm-aryan/fasth3-eval/venv/bin/python

export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
mkdir -p "$SWEEP"

echo "=== host=$(hostname) $(date -u) ==="

echo "--- 1. identity gate (PyAV, wan venv) ---"
A="${SWEEP}/smoke/arm_r768/t2va-2026082050-007339__seed20260917.mp4"
B="${SWEEP}/smoke/arm_r-1/t2va-2026082050-007339__seed20260917.mp4"
if [[ -f "$A" && -f "$B" ]]; then
  "${PY_WAN}" "${SWEEP}/check_identity_gate.py" --a "$A" --b "$B"
  echo "GATE_EXIT=$?"
else
  echo "GATE SKIPPED: missing $A or $B"
  echo "GATE_EXIT=3"
fi

echo "--- 2. dependency probe: wan venv cv2 ---"
"${PY_WAN}" -c "
try:
    import cv2, numpy; print('wan cv2 OK', cv2.__version__, 'numpy', numpy.__version__)
except Exception as e:
    print('wan cv2 FAIL', type(e).__name__, e)
" 2>&1 | tail -3

echo "--- 3. dependency probe: eval venv (the measurement interpreter) ---"
"${PY_EVAL}" -c "
import numpy, cv2
print('eval numpy', numpy.__version__, 'cv2', cv2.__version__)
import numpy as np
a = (np.random.RandomState(0).rand(64,64)*255).astype('uint8')
b = np.roll(a, 2, axis=1)
f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
w = cv2.remap(b, np.zeros_like(a, dtype='float32'), np.zeros_like(a, dtype='float32'), cv2.INTER_LINEAR)
n, lab, stats, cent = cv2.connectedComponentsWithStats((a>128).astype('uint8'), 8)
m = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
labimg = cv2.cvtColor(cv2.merge([a,a,a]), cv2.COLOR_BGR2LAB)
casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print('FARNEBACK_OK', f.shape, 'REMAP_OK', w.shape, 'CC_OK', n, 'MORPH_OK', m.shape,
      'LAB_OK', labimg.shape, 'HAAR_OK', (not casc.empty()))
" 2>&1 | tail -4

echo "--- 4. dependency probe: av + scorer imports ---"
"${PY_EVAL}" -c "
import av, numpy; print('av OK', av.__version__, 'numpy', numpy.__version__)
" 2>&1 | tail -2
"${PY_EVAL}" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('sc', '/mnt/nfs/vlm-aryan/fasth3-eval/score_34block.py')
m = importlib.util.module_from_spec(spec); sys.modules['sc']=m; spec.loader.exec_module(m)
print('SCORER_IMPORT_OK families:', sorted(m.FAMILIES))
print('HAVE:', {k:v for k,v in m.HAVE.items()})
" 2>&1 | tail -6

date -u
