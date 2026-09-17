#!/bin/bash
# Export compact NVFP4 weights using the worker-side hook.
set -euo pipefail
S=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
RUN=${S}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3
OUT=${RUN}/exports/nvfp4-ckpt1400
mkdir -p "$OUT"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${M}:${S}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTVIDEO_EXPORT_QUANT_SIDECAR="$OUT"
export FASTVIDEO_EXPORT_LANE=nvfp4
cd "$M"
echo "=== NVFP4 export -> $OUT ==="
/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python examples/inference/basic/basic_fasth3.py \
  --model-path "$RUN/inference/checkpoint-1400" \
  --prompt "export" --output "$OUT/tmp" \
  --height 480 --width 832 --num-frames 124 --steps 5 --num-gpus 4 \
  --repeats 1 --transformer-quant NVFP4H3 --no-fa4 2>&1 | grep -aE "EXPORT|Converting loaded|purge receipt|Error|Traceback" | tail -20
echo "=== output dir ==="
ls -la "$OUT"
