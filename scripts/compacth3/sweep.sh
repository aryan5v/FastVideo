#!/bin/bash
# Usage: sweep.sh <MODEL> <LABEL> <GRID> <QUANT|none> <SEEDS>
set -uo pipefail
MODEL="$1"; LABEL="$2"; STEPS="$3"; QUANT="${4:-none}"; SEEDS="${5:-20260912,4242,777}"

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
OUT=${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/sweep-$LABEL
mkdir -p "$OUT"

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env || true
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${M}:${SPRINT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250

QARG=""
[[ "$QUANT" != "none" && "$QUANT" != "-" ]] && QARG="--transformer-quant $QUANT"

PROMPTS_JSON=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/sweep_prompts.json

cd "$M"
echo "=== SWEEP $LABEL : grid=$STEPS quant=$QUANT seeds=$SEEDS ==="
python3 - "$PROMPTS_JSON" "$SEEDS" > "$OUT/jobs.txt" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for s in sys.argv[2].split(","):
    for r in rows:
        print(f"{r['id']}|{r['width']}|{r['height']}|{r['frames']}|{s.strip()}|{r['prompt']}")
PY

n=0
while IFS='|' read -r pid w h f seed ptext; do
  [ -z "$pid" ] && continue
  outdir="$OUT/${pid}_seed${seed}"
  mkdir -p "$outdir"
  printf "[%02d] %-10s %sx%s %sf seed=%s\n" "$n" "$pid" "$w" "$h" "$f" "$seed"
  $PY examples/inference/basic/basic_fasth3.py \
    --model-path "$MODEL" --prompt "$ptext" --output "$outdir" \
    --height "$h" --width "$w" --num-frames "$f" --steps "$STEPS" --num-gpus 4 \
    --seed "$seed" --repeats 1 $QARG --no-fa4 --video-decode-backend h3-vae \
    >> "$OUT/run.log" 2>&1 || true
  real=$(find "$outdir" -name "*.mp4" -type f ! -name "_fasth3_warmup.mp4" | head -1)
  [ -n "$real" ] && mv "$real" "$OUT/${pid}_seed${seed}.mp4" 2>/dev/null
  n=$((n+1))
done < "$OUT/jobs.txt"

echo "=== $LABEL: $(find "$OUT" -maxdepth 1 -name '*.mp4' | wc -l) of $n videos ==="
date -Is > "$OUT/completed_at.txt"
