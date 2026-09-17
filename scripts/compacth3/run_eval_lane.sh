#!/bin/bash
# Run the real eval set, honouring each case's declared generation exactly.
#   usage: run_eval_lane.sh <MODEL> <LABEL> <BACKEND: h3-vae|taeh3> <QUANT|none> [MAXCASES]
set -euo pipefail

MODEL="$1"; LABEL="$2"; BACKEND="${3:-h3-vae}"; QUANT="${4:-none}"; MAXCASES="${5:-36}"
TEW="${6:-}"   # optional: serialized quantized text-encoder checkpoint dir

SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
M=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
OUT=${SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/eval-$LABEL
EVALJSON=${SPRINT}/eval.jsonl
mkdir -p "$OUT"
TAEH3CKPT=/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/checkpoints/taeh3.safetensors
mkdir -p "$HOME/.cache/fastvideo/taehv"; ln -sfn "$TAEH3CKPT" "$HOME/.cache/fastvideo/taehv/taeh3.safetensors" 2>/dev/null || true

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
TEWARG=""
[[ -n "$TEW" ]] && TEWARG="--text-encoder-weights $TEW"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 > "$OUT/vram_samples.csv" 2>/dev/null &
SAMPLER=$!

cd "$M"
echo "=== $LABEL : backend=$BACKEND quant=$QUANT cases=$MAXCASES (declared shapes enforced) ==="

# emit: case_id|width|height|frames|fps|prompt
python3 - "$EVALJSON" "$MAXCASES" > "$OUT/case_list.txt" <<'PY'
import json, sys
path, maxc = sys.argv[1], int(sys.argv[2])
n = 0
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    g = r["generation"]
    prompt = r["prompt"].replace("\n", " ").replace("|", "/")
    print(f"{r['case_id']}|{int(g['width'])}|{int(g['height'])}|{int(g['num_frames'])}|{g['fps']}|{prompt}")
    n += 1
    if n >= maxc:
        break
PY

i=0
while IFS='|' read -r cid w h f fps ptext; do
  [ -z "$cid" ] && continue
  # --output is a DIRECTORY (the harness names the mp4 inside it)
  outdir="$OUT/case_${i}_${cid}"
  mkdir -p "$outdir"
  echo "--- [$i/$MAXCASES] $cid  ${w}x${h} ${f}f ---"
  START=$(date +%s.%N)
  # shellcheck disable=SC2086
  $PY examples/inference/basic/basic_fasth3.py \
    --model-path "$MODEL" --prompt "$ptext" --output "$outdir" \
    --height "$h" --width "$w" --num-frames "$f" --steps 5 --num-gpus 4 \
    --repeats 1 $QARG $TEWARG --no-fa4 --video-decode-backend "$BACKEND" \
    >> "$OUT/run.log" 2>&1 || true
  END=$(date +%s.%N)
  # flatten: pick up whichever mp4 the harness wrote (skip the warmup)
  real=$(find "$outdir" -name "*.mp4" -type f ! -name "_fasth3_warmup.mp4" | head -1)
  [ -n "$real" ] && mv "$real" "$OUT/${i}_${cid}.mp4" 2>/dev/null || true
  awk -v a="$START" -v b="$END" -v c="$cid" -v w="$w" -v h="$h" -v f="$f" \
      'BEGIN{printf "%s %sx%s %sf wall=%.2f\n", c, w, h, f, b-a}' >> "$OUT/walls.txt"
  i=$((i+1))
done < "$OUT/case_list.txt"

sleep 2; kill -9 $SAMPLER 2>/dev/null || true; wait $SAMPLER 2>/dev/null || true

echo "=== $LABEL SUMMARY ==="
echo "  videos: $(find "$OUT" -maxdepth 1 -name '*.mp4' | wc -l)"
echo "  generation times (s):"
grep -oE "Generation time: [0-9.]+" "$OUT/run.log" 2>/dev/null | grep -oE "[0-9.]+" | tr "\n" " " | sed 's/^/    /'; echo
echo "  peak device VRAM:"
awk -F, 'NF==2 {gsub(/ /,"",$1); gsub(/ /,"",$2); if($2+0>m[$1]) m[$1]=$2+0}
         END {for (g in m) printf "    gpu%s peak = %d MiB (%.1f GB)\n", g, m[g], m[g]/1024}' "$OUT/vram_samples.csv" | sort
date -Is > "$OUT/completed_at.txt"
echo "=== $LABEL done ==="
