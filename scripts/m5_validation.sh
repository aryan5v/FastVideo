#!/usr/bin/env bash
# Apple-Silicon validation runbook: run identically on M4 (baseline) and on an
# M5 (payoff) and diff the outputs. Proves whether the Metal 4 Neural Accelerator
# path is engaged, and measures the distilled QAD 1.3B model end-to-end.
#
# Usage:
#   scripts/m5_validation.sh <tag> [model_root]
#     <tag>        label for output dir, e.g. "m4_max" or "friend_m5_24gb"
#     [model_root] defaults to the run-2 raw QAD export
#
# Prereqs: the lightweight MLX env at .venv (mlx==0.31.2), the distilled model
# root present locally. Does NOT touch git.
set -euo pipefail

TAG="${1:?usage: m5_validation.sh <tag> [model_root]}"
MODEL_ROOT="${2:-/Users/aryank/models/qad_int8_v2}"
OUT="bench/m5/${TAG}"

cd "$(dirname "$0")/.."
source .venv/bin/activate
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
mkdir -p "$OUT"

echo "== [1/3] accelerator probe (is the Neural Accelerator engaged?) =="
python -m fastvideo.mlx_runtime.accel_probe --json "${OUT}/accel.json"

PROMPT="A red fox trotting through a snowy pine forest at golden hour, cinematic"
COMMON=(--prompt "$PROMPT" --height 480 --width 832 --num-frames 81
        --modes fp16,int8 --decoders taehv --model-root "$MODEL_ROOT")

echo "== [2/3] distilled QAD end-to-end: BASELINE (no compile/fast_norm) =="
python -m fastvideo.benchmarks.mlx_fastwan_bench "${COMMON[@]}" \
  --output-dir "${OUT}/qad_baseline"

echo "== [3/3] distilled QAD end-to-end: compile + fast_norm =="
FASTVIDEO_MLX_FAST_NORM=1 python -m fastvideo.benchmarks.mlx_fastwan_bench "${COMMON[@]}" \
  --compile --output-dir "${OUT}/qad_compile"

echo "== done. Artifacts under ${OUT}/ =="
echo "   accel.json           -> quant-vs-fp16 speedup + accelerator_likely_engaged"
echo "   qad_baseline/*.mp4    -> distilled model video (eyeball the fox!)"
echo "   qad_*/metrics.json    -> denoise_steady_step_s, peak_gib, ms_ssim_vs_ref"
