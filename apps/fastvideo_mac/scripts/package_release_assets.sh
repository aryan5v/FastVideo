#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
OUTPUT=${1:-"$APP_ROOT/release"}
MODEL_ROOT=${FASTWAN_MODEL_ROOT:-"$HOME/models/qad_int8_v2_ema"}
EMA_ROOT=${FASTWAN_EMA_ROOT:-"$HOME/mlx-ckpt-cache-qad-v2-ema/int8"}
RAW_ROOT=${FASTWAN_RAW_ROOT:-"$HOME/mlx-ckpt-cache-qad-v2/int8"}
RIFE_ROOT=${FASTWAN_RIFE_ROOT:-"$MODEL_ROOT/rife/RIFE-4.25"}
mkdir -p "$OUTPUT"

# `-h` follows the development cache symlinks so published archives are fully
# self-contained and never reference this Mac's Hugging Face blob directory.
COPYFILE_DISABLE=1 tar -h -C "$MODEL_ROOT" -czf "$OUTPUT/fastwan-qad-v2-shared.tar.gz" \
    tokenizer text_encoder scheduler vae transformer/config.json
COPYFILE_DISABLE=1 tar -h -C "$(dirname "$RIFE_ROOT")" -czf "$OUTPUT/fastwan-qad-rife-4.25.tar.gz" \
    "$(basename "$RIFE_ROOT")"
COPYFILE_DISABLE=1 tar -C "$EMA_ROOT" -czf "$OUTPUT/fastwan-qad-v2-ema-int8.tar.gz" \
    mlx_dit.json mlx_dit.safetensors
COPYFILE_DISABLE=1 tar -C "$RAW_ROOT" -czf "$OUTPUT/fastwan-qad-v2-raw-int8.tar.gz" \
    mlx_dit.json mlx_dit.safetensors
ditto -c -k --sequesterRsrc --keepParent "$APP_ROOT/dist/FastWan QAD.app" "$OUTPUT/FastWan-QAD-macOS.zip"

shasum -a 256 "$OUTPUT"/*
printf 'Release assets written to %s\n' "$OUTPUT"
