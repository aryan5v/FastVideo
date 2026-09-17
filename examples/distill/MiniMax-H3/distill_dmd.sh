#!/usr/bin/env bash
# Direct torchrun wrapper for the current MiniMax-H3 DMD2 config.
# Use examples/train/slurm/dmd2_32xgb200.sbatch for production allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MASTER_PORT="${MASTER_PORT:-29513}"
export FASTVIDEO_FA4="${FASTVIDEO_FA4:-1}"

export NUM_GPUS="${NUM_GPUS:-4}"
WORLD_SIZE="${NUM_GPUS}"
SP_SIZE="${SP_SIZE:-1}"
HSDP_REPLICATE="${HSDP_REPLICATE:-1}"
HSDP_SHARD="${HSDP_SHARD:-${WORLD_SIZE}}"
CONFIG="${CONFIG:-examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minimax_h3_dmd2_local}"

exec bash examples/train/run.sh "${CONFIG}" \
  --training.distributed.num_gpus "${WORLD_SIZE}" \
  --training.distributed.sp_size "${SP_SIZE}" \
  --training.distributed.hsdp_replicate_dim "${HSDP_REPLICATE}" \
  --training.distributed.hsdp_shard_dim "${HSDP_SHARD}" \
  --training.checkpoint.output_dir "${OUTPUT_DIR}" \
  "$@"
