#!/bin/bash
# DMD2 distillation for MiniMax H3 (joint video + audio) on the modular
# trainer (fastvideo/train). Unlike the legacy SFWan scripts, the modern
# stack is YAML-driven: the {student, teacher, critic} trio, DMD2 method
# knobs, and callbacks all live in the run config.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-29513}
# Per-role backends come from the YAML (models.<role>.attention_backend);
# do not export FASTVIDEO_ATTENTION_BACKEND globally. FA4 selects the fast
# path inside the FLASH_ATTN roles.
export FASTVIDEO_FA4=${FASTVIDEO_FA4:-1}

NUM_GPUS=${NUM_GPUS:-4}
CONFIG=${CONFIG:-examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml}
DATA_DIR=${DATA_DIR:-data/crush-smol_h3_t2va_single_sample_preprocessed}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/minimax_h3_dmd2_3steps}

torchrun \
  --nnodes 1 \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NUM_GPUS" \
  -m fastvideo.train.entrypoint.train \
  --config "$CONFIG" \
  --training.distributed.num_gpus "$NUM_GPUS" \
  --training.distributed.sp_size "$NUM_GPUS" \
  --training.distributed.hsdp_shard_dim "$NUM_GPUS" \
  --training.data.data_path "$DATA_DIR" \
  --training.checkpoint.output_dir "$OUTPUT_DIR"
