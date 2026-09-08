#!/bin/bash
# Prune the best activation-42 + 58k checkpoint to 34 blocks, then recover on 58k prompts.
#SBATCH --job-name=h3-a34-from42-p58k
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export PARENT="$SPRINT_ROOT/runs/base42-prompt58k-recovery/job-6929/export-200"
export BLOCK_MAP="$SPRINT_ROOT/runs/activation34-recovery/job-6928/maps/activation_34_block_map.json"
export OUTPUT_ROOT="$SPRINT_ROOT/runs/activation34-from42-prompt58k-recovery/job-${SLURM_JOB_ID}"
export STUDENT="$OUTPUT_ROOT/student34"
test -s "$CODE_ROOT/CODE_COMMIT"
test -s "$PARENT/transformer/config.json"
test -s "$BLOCK_MAP"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="$CODE_ROOT" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"$PY" "$CODE_ROOT/scripts/checkpoint_conversion/prune_minimax_h3_blocks.py" \
  --src "$PARENT/transformer" --dst "$STUDENT/transformer" \
  --block-map "$BLOCK_MAP" --strategy activation34-from-base42-prompt58k \
  --source-model "$PARENT" --source-revision base42-prompt58k-job6929-step200
"$PY" "$CODE_ROOT/scripts/fasth3_sprint/assemble_h3_stage.py" --base "$BASE" --student "$STUDENT"
"$PY" "$CODE_ROOT/scripts/fasth3_sprint/build_h3_recovery_config.py" \
  --template "$CODE_ROOT/examples/train/configs/fasth3_base42_recovery.yaml" \
  --student "$STUDENT" --output "$OUTPUT_ROOT/config.yaml" --depth 34
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "$OUTPUT_ROOT/config.yaml" \
  --training.loop.max_train_steps 200 \
  --method.allow_prompt_only true \
  --training.data.data_path "$SPRINT_ROOT/data/user-prompts-20260906/index-v2" \
  --training.checkpoint.output_dir "$OUTPUT_ROOT" \
  --training.checkpoint.training_state_checkpointing_steps 50 \
  --training.checkpoint.preserve_every_steps 50 \
  --training.tracker.run_name "activation34-from42-prompt58k-${SLURM_JOB_ID}"
test -s "$OUTPUT_ROOT/checkpoint-200/dcp/.metadata"
"$PY" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "$OUTPUT_ROOT/checkpoint-200" \
  --output-dir "$OUTPUT_ROOT/export-200" --role student --load-precision bf16 --verify
"$PY" "$CODE_ROOT/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "$OUTPUT_ROOT/export-200" --checkpoint-role activation34-from42-prompt58k-200 \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "$CODE_ROOT/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
  --output-dir "$OUTPUT_ROOT/media-200" --run-id "activation34-from42-prompt58k-200-${SLURM_JOB_ID}" \
  --source-commit "$(cat "$CODE_ROOT/CODE_COMMIT")" --max-prompts 5 \
  --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 \
  --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos
date -Is > "$OUTPUT_ROOT/awaiting_quality_review.txt"
'
