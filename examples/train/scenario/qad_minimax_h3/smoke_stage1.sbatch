#!/bin/bash
# Two-step end-to-end smoke for stage 1 on ONE 4-GPU node.
#
# Proves the things load_run_config cannot: MiniMaxH3Model + FineTuneMethod +
# nvfp4_qat_train linears + the t2va parquet dataloader + native shape
# bucketing compose into a real training step, with a real optimizer step.
#
# Uses a fresh scratch output dir and 2 steps, so it never touches the
# production stage-1 namespace.
#SBATCH --job-name=h3-qad2s-smoke
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-smoke-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-smoke-%j.log

set -euo pipefail
SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
TREE=$SPRINT/code/release20b-dmd2-v12-corrected-v17
SCEN=$TREE/examples/train/scenario/qad_minimax_h3
OUT=$SPRINT/runs/qad2s-stage1-smoke-$SLURM_JOB_ID

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$TREE:$SPRINT/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages'
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=disabled
    export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_FA4=0 FASTVIDEO_MINIMAX_H3_FUSIONS=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
    export OMP_NUM_THREADS=1
    cd '$TREE'
    NUM_GPUS=4 SP_SIZE=1 GRAD_ACCUM=1 \
    DATA_DIR=/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data \
    bash '$SCEN/run_stage1.sh' \
      --training.loop.max_train_steps 2 \
      --training.checkpoint.output_dir '$OUT' \
      --training.checkpoint.training_state_checkpointing_steps 0 \
      --training.checkpoint.checkpoints_total_limit 1 \
      --callbacks.validation.every_steps 1000000 \
      --callbacks.validation.run_at_start false
  "
echo "SMOKE_EXIT=$?"
