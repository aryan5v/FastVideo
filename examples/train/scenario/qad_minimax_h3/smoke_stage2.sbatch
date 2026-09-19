#!/bin/bash
# One-node stage-2 smoke: proves the STAGE-2 config composes (H3 DMD2 + NVFP4-QAT
# student + dense teacher/critic) and, critically, that
# `models.student.transformer_override_safetensor` is actually HONORED --
# `component_loader` silently falls back to init_from when the override path
# does not resolve, which would reproduce the exact failure this recipe exists
# to fix, and no static config check can catch that.
#
# The override is pointed at a directory of SYMLINKS to the bf16 student's
# shards, so the loaded paths in the log are distinguishable from init_from.
# Stage 1's real export replaces this with a single model.safetensors.
#
# Shape/step reductions follow the sprint's existing qad_smoke_1node.yaml
# (num_latent_t 7 / num_frames 22, generator_update_interval 1, no critic
# optimizer) so it fits one 4-GPU node.
#SBATCH --job-name=h3-qad2s-smoke2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-smoke2-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-smoke2-%j.log

set -euo pipefail
SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
TREE=$SPRINT/code/release20b-dmd2-v12-corrected-v17
SCEN=$TREE/examples/train/scenario/qad_minimax_h3
OUT=$SPRINT/runs/qad2s-stage2-smoke-$SLURM_JOB_ID

# Shape ladder: num_frames must be 17n+5 and num_latent_t is its
# video_latent_num_frames() value = 5n+2. n=1 is the sprint's smoke shape
SHAPE_N=${SHAPE_N:-1}
NUM_FRAMES=$((17 * SHAPE_N + 5))
NUM_LATENT_T=$((5 * SHAPE_N + 2))

STUDENT_REL=${STUDENT_REL:-release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3/inference/checkpoint-1400}
STUDENT=runs/$STUDENT_REL
PROBE=$SPRINT/runs/qad2s-override-probe-$(basename "$STUDENT_REL")
mkdir -p "$PROBE"
ln -sfn "$SPRINT/$STUDENT/transformer"/*.safetensors "$PROBE/" 2>/dev/null || true

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
    export STUDENT_BASE='$SPRINT/$STUDENT'
    NUM_GPUS=4 SP_SIZE=1 GRAD_ACCUM=1 PRODUCTION_TARGET=2 \
    bash '$SCEN/run_stage2.sh' '$PROBE' '$PROBE' \
      --method.generator_update_interval 1 \
      --method.rollout_carry_slots 1 \
      --training.data.data_path /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data \
      --training.data.num_latent_t '$NUM_LATENT_T' \
      --training.data.num_frames '$NUM_FRAMES' \
      --training.checkpoint.output_dir '$OUT' \
      --training.checkpoint.training_state_checkpointing_steps 0 \
      --training.checkpoint.checkpoints_total_limit 1 \
      --callbacks.validation.every_steps 1000000
  "
echo "SMOKE2_EXIT=$?"
