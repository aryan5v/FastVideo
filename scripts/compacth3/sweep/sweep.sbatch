#!/bin/bash
#SBATCH --job-name=h3-adaln-rank-sweep
#SBATCH --partition=hpc-rack-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/adaln_rank_analysis/sweep/slurm-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/adaln_rank_analysis/sweep/slurm-%j.log
set -euo pipefail
SCRIPT="$1"
srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  bash "$SCRIPT"
