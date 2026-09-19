#!/bin/bash
# Parse-validate the two-stage H3 QAD configs inside the training container.
#   sbatch validate_configs.sbatch
# One node / one GPU is enough: load_run_config never touches the model.
#SBATCH --job-name=h3-qad2s-validate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-validate-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/slurm-qad2s-validate-%j.log

set -euo pipefail
SPRINT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
TREE=$SPRINT/code/release20b-dmd2-v12-corrected-v17
SCEN=$TREE/examples/train/scenario/qad_minimax_h3

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$TREE:$SPRINT/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages'
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_FA4=0 FASTVIDEO_MINIMAX_H3_FUSIONS=0
    cd '$TREE'
    python '$SCEN/validate_configs.py' \
      '$SCEN/stage1_qat_finetune.yaml' \
      '$SCEN/stage2_qad_distill.yaml'
  "
