#!/usr/bin/env bash
#SBATCH --job-name=h3-tests
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00

set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
RESULT_DIR="${RESULT_DIR:-${SPRINT_ROOT}/eval/h0-training-contract}"
PRECOMMIT_SITE="${SPRINT_ROOT}/tools/precommit-site"
mkdir -p "${RESULT_DIR}" "${PRECOMMIT_SITE}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  bash -lc "
    set -euo pipefail
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export PYTHONPATH=${PRECOMMIT_SITE}:${REPO_ROOT}
    export PYTHONDONTWRITEBYTECODE=1
    if ! /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -c 'import pre_commit' >/dev/null 2>&1; then
      /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pip install \
        --target '${PRECOMMIT_SITE}' 'pre-commit==4.3.0'
    fi
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pre_commit run --files \
      examples/train/configs/example.yaml \
      fastvideo/configs/models/dits/minimax_h3.py \
      fastvideo/train/methods/base.py \
      fastvideo/train/models/minimax_h3/minimax_h3.py \
      fastvideo/train/utils/config.py \
      fastvideo/train/utils/training_config.py \
      scripts/checkpoint_conversion/prune_minimax_h3_blocks.py \
      scripts/fasth3_sprint/run_baseline_matrix.py \
      scripts/fasth3_sprint/score_minimax_h3_blocks.py \
      scripts/fasth3_sprint/aggregate_minimax_h3_block_scores.py \
      scripts/fasth3_sprint/build_multishot_calibration_supplement.py \
      | tee '${RESULT_DIR}/pre_commit.log'
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pytest -q \
      fastvideo/tests/train/methods/test_minimax_h3_finetune.py \
      fastvideo/tests/train/utils/test_config.py \
      fastvideo/tests/transformers/test_minimax_h3_block_map_config.py \
      fastvideo/tests/transformers/test_minimax_h3_block_pruning.py \
      fastvideo/tests/transformers/test_minimax_h3_block_scoring.py \
      | tee '${RESULT_DIR}/pytest.log'
    date -Is > '${RESULT_DIR}/completed_at.txt'
    git rev-parse HEAD > '${RESULT_DIR}/base_commit.txt'
  "
