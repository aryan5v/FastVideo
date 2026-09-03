#!/usr/bin/env bash
# Export a node-local step-260 DCP snapshot on a separate GPU and upload it.

set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
IMAGE=/mnt/lustre/vlm-wlsaidhi/fastvideo/images/fastvideo-dev-sm100-3da750ded0ec.sqsh
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_v1_16gpu_overfit/export-step260
TOKEN_FILE=/mnt/lustre/vlm-aryan/.secrets/huggingface_token

mkdir -p "${RESULT_DIR}"
sbatch \
  --job-name=fasth3-export-260 \
  --partition=all \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=32 \
  --mem=400G \
  --time=04:00:00 \
  --chdir="${WORKTREE_ROOT}" \
  --output="${RESULT_DIR}/%x_%j.out" \
  --error="${RESULT_DIR}/%x_%j.err" <<'SBATCH_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
IMAGE=/mnt/lustre/vlm-wlsaidhi/fastvideo/images/fastvideo-dev-sm100-3da750ded0ec.sqsh
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_v1_16gpu_overfit/export-step260
TOKEN_FILE=/mnt/lustre/vlm-aryan/.secrets/huggingface_token
PORT=29660
ARCHIVE=/tmp/fasth3-checkpoint-260.tar
SOURCE=/tmp/fasth3_checkpoint_260_export_source
EXPORT=/tmp/fasth3_hybrid_step260_export

rm -rf -- "${SOURCE}" "${EXPORT}"
rm -f -- "${ARCHIVE}"
hostname > "${RESULT_DIR}/receiver-${SLURM_JOB_ID}.txt"
python3 "${WORKTREE_ROOT}/scripts/transfer_stream.py" receive \
  --port "${PORT}" --output "${ARCHIVE}"
tar -C /tmp -xf "${ARCHIVE}"
test "$(find "${SOURCE}/dcp" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 16
test "$(find "${SOURCE}" -maxdepth 1 -type f -name 'rng_state_rank*.pt' | wc -l)" -eq 16
test -s "${SOURCE}/dcp/.metadata"
test -s "${SOURCE}/metadata.json"

srun --nodes=1 --ntasks=1 --gres=gpu:1 \
  --container-image="${IMAGE}" \
  --container-mounts=/mnt/lustre:/mnt/lustre,/tmp:/host-tmp \
  --container-workdir="${WORKTREE_ROOT}" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
    /opt/venv/bin/python3 -m fastvideo.train.entrypoint.dcp_to_minimax_h3_hybrid \
      --checkpoint /host-tmp/fasth3_checkpoint_260_export_source \
      --step 260 \
      --output-dir /host-tmp/fasth3_hybrid_step260_export
    /opt/venv/bin/python3 scripts/upload_hf_hybrid_export.py \
      --folder /host-tmp/fasth3_hybrid_step260_export \
      --token-file /mnt/lustre/vlm-aryan/.secrets/huggingface_token \
      --repo-name FastH3-Hybrid-Preview-v1-step-260
  '

echo complete > "${RESULT_DIR}/complete-${SLURM_JOB_ID}.txt"
rm -rf -- "${SOURCE}" "${EXPORT}"
rm -f -- "${ARCHIVE}"
SBATCH_SCRIPT
