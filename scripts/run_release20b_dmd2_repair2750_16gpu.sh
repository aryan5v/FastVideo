#!/bin/bash
# Export only the old phase-2750 student, then train it with the corrected
# V12 teacher/critic/optimizer contract. No old critic or optimizer state is
# loaded into the repair lineage.
set -euo pipefail

: "${CODE_ROOT:?}"
: "${DMD_CHECKPOINT:?}"
: "${REPAIR_PARENT:?}"
: "${OUTPUT_BASE:?}"
: "${TEACHER_PARENT:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python

if [[ "${SLURM_PROCID}" == "0" ]]; then
  test ! -e "${REPAIR_PARENT}"
  source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
  export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
  export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
  export PYTHONDONTWRITEBYTECODE=1
  cd "${CODE_ROOT}"
  # The export runs on rank 0 only.  Override the inherited four-task SLURM
  # rendezvous for this command so it cannot wait forever for absent ranks.
  env RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${EXPORT_MASTER_PORT:-31975}" \
    "${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint "${DMD_CHECKPOINT}" \
    --output-dir "${REPAIR_PARENT}" \
    --config "${CODE_ROOT}/examples/train/configs/distribution_matching/minimax_h3/release20b_dmd2_v12_dense.yaml" \
    --role student --weights-only --link-base
  test -s "${REPAIR_PARENT}/transformer/model.safetensors"
  test -s "${REPAIR_PARENT}/transformer/config.json"
  touch "${REPAIR_PARENT}/.repair-parent-complete"
fi

for _ in $(seq 1 720); do
  [[ -e "${REPAIR_PARENT}/.repair-parent-complete" ]] && break
  sleep 5
done
test -e "${REPAIR_PARENT}/.repair-parent-complete"

export SELECTED_PARENT="${REPAIR_PARENT}"
export PRODUCTION_TARGET=500
exec bash "${CODE_ROOT}/scripts/run_release20b_dmd2_v12_16gpu.sh"
