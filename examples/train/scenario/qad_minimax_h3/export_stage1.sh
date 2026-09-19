#!/usr/bin/env bash
# Export the stage-1 DCP checkpoint into a diffusers dir that stage 2 uses as
# the student init.
#
# Equivalent of examples/train/scenario/qad_wan2_1_mixkit/export_stage1.sh.
# H3 needs no different export path: `dcp_to_diffusers` reshards on one GPU and
# writes `<out>/transformer/model.safetensors` as a SINGLE file, which is
# exactly the shape `models.student.transformer_override_safetensor` expects.
#
# Usage:
#   bash export_stage1.sh [CHECKPOINT_DIR] [OUTPUT_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

SPRINT_ROOT=${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}
STAGE1_OUT=${STAGE1_OUT:-${SPRINT_ROOT}/runs/release20b-qad2s-nvfp4-qat-stage1-v1}

CHECKPOINT_DIR=${1:-${STAGE1_OUT}/checkpoint-4000}
OUTPUT_DIR=${2:-${STAGE1_OUT}/diffusers}

if [[ ! -e "${CHECKPOINT_DIR}/.complete" ]]; then
    echo "Stage-1 checkpoint is not complete: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ ! -e "${CHECKPOINT_DIR}/dcp/.metadata" ]]; then
    echo "Missing DCP metadata under ${CHECKPOINT_DIR}/dcp" >&2
    exit 1
fi

# --weights-only: a FineTuneMethod checkpoint only carries roles.student, and
#                 this skips optimizer/scheduler restore (half the GPU memory).
# --link-base:    the base pipeline components (Qwen3-VL text encoder, VAEs)
#                 are symlinked instead of copied -- hundreds of GB saved.
#                 The export then depends on the stage-1 base staying in place.
# --verify:       strictly reload the exported transformer immediately, so a
#                 key-mapping bug fails here and not inside the stage-2 launch.
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role student \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --weights-only \
    --link-base \
    --verify

echo
echo "Stage-1 export ready: ${OUTPUT_DIR}"
echo "Stage 2 reads:"
echo "  models.student.init_from                        = ${OUTPUT_DIR}"
echo "  models.student.transformer_override_safetensor  = ${OUTPUT_DIR}/transformer/model.safetensors"
