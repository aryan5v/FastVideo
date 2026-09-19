#!/usr/bin/env bash
# QAD stage 1 for MiniMax-H3: supervised NVFP4-QAT finetune.
#
# Modeled on examples/train/scenario/qad_wan2_1_mixkit/run_stage1.sh.
# Differences, all forced by H3 (see stage1_qat_finetune.yaml):
#   * no FASTVIDEO_ATTN_QAT_FWD_EXACT_M -- Attn-QAT is not used at all;
#   * SP/HSDP geometry follows the H3 QAD lane (sp_size 4, full-world shard)
#     instead of the Wan template's sp_size == num_gpus;
#   * the dataset is a list of five production subtrees, so it is left to the
#     YAML unless DATA_DIR is set to a single replacement root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Optional single-root override (smoke runs / a different corpus). Unset means
# "use the five t2va subtrees declared in the YAML".
DATA_DIR=${DATA_DIR:-}
NUM_GPUS=${NUM_GPUS:-32}
# Stage-1 effective batch. Deliberately 1, matching the Wan template this is
# ported from, and NOT the 8 that the H3 DMD2 lane ran: those are different
# stages. DMD2 needs a large batch because its gradient is a noisy
# distribution-matching estimate; a supervised finetune regressing on real
# targets does not. Accumulating 8 here would also multiply the step budget by
# eight (4000 steps x 8 microbatches), turning an ~11h stage into days. Raise it
# only with a matching reduction in max_train_steps.
GRAD_ACCUM=${GRAD_ACCUM:-1}
SP_SIZE=${SP_SIZE:-4}

if (( NUM_GPUS % SP_SIZE != 0 )); then
    echo "NUM_GPUS (${NUM_GPUS}) must be a multiple of SP_SIZE (${SP_SIZE})" >&2
    exit 2
fi

export NUM_GPUS

OVERRIDES=(
    --training.distributed.num_gpus "${NUM_GPUS}"
    --training.distributed.sp_size "${SP_SIZE}"
    --training.distributed.hsdp_replicate_dim 1
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}"
    --training.loop.gradient_accumulation_steps "${GRAD_ACCUM}"
)

if [[ -n "${DATA_DIR}" ]]; then
    OVERRIDES+=(--training.data.data_path "${DATA_DIR}")
fi

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/stage1_qat_finetune.yaml" \
    "${OVERRIDES[@]}" \
    "$@"
