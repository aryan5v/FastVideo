#!/usr/bin/env bash
# QAD stage 2 for MiniMax-H3: DMD2 distillation on the stage-1 student.
#
# Modeled on examples/train/scenario/qad_wan2_1_mixkit/run_stage2.sh.
# Differences, all forced by H3:
#   * no Attn-QAT (H3 has no ATTN_QAT_TRAIN backend);
#   * SP/HSDP geometry follows the H3 QAD lane;
#   * the stage-1 result is delivered through
#     models.student.transformer_override_safetensor only. init_from stays on
#     the bf16 student because load_run_config reads <init_from>/model_index.json
#     to resolve the pipeline config class, so it cannot point at the stage-1
#     export before stage 1 has produced it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

SPRINT_ROOT=${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}
STAGE1_OUT=${STAGE1_OUT:-${SPRINT_ROOT}/runs/release20b-qad2s-nvfp4-qat-stage1-v1}

# The two checkpoint paths are optional positionals. Read them only while the
# next argument is not a flag: consuming a fixed $1/$2 would swallow an override
# (run_stage2.sh --some.key value) as a path, and run.sh would never see it.
POSITIONAL=()
while (( $# > 0 )) && [[ "$1" != -* ]]; do
    POSITIONAL+=("$1")
    shift
done
STAGE1_DIFFUSERS=${POSITIONAL[0]:-${STAGE1_OUT}/diffusers}
INIT_WEIGHTS=${POSITIONAL[1]:-${STAGE1_DIFFUSERS}/transformer/model.safetensors}
# Everything left is an override for run.sh, whose parser rejects bare tokens.
EXTRA=("$@")
STUDENT_BASE=${STUDENT_BASE:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3/inference/checkpoint-1400}
TEACHER_BASE=${TEACHER_BASE:-${SPRINT_ROOT}/release-candidates/base-h3-teacher-complete-v1}
NUM_GPUS=${NUM_GPUS:-32}
SP_SIZE=${SP_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
# The proven H3 QAD lane ran a short 200-step production target.
PRODUCTION_TARGET=${PRODUCTION_TARGET:-200}

# -e, not -f: component_loader accepts either a single .safetensors file (what
# dcp_to_diffusers writes) or a directory of shards.
if [[ ! -e "${INIT_WEIGHTS}" ]]; then
    echo "Missing exported stage-1 weights: ${INIT_WEIGHTS}" >&2
    echo "Run export_stage1.sh before stage 2." >&2
    exit 1
fi
if [[ ! -f "${STUDENT_BASE}/model_index.json" ]]; then
    echo "Student base is not a resolvable pipeline dir: ${STUDENT_BASE}" >&2
    exit 1
fi
if [[ ! -f "${TEACHER_BASE}/model_index.json" ]]; then
    echo "Teacher base is not a resolvable pipeline dir: ${TEACHER_BASE}" >&2
    exit 1
fi
if (( NUM_GPUS % SP_SIZE != 0 )); then
    echo "NUM_GPUS (${NUM_GPUS}) must be a multiple of SP_SIZE (${SP_SIZE})" >&2
    exit 2
fi

export NUM_GPUS

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/stage2_qad_distill.yaml" \
    --models.student.init_from "${STUDENT_BASE}" \
    --models.student.transformer_override_safetensor "${INIT_WEIGHTS}" \
    --models.teacher.init_from "${TEACHER_BASE}" \
    --models.critic.init_from "${TEACHER_BASE}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.sp_size "${SP_SIZE}" \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}" \
    --training.loop.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --training.loop.max_train_steps "${PRODUCTION_TARGET}" \
    --method.rollout_carry_slots "${GRAD_ACCUM}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
