#!/usr/bin/env bash
# Export recovered H3 students and run the locked four-call comparison matrix.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
EXPORT_ROOT="${SPRINT_ROOT}/exports/h36-recovery-comparison"
PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/quick_gate_prompts.json"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export HF_XET_CACHE="${SPRINT_ROOT}/caches/hf-xet"
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXPORT_ROOT}" "${HF_XET_CACHE}" "${WANDB_CACHE_DIR}"

export_model() {
  local source_kind="$1"
  local checkpoint="${SPRINT_ROOT}/runs/h36-recovery/${source_kind}-activation/checkpoint-200"
  local export_dir="${EXPORT_ROOT}/${source_kind}-activation-step200-raw"

  test -s "${checkpoint}/metadata.json"
  test -s "${checkpoint}/dcp/.metadata"
  if [[ -s "${export_dir}/export_sha256.txt" ]]; then
    sha256sum --check "${export_dir}/export_sha256.txt"
    echo "$(date -Is) reusing verified ${source_kind} export=${export_dir}"
    return
  fi

  local overwrite_args=()
  if [[ -e "${export_dir}" ]]; then
    echo "$(date -Is) replacing incomplete ${source_kind} export=${export_dir}"
    overwrite_args+=(--overwrite)
  fi

  echo "$(date -Is) exporting ${source_kind} checkpoint=${checkpoint} output=${export_dir}"
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint "${checkpoint}" \
    --output-dir "${export_dir}" \
    --role student \
    "${overwrite_args[@]}" \
    --verify

  test -s "${export_dir}/transformer/model.safetensors"
  test ! -e "${export_dir}/transformer/diffusion_pytorch_model.safetensors.index.json"
  sha256sum \
    "${checkpoint}/metadata.json" \
    "${checkpoint}/dcp/.metadata" \
    "${export_dir}/transformer/model.safetensors" \
    > "${export_dir}/export_sha256.txt"
}

generate_recovery_matrix() {
  local source_kind="$1"
  local attention="$2"
  local export_dir="${EXPORT_ROOT}/${source_kind}-activation-step200-raw"
  local run_id="h36-recovery-compare-${source_kind}-activation-step200-raw"
  local output_dir="${SPRINT_ROOT}/videos/${run_id}"
  local log_dir="${SPRINT_ROOT}/logs/${run_id}"

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse existing comparison directory: ${output_dir}" >&2
    exit 3
  fi
  export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${run_id}"
  export WANDB_DIR="${log_dir}"
  mkdir -p "${TRITON_CACHE_DIR}" "${log_dir}"

  echo "$(date -Is) generating locked matrix ${source_kind} run=${run_id}"
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    scripts/fasth3_sprint/run_baseline_matrix.py \
    --model-path "${export_dir}" \
    --checkpoint-role "recovered-${source_kind}-activation-step200-raw" \
    --attention "${attention}" \
    --prompts "${PROMPTS}" \
    --output-dir "${output_dir}" \
    --run-id "${run_id}" \
    --source-commit "${SOURCE_COMMIT}" \
    --max-prompts 12 \
    --profile strict \
    --no-fa4 \
    --no-compile \
    --no-upload-videos
}

export_model dense
export_model vsa

SHOWCASE_PROMPTS="${REPO_ROOT}/examples/training/fasth3_14b_2step_qad/showcase_prompt.json"
SHOWCASE_SEED=2026082050
BASE_H3="/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe"
FASTH3_V1_VSA="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/snapshots/b65818d41939b5085451074fe8ca8b799f8d4921"

generate_showcase() {
  local model_path="$1"
  local checkpoint_role="$2"
  local attention="$3"
  local steps="$4"
  local run_id="$5"
  local output_dir="${SPRINT_ROOT}/videos/${run_id}"
  local log_dir="${SPRINT_ROOT}/logs/${run_id}"

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse existing showcase directory: ${output_dir}" >&2
    exit 3
  fi
  export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${run_id}"
  export WANDB_DIR="${log_dir}"
  mkdir -p "${TRITON_CACHE_DIR}" "${log_dir}"

  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    scripts/fasth3_sprint/run_baseline_matrix.py \
    --model-path "${model_path}" \
    --checkpoint-role "${checkpoint_role}" \
    --attention "${attention}" \
    --prompts "${SHOWCASE_PROMPTS}" \
    --output-dir "${output_dir}" \
    --run-id "${run_id}" \
    --source-commit "${SOURCE_COMMIT}" \
    --max-prompts 1 \
    --height 480 \
    --width 832 \
    --num-frames 124 \
    --seed "${SHOWCASE_SEED}" \
    --steps "${steps}" \
    --profile strict \
    --no-fa4 \
    --no-compile \
    --no-upload-videos
}

# Same prompt, seed, resolution, frame count, and decode path. Base H3 uses its
# native 50-call quality schedule; FastH3 V1 and both recovered students use the
# released four-call schedule. This is a quality-ceiling comparison, not an
# equal-compute benchmark.
generate_showcase \
  "${BASE_H3}" base-h3-native-50-call dense 51 \
  h36-showcase-base-h3-native50
generate_showcase \
  "${FASTH3_V1_VSA}" fasth3-v1-vsa-4-call vsa 5 \
  h36-showcase-fasth3-v1-vsa-4call
generate_showcase \
  "${EXPORT_ROOT}/dense-activation-step200-raw" recovered-14b-dense-step200-4-call dense 5 \
  h36-showcase-14b-dense-step200-4call
generate_showcase \
  "${EXPORT_ROOT}/vsa-activation-step200-raw" recovered-14b-vsa-step200-4-call vsa 5 \
  h36-showcase-14b-vsa-step200-4call

# Continue with the full locked recovered-student matrices after the four
# showcase outputs are safely persisted.
generate_recovery_matrix dense dense
generate_recovery_matrix vsa vsa

echo "$(date -Is) recovery comparison generation complete"
