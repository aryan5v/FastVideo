#!/usr/bin/env bash
#SBATCH --job-name=h3-serve-pr1798
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail

: "${EVAL_ROOT:?EVAL_ROOT is required}"
: "${PR_REPO:?PR_REPO is required}"
: "${PR_COMMIT:?PR_COMMIT is required}"
: "${RUN_ID:?RUN_ID is required}"

HARNESS_ROOT="${EVAL_ROOT}/harness"
: "${PROMPTS_FILE:=${HARNESS_ROOT}/h3_serve_eval_prompts.json}"
: "${EXPECTED_CASES:=10}"
LOG_ROOT="${EVAL_ROOT}/logs/${RUN_ID}"
OUTPUT_ROOT="${EVAL_ROOT}/runs/${RUN_ID}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${EVAL_ROOT}/caches/triton" "${EVAL_ROOT}/caches/wandb" "${EVAL_ROOT}/exports"
exec > >(tee -a "${LOG_ROOT}/driver.log") 2>&1

echo "$(date -Is) job=${SLURM_JOB_ID} run=${RUN_ID} node=${SLURM_JOB_NODELIST} pr_commit=${PR_COMMIT}"
test "$(git -C "${PR_REPO}" rev-parse HEAD)" = "${PR_COMMIT}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${PR_REPO}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_ROOT}
    export WANDB_CACHE_DIR=${EVAL_ROOT}/caches/wandb
    export PYTHONPATH=${EVAL_ROOT}/python-packages:${PR_REPO}:/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/python-packages
    export PYTHONDONTWRITEBYTECODE=1
    export TOKENIZERS_PARALLELISM=false
    export TRITON_CACHE_DIR=${EVAL_ROOT}/caches/triton/${RUN_ID}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250
    export FASTVIDEO_VSA_SM100A=1
    export FASTVIDEO_VSA_CUTEDSL=0
    unset FASTVIDEO_H3_VSA_PROBE
    export FASTVIDEO_FA4=0
    export FASTVIDEO_NVFP4_FA4=0
    export FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN=0
    export FASTVIDEO_MINIMAX_H3_FUSIONS=0
    export FASTVIDEO_INFERENCE_TORCH_COMPILE=0
    export FASTVIDEO_VAE_PARALLEL_DECODE=1
    export FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY=gather
    mkdir -p \"\${TRITON_CACHE_DIR}\"

    if ! /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -c 'import openai; assert openai.__version__ == \"3.6.0\"'; then
      /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pip install \
        --disable-pip-version-check \
        --target ${EVAL_ROOT}/python-packages \
        'openai==3.6.0'
    fi
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -c \
      'import openai; print(\"openai_sdk_version=\" + openai.__version__)' | tee ${LOG_ROOT}/openai-sdk.txt

    nvidia-smi -L | tee ${LOG_ROOT}/nvidia-smi-L.txt
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader | tee ${LOG_ROOT}/hardware.csv
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pytest \
      fastvideo/tests/entrypoints/test_openai_video_client.py \
      tests/local_tests/test_cookbook_serving.py \
      -q | tee ${LOG_ROOT}/cookbook-tests.log

    server_pid=
    cleanup() {
      if [[ -n \"\${server_pid}\" ]] && kill -0 \"\${server_pid}\" 2>/dev/null; then
        kill -TERM \"\${server_pid}\" 2>/dev/null || true
        wait \"\${server_pid}\" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT

    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m fastvideo.entrypoints.cli.main serve \
      --config ${HARNESS_ROOT}/openai_fasth3_v1_eval.yaml \
      > ${LOG_ROOT}/server.log 2>&1 &
    server_pid=\$!
    echo \"\$(date -Is) server_pid=\${server_pid}\"
    sleep 10
    if ! kill -0 \"\${server_pid}\" 2>/dev/null; then
      echo \"Server process exited before becoming ready\" >&2
      tail -n 200 ${LOG_ROOT}/server.log >&2 || true
      exit 1
    fi

    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
      ${HARNESS_ROOT}/run_h3_serve_cookbook_eval.py \
      --base-url http://127.0.0.1:8000 \
      --model fasth3-v1 \
      --prompts ${PROMPTS_FILE} \
      --output-root ${OUTPUT_ROOT} \
      --model-id FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
      --model-revision b65818d41939b5085451074fe8ca8b799f8d4921 \
      --source-commit ${PR_COMMIT} \
      --run-id ${RUN_ID} \
      --expected-cases ${EXPECTED_CASES}

    test \"\$(find ${OUTPUT_ROOT}/videos -type f -name '*.mp4' | wc -l)\" -eq ${EXPECTED_CASES}
    find ${OUTPUT_ROOT} -type f -print0 | sort -z | xargs -0 sha256sum > ${OUTPUT_ROOT}/SHA256SUMS
    tar -C ${OUTPUT_ROOT} -czf ${EVAL_ROOT}/exports/${RUN_ID}-metadata.tar.gz receipts summary.json SHA256SUMS
    echo \"\$(date -Is) complete videos=${EXPECTED_CASES} summary=${OUTPUT_ROOT}/summary.json\"
  "
