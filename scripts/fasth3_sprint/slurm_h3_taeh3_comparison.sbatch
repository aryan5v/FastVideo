#!/usr/bin/env bash
#SBATCH --job-name=h3-taeh3-pair
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

set -euo pipefail

EVAL_ROOT="${EVAL_ROOT:-/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831}"
MAIN_REPO="${MAIN_REPO:-${EVAL_ROOT}/repo-main-3d8ac9d1}"
SOURCE_COMMIT="${SOURCE_COMMIT:-3d8ac9d14bd697a89ede8f170cbfbca012a9edcc}"
HARNESS_ROOT="${HARNESS_ROOT:-${EVAL_ROOT}/harness}"
PAIR_ID="${PAIR_ID:-${1:?PAIR_ID argument is required when PAIR_ID is unset}}"

LOG_ROOT="${EVAL_ROOT}/logs/${PAIR_ID}"
mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}/caches/triton" "${EVAL_ROOT}/caches/wandb" \
  "${EVAL_ROOT}/checkpoints" "${EVAL_ROOT}/exports" "${EVAL_ROOT}/server_outputs"
exec > >(tee -a "${LOG_ROOT}/driver.log") 2>&1

echo "$(date -Is) job=${SLURM_JOB_ID} pair=${PAIR_ID} node=${SLURM_JOB_NODELIST} source=${SOURCE_COMMIT}"
test "$(git -C "${MAIN_REPO}" rev-parse HEAD)" = "${SOURCE_COMMIT}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${MAIN_REPO}" \
  bash -lc "
    set -euo pipefail
    source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
    export HOME=${EVAL_ROOT}/home
    export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
    export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=online
    export WANDB_DIR=${LOG_ROOT}
    export WANDB_CACHE_DIR=${EVAL_ROOT}/caches/wandb
    export PYTHONPATH=${EVAL_ROOT}/python-packages:${MAIN_REPO}:/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/python-packages
    export PYTHONDONTWRITEBYTECODE=1
    export TOKENIZERS_PARALLELISM=false
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
    mkdir -p \"\${HOME}/.cache/fastvideo/taehv\" ${EVAL_ROOT}/checkpoints

    checkpoint=${EVAL_ROOT}/checkpoints/taeh3.safetensors
    if [[ ! -f \"\${checkpoint}\" ]]; then
      curl --fail --location --retry 3 \
        --output \"\${checkpoint}.partial\" \
        https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors
      mv \"\${checkpoint}.partial\" \"\${checkpoint}\"
    fi
    echo '4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13  '${EVAL_ROOT}'/checkpoints/taeh3.safetensors' | sha256sum --check
    ln -sfn \"\${checkpoint}\" \"\${HOME}/.cache/fastvideo/taehv/taeh3.safetensors\"

    nvidia-smi -L | tee ${LOG_ROOT}/nvidia-smi-L.txt
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader | tee ${LOG_ROOT}/hardware.csv
    /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m pytest \
      fastvideo/tests/vaes/test_minimax_h3_taeh3.py \
      fastvideo/tests/stages/test_minimax_h3_sequential_start.py \
      -q | tee ${LOG_ROOT}/taeh3-tests.log

    server_pid=
    cleanup_server() {
      if [[ -n \"\${server_pid}\" ]] && kill -0 \"\${server_pid}\" 2>/dev/null; then
        kill -TERM -- -\"\${server_pid}\" 2>/dev/null || true
        wait \"\${server_pid}\" 2>/dev/null || true
      fi
      server_pid=
    }
    trap cleanup_server EXIT

    run_backend() {
      backend=\"\$1\"
      config=\"\$2\"
      run_id=${PAIR_ID}-\"\${backend}\"
      backend_log=${LOG_ROOT}/\"\${backend}\"
      output_root=${EVAL_ROOT}/runs/\"\${run_id}\"
      mkdir -p \"\${backend_log}\" \"\${output_root}\"
      export TRITON_CACHE_DIR=${EVAL_ROOT}/caches/triton/\"\${run_id}\"
      mkdir -p \"\${TRITON_CACHE_DIR}\"

      echo \"\$(date -Is) backend=\${backend} phase=server_start config=\${config}\"
      setsid /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m fastvideo.entrypoints.cli.main serve \
        --config \"\${config}\" > \"\${backend_log}/server.log\" 2>&1 &
      server_pid=\$!
      echo \"\$(date -Is) backend=\${backend} server_pid=\${server_pid}\"
      sleep 10
      if ! kill -0 \"\${server_pid}\" 2>/dev/null; then
        echo \"Server process exited before becoming ready for \${backend}\" >&2
        tail -n 200 \"\${backend_log}/server.log\" >&2 || true
        exit 1
      fi

      /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
        ${HARNESS_ROOT}/run_h3_serve_cookbook_eval.py \
        --base-url http://127.0.0.1:8000 \
        --model fasth3-v1 \
        --prompts ${HARNESS_ROOT}/h3_taeh3_kitchen_prompt.json \
        --output-root \"\${output_root}\" \
        --model-id FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
        --model-revision b65818d41939b5085451074fe8ca8b799f8d4921 \
        --source-commit ${SOURCE_COMMIT} \
        --run-id \"\${run_id}\" \
        --video-decode-backend \"\${backend}\" \
        --expected-cases 2

      test \"\$(find \"\${output_root}/videos\" -type f -name '*.mp4' | wc -l)\" -eq 2
      find \"\${output_root}\" -type f -print0 | sort -z | xargs -0 sha256sum > \"\${output_root}/SHA256SUMS\"
      tar -C \"\${output_root}\" -czf ${EVAL_ROOT}/exports/\"\${run_id}-metadata.tar.gz\" receipts summary.json SHA256SUMS
      echo \"\$(date -Is) backend=\${backend} phase=complete summary=\${output_root}/summary.json\"
      cleanup_server
      for _ in {1..30}; do
        if ! curl --silent --max-time 1 http://127.0.0.1:8000/health >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
    }

    run_backend h3-vae ${HARNESS_ROOT}/openai_fasth3_h3_vae_comparison.yaml
    run_backend taeh3 ${HARNESS_ROOT}/openai_fasth3_taeh3_comparison.yaml
    echo \"\$(date -Is) pair=${PAIR_ID} phase=complete\"
  "
