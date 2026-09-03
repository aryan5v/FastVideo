#!/usr/bin/env bash
# Submit the FastH3-native hybrid KD overfit to four 4xGB200 Slinky nodes.

set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
PREVIEW_ROOT=/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-Minimax-FastH3-Preview-v0.2
IMAGE=/mnt/lustre/vlm-wlsaidhi/fastvideo/images/fastvideo-dev-sm100-3da750ded0ec.sqsh
CONFIG_FILE=${WORKTREE_ROOT}/examples/train/configs/overfit_minimax_h3_hybrid_kd_16gpu.yaml
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_16gpu_overfit
SECRET_FILE=/mnt/lustre/vlm-aryan/.secrets/wandb_api_key

[[ -f "${PREVIEW_ROOT}/modular_model_index.json" ]] || { echo "Missing Preview model: ${PREVIEW_ROOT}" >&2; exit 1; }
[[ -f "${SECRET_FILE}" ]] || { echo "Missing W&B secret: ${SECRET_FILE}" >&2; exit 1; }
[[ -f "${IMAGE}" ]] || { echo "Missing runtime image: ${IMAGE}" >&2; exit 1; }

mkdir -p "${RESULT_DIR}/slurm"

# Slinky's topology plugin exposes one allocatable node per topology block to
# a conventional homogeneous request. Four co-scheduled one-node components
# form one heterogeneous allocation and let one srun span all 16 GPUs.
sbatch \
  --job-name=fasth3-hybrid-kd-16g \
  --partition=all \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gres=gpu:4 \
  --cpus-per-task=120 \
  --mem=900G \
  --time=24:00:00 \
  --exclusive \
  --chdir="${WORKTREE_ROOT}" \
  --container-image="${IMAGE}" \
  --container-mounts=/mnt/lustre:/mnt/lustre \
  --output="${RESULT_DIR}/slurm/%x_%j.out" \
  --error="${RESULT_DIR}/slurm/%x_%j.err" \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive \
  --chdir="${WORKTREE_ROOT}" --container-image="${IMAGE}" --container-mounts=/mnt/lustre:/mnt/lustre \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive \
  --chdir="${WORKTREE_ROOT}" --container-image="${IMAGE}" --container-mounts=/mnt/lustre:/mnt/lustre \
  : \
  --partition=all --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=120 --mem=900G --time=24:00:00 --exclusive \
  --chdir="${WORKTREE_ROOT}" --container-image="${IMAGE}" --container-mounts=/mnt/lustre:/mnt/lustre <<'SBATCH_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/lustre/vlm-aryan/fastvideo-h3-hybrid-training
PREVIEW_ROOT=/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-Minimax-FastH3-Preview-v0.2
CONFIG_FILE=${WORKTREE_ROOT}/examples/train/configs/overfit_minimax_h3_hybrid_kd_16gpu.yaml
RESULT_DIR=${WORKTREE_ROOT}/runs/fasth3_hybrid_kd_16gpu_overfit
DATA_DIR=${WORKTREE_ROOT}/data/crush-smol
PARQUET=${WORKTREE_ROOT}/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet
EXPLODED=${RESULT_DIR}/exploded
CONVERTED=${RESULT_DIR}/FastH3-Hybrid

export PYTHONPATH="${WORKTREE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH=/opt/venv/bin:${PATH}
export HF_HOME=${WORKTREE_ROOT}/.cache/huggingface
export WANDB_API_KEY="$(< /mnt/lustre/vlm-aryan/.secrets/wandb_api_key)"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FASTVIDEO_VSA_SM100A=1
export FASTVIDEO_VSA_CUTEDSL=0
export FASTVIDEO_MINIMAX_H3_FUSIONS=0

cd "${WORKTREE_ROOT}"
mkdir -p data/models "${RESULT_DIR}"
if [[ ! -e data/models/MiniMax-H3 ]]; then
  ln -s "${PREVIEW_ROOT}" data/models/MiniMax-H3
fi
if [[ ! -f "${PARQUET}" ]]; then
  hf download wlsaidhi/crush-smol-merged \
    --repo-type dataset \
    --revision 1a850a74e92d5ac3daa273ea658ec60e92fbaf4e \
    --local-dir "${DATA_DIR}"
  torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit
fi
python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit --validate-only

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}") )
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=29541
launch_training_node() {
  export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_PROCID}"
  exec torchrun \
    --nnodes 4 \
    --nproc_per_node 4 \
    --node_rank "${SLURM_PROCID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    fastvideo/train/entrypoint/train.py \
    --config "${CONFIG_FILE}"
}
export -f launch_training_node
srun --het-group=0-3 bash -c launch_training_node

mkdir -p "${CONVERTED}"
for entry in "${PREVIEW_ROOT}"/*; do
  [[ "$(basename "${entry}")" == transformer ]] && continue
  [[ -e "${CONVERTED}/$(basename "${entry}")" ]] || ln -s "${entry}" "${CONVERTED}/$(basename "${entry}")"
done
python scripts/checkpoint_conversion/convert_vdn_h3_to_fastvideo.py \
  --base "${PREVIEW_ROOT}/transformer" \
  --hybrid "${EXPLODED}" \
  --dst "${CONVERTED}/transformer"

srun --het-group=0 --nodes=1 --ntasks=1 --gres=gpu:4 \
  python examples/inference/basic/basic_fasth3.py \
  --model-path "${CONVERTED}" \
  --no-vsa \
  --no-fa4 \
  --profile strict \
  --no-inference-torch-compile \
  --steps 5 \
  --num-gpus 4 \
  --seed 1000 \
  --no-warmup \
  --repeats 1 \
  --prompt "A large metal cylinder presses down on a pile of Oreo cookies under a hydraulic press." \
  --output "${RESULT_DIR}/inference"
SBATCH_SCRIPT
