#!/bin/bash
#SBATCH --job-name=h3-release-long
#SBATCH --partition=all
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export LANE="${1:?usage: sbatch $0 A|B FROM_STEP TO_STEP}"
export FROM_STEP="${2:?FROM_STEP required}"
export TO_STEP="${3:?TO_STEP required}"
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
case "${LANE}" in
  A)
    export STUDENT="${SPRINT_ROOT}/release-candidates/20b-backbone-r768-job6929-v1"
    export OUTPUT_ROOT="${SPRINT_ROOT}/runs/release20b-folded-long-v1"
    export RUN_ID=release20b-folded-long-v1 TOTAL_TARGET=5000
    ;;
  B)
    export STUDENT="${SPRINT_ROOT}/release-candidates/14b-backbone-r16-job7048-v1"
    export OUTPUT_ROOT="${SPRINT_ROOT}/runs/release14b-folded-long-v1"
    export RUN_ID=release14b-folded-long-v1 TOTAL_TARGET=6000
    ;;
  *) echo "LANE must be A or B" >&2; exit 2 ;;
esac
if (( FROM_STEP < 0 || TO_STEP <= FROM_STEP || TO_STEP > TOTAL_TARGET )); then
  echo "invalid phase ${FROM_STEP}->${TO_STEP} for target ${TOTAL_TARGET}" >&2
  exit 2
fi
test -s "${STUDENT}/RECOVERY_ADMISSION.json"
test -s "${STUDENT}/production_media_parity.json"
test -s "${STUDENT}/training-config.yaml"
if (( FROM_STEP > 0 )); then
  test -s "${OUTPUT_ROOT}/checkpoint-${FROM_STEP}/metadata.json"
  test -d "${OUTPUT_ROOT}/checkpoint-${FROM_STEP}/dcp"
fi
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/tracker" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}" MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
export TOTAL_GPUS="$((SLURM_JOB_NUM_NODES * 4))"
(( TOTAL_GPUS == 12 )) || { echo "release recovery requires exactly 12 GPUs" >&2; exit 4; }

srun --export=ALL --kill-on-bad-exit=1 --nodes=3 --ntasks=3 --ntasks-per-node=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}/tracker"
export WANDB_RUN_ID="${RUN_ID}" WANDB_RESUME=allow
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}/node-${SLURM_PROCID}"
mkdir -p "${TRITON_CACHE_DIR}"
resume_args=()
if (( FROM_STEP > 0 )); then
  resume_args=(--training.checkpoint.resume_from_checkpoint "${OUTPUT_ROOT}/checkpoint-${FROM_STEP}")
fi
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
  --nnodes 3 --nproc_per_node 4 --node_rank "${SLURM_PROCID}" \
  --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  -m fastvideo.train.entrypoint.train --config "${STUDENT}/training-config.yaml" \
  --training.distributed.num_gpus 12 --training.distributed.hsdp_replicate_dim 3 \
  --training.loop.max_train_steps "${TO_STEP}" \
  --training.optimizer.lr_scheduler constant --training.optimizer.lr_warmup_steps 0 \
  --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
  --training.checkpoint.training_state_checkpointing_steps 999999 \
  --training.tracker.run_name "${RUN_ID}" "${resume_args[@]}"
'
test -s "${OUTPUT_ROOT}/checkpoint-${TO_STEP}/metadata.json"
test -d "${OUTPUT_ROOT}/checkpoint-${TO_STEP}/dcp"
python3 - "${OUTPUT_ROOT}" <<PY
import json, os, pathlib
root = pathlib.Path(os.environ["OUTPUT_ROOT"])
receipt = {
  "lane": os.environ["LANE"], "from_step": int(os.environ["FROM_STEP"]),
  "to_step": int(os.environ["TO_STEP"]), "slurm_job_id": os.environ["SLURM_JOB_ID"],
  "sp_size": 4, "data_parallel_replicas": 3, "gradient_accumulation": 4,
  "effective_global_batch": 12, "cumulative_prompt_exposures": 12 * int(os.environ["TO_STEP"]),
}
(root / f"phase-{os.environ['FROM_STEP']}-{os.environ['TO_STEP']}.json").write_text(json.dumps(receipt, indent=2)+"\n")
PY
