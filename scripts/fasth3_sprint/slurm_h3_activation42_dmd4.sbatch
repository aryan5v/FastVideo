#!/bin/bash
#SBATCH --job-name=h3-a42-dmd4
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/activation42-dmd4/job-${SLURM_JOB_ID}"
export CONFIG_PATH="${CODE_ROOT}/examples/train/configs/fasth3_activation42_dmd4.yaml"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${SPRINT_ROOT}/preserved/activation42-step500-job6878/export-500/transformer/model.safetensors"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
# Five phases prove four critic updates and one student update, then checkpoint/reload.
for target in 5 500; do
 export TARGET_STEPS="${target}"
 srun --export=ALL --kill-on-bad-exit=1 --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
resume_args=()
if [[ "${TARGET_STEPS}" == 500 ]]; then
 resume_args=(--training.checkpoint.resume_from_checkpoint "${OUTPUT_ROOT}/checkpoint-5")
fi
exec /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
 --nnodes "${SLURM_JOB_NUM_NODES}" --nproc_per_node 4 --node_rank "${SLURM_PROCID}" \
 --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
 -m fastvideo.train.entrypoint.train --config "${CONFIG_PATH}" \
 --training.loop.max_train_steps "${TARGET_STEPS}" \
 --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
 --training.checkpoint.training_state_checkpointing_steps "$([[ "${TARGET_STEPS}" == 5 ]] && echo 5 || echo 100)" \
 --training.tracker.run_name "activation42-dmd4-${SLURM_JOB_ID}-to${TARGET_STEPS}" "${resume_args[@]}"
'
 if [[ "$target" == 5 ]]; then
  python3 - "$OUTPUT_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert (p/'checkpoint-5/dcp/.metadata').is_file()
for role in ['student','critic']:
 for rank in range(8):
  d=json.loads((p/f'dmd_update_{role}_rank{rank}.json').read_text())
  assert d['passed'] and d['changed_probe_elements']>0
(p/'dmd_smoke_passed.json').write_text(json.dumps({'phases':5,'critic_updates':4,'student_updates':1,'ranks':8}))
PY
 fi
done
srun --nodes=1 --ntasks=1 --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"$PY" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-500" \
 --output-dir "${OUTPUT_ROOT}/export-500" --role student --load-precision bf16 --verify
"$PY" scripts/fasth3_sprint/run_baseline_matrix.py --model-path "${OUTPUT_ROOT}/export-500" \
 --checkpoint-role activation42-dmd4 --attention dense --attention-backend TORCH_SDPA \
 --prompts examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json \
 --output-dir "${OUTPUT_ROOT}/media-500" --run-id "activation42-dmd4-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 4 --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 5 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
'
