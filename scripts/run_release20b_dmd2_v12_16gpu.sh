#!/bin/bash
# Compute-node payload for a Release-20B DMD2 V12 run.
# Run as one top-level srun task per four-GPU node. The payload first proves
# four critic updates and one student update, then resumes in the same
# allocation to the requested target. Intermediate checkpoints are promotion gates,
# not stop/requeue boundaries.
set -euo pipefail

: "${CODE_ROOT:?Set CODE_ROOT to the immutable execution checkout}"
: "${SELECTED_PARENT:?Set SELECTED_PARENT to the gated BF16 parent export}"
: "${TEACHER_PARENT:?Set TEACHER_PARENT to the full base H3 checkpoint}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to a fresh DMD2 namespace}"
: "${MASTER_ADDR:?Resolve MASTER_ADDR on the SLURM host before entering the container}"
: "${MASTER_PORT:?Resolve MASTER_PORT on the SLURM host before entering the container}"

NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-4}}}"
NODE_RANK_BASE="${NODE_RANK_BASE:-0}"
NODE_RANK="$((NODE_RANK_BASE + SLURM_PROCID))"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
PRODUCTION_TARGET="${PRODUCTION_TARGET:-1000}"
NUM_GPUS="$((NNODES * 4))"
if [[ "${NUM_GPUS}" -ne 16 && "${NUM_GPUS}" -ne 32 ]]; then
  echo "Expected 16 or 32 GPUs at four GPUs per node, got ${NUM_GPUS}" >&2
  exit 2
fi
if [[ "${PRODUCTION_TARGET}" -lt 100 || "$((PRODUCTION_TARGET % 100))" -ne 0 ]]; then
  echo "PRODUCTION_TARGET must be a multiple of 100 and at least 100, got ${PRODUCTION_TARGET}" >&2
  exit 2
fi
GRAD_ACCUM="$((256 / NUM_GPUS))"
if [[ "${NODE_RANK}" -lt 0 || "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "Invalid node rank ${NODE_RANK} for ${NNODES} nodes" >&2
  exit 2
fi
export NNODES NODE_RANK RUN_ID NUM_GPUS GRAD_ACCUM

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
CONFIG_PATH="${CODE_ROOT}/examples/train/configs/distribution_matching/minimax_h3/release20b_dmd2_v12_dense.yaml"
OUTPUT_ROOT="${OUTPUT_BASE}/job-${RUN_ID}"
export SPRINT_ROOT CONFIG_PATH OUTPUT_ROOT

test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${CONFIG_PATH}"
test -s "${SELECTED_PARENT}/transformer/config.json"
test -s "${SELECTED_PARENT}/transformer/model.safetensors"
test -s "${TEACHER_PARENT}/transformer/config.json"
test -s "${TEACHER_PARENT}/transformer/diffusion_pytorch_model.safetensors.index.json"

if [[ "${NODE_RANK}" == "0" ]]; then
  test ! -e "${OUTPUT_ROOT}"
  mkdir -p "${OUTPUT_ROOT}"
else
  for _ in $(seq 1 120); do
    [[ -d "${OUTPUT_ROOT}" ]] && break
    sleep 1
  done
  test -d "${OUTPUT_ROOT}"
fi

export MASTER_ADDR MASTER_PORT

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export FASTVIDEO_FA4=0
export FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0
# NVLS has intermittently failed during communicator creation on these GB200
# allocations.  Ordinary NCCL collectives are slower only at startup scale and
# avoid losing an otherwise healthy four-node allocation to a transport fault.
export NCCL_NVLS_ENABLE=0
# The Slinky rack-2 trays have also produced CUDA error 400 while creating
# NCCL's direct P2P transport.  The established H3 launchers disable this
# path and use the stable shared-memory/network transports instead.
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python

# Run the exact V12 math/config suite once on the head node. Other node
# launchers wait for its receipt so a failed unit contract cannot fall through
# into model loading or consume training steps.
if [[ "${NODE_RANK}" == "0" ]]; then
  "${PY}" -m pytest -q \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_fastgen_parity.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_fake_score_loss_space.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_rollout_carry.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_timestep_bounds.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_vsd_normalizer.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_minimax_h3_dmd2.py" \
    | tee "${OUTPUT_ROOT}/preflight-pytest.log"
  "${PY}" - "${CONFIG_PATH}" "${SELECTED_PARENT}" "${OUTPUT_ROOT}" "${NUM_GPUS}" "${GRAD_ACCUM}" <<'PY'
import sys
from fastvideo.train.utils.config import load_run_config

path, parent, output, num_gpus, grad_accum = sys.argv[1:]
num_gpus, grad_accum = int(num_gpus), int(grad_accum)
cfg = load_run_config(path, [
    "--models.student.init_from", parent,
    "--training.checkpoint.output_dir", output,
    "--training.distributed.num_gpus", str(num_gpus),
    "--training.distributed.hsdp_shard_dim", str(num_gpus),
    "--training.loop.gradient_accumulation_steps", str(grad_accum),
])
assert cfg.training.distributed.num_gpus == num_gpus
assert cfg.training.distributed.sp_size == 4
assert cfg.training.distributed.hsdp_shard_dim == num_gpus
assert cfg.training.loop.gradient_accumulation_steps == grad_accum
assert cfg.method["generator_update_interval"] == 5
assert cfg.method["dmd_denoising_steps"] == [999, 749, 500, 250]
assert cfg.method["fake_score_loss_space"] == "x0"
assert cfg.models["critic"]["init_from"] != parent
PY
  touch "${OUTPUT_ROOT}/.preflight-passed"
fi

for _ in $(seq 1 180); do
  [[ -e "${OUTPUT_ROOT}/.preflight-passed" ]] && break
  sleep 5
done
test -e "${OUTPUT_ROOT}/.preflight-passed"

train_phase() {
  local target="$1" checkpoint_every="$2" checkpoint_start="$3" resume="$4" port="$5" name="$6"
  local resume_args=()
  if [[ -n "${resume}" ]]; then
    resume_args=(--training.checkpoint.resume_from_checkpoint "${resume}")
  fi
  "${PY}" -m torch.distributed.run \
    --nnodes "${NNODES}" --nproc_per_node 4 \
    --node_rank "${NODE_RANK}" --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${port}" \
    -m fastvideo.train.entrypoint.train --config "${CONFIG_PATH}" \
    --models.student.init_from "${SELECTED_PARENT}" \
    --models.teacher.init_from "${TEACHER_PARENT}" \
    --models.critic.init_from "${TEACHER_PARENT}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}" \
    --training.loop.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --training.loop.max_train_steps "${target}" \
    --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
    --training.checkpoint.training_state_checkpointing_steps "${checkpoint_every}" \
    --training.checkpoint.checkpointing_start_step "${checkpoint_start}" \
    --training.tracker.run_name "${name}-${RUN_ID}" \
    "${resume_args[@]}"
}

train_phase 5 5 5 "" "${MASTER_PORT}" release20b-dmd2-v12-smoke
touch "${OUTPUT_ROOT}/.phase5-node-${NODE_RANK}"

if [[ "${NODE_RANK}" == "0" ]]; then
  for _ in $(seq 1 180); do
    [[ "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name '.phase5-node-*' | wc -l)" -eq "${NNODES}" ]] && break
    sleep 5
  done
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name '.phase5-node-*' | wc -l)" -eq "${NNODES}"
  test -s "${OUTPUT_ROOT}/checkpoint-5/dcp/.metadata"
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name 'dmd2_update_critic_rank*.json' | wc -l)" -eq "${NUM_GPUS}"
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name 'dmd2_update_student_rank*.json' | wc -l)" -eq "${NUM_GPUS}"
  "${PY}" - "${OUTPUT_ROOT}" "${NUM_GPUS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_gpus = int(sys.argv[2])
receipts = sorted(root.glob("dmd2_update_*_rank*.json"))
payloads = [json.loads(path.read_text()) for path in receipts]
assert len(payloads) == 2 * num_gpus
assert all(row["passed"] and row["changed_probe_elements"] > 0 for row in payloads)
(root / "dmd2_smoke_passed.json").write_text(json.dumps({
    "phases": 5,
    "critic_updates": 4,
    "student_updates": 1,
    "ranks": num_gpus,
    "finite_fp32_updates": True,
}, indent=2) + "\n")
PY
  touch "${OUTPUT_ROOT}/.phase5-passed"
fi

for _ in $(seq 1 180); do
  [[ -e "${OUTPUT_ROOT}/.phase5-passed" ]] && break
  sleep 5
done
test -e "${OUTPUT_ROOT}/.phase5-passed"

# Continue uninterrupted after the contract smoke. Validation and immutable
# checkpoints are produced every 100 phases (20 student updates); selection
# is based on those checkpoints rather than assuming phase 1,000 is best.
train_phase "${PRODUCTION_TARGET}" 100 100 "${OUTPUT_ROOT}/checkpoint-5" "$((MASTER_PORT + 1))" release20b-dmd2-v12-production
