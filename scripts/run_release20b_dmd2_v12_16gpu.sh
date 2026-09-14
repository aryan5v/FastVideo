#!/bin/bash
# Compute-node payload for a four-node / 16-GPU Release-20B DMD2 V12 pilot.
# Run as one top-level srun task per four-GPU node. The payload first proves
# four critic updates and one student update, then resumes in the same
# allocation to 1,000 phases. Intermediate checkpoints are promotion gates,
# not stop/requeue boundaries.
set -euo pipefail

: "${CODE_ROOT:?Set CODE_ROOT to the immutable execution checkout}"
: "${SELECTED_PARENT:?Set SELECTED_PARENT to the gated BF16 parent export}"
: "${TEACHER_PARENT:?Set TEACHER_PARENT to the full base H3 checkpoint}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to a fresh DMD2 namespace}"

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
CONFIG_PATH="${CODE_ROOT}/examples/train/configs/distribution_matching/minimax_h3/release20b_dmd2_v12_dense.yaml"
OUTPUT_ROOT="${OUTPUT_BASE}/job-${SLURM_JOB_ID}"
export SPRINT_ROOT CONFIG_PATH OUTPUT_ROOT

test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${CONFIG_PATH}"
test -s "${SELECTED_PARENT}/transformer/config.json"
test -s "${SELECTED_PARENT}/transformer/model.safetensors"
test -s "${TEACHER_PARENT}/transformer/config.json"
test -s "${TEACHER_PARENT}/transformer/diffusion_pytorch_model.safetensors.index.json"

if [[ "${SLURM_PROCID}" == "0" ]]; then
  test ! -e "${OUTPUT_ROOT}"
  mkdir -p "${OUTPUT_ROOT}"
else
  for _ in $(seq 1 120); do
    [[ -d "${OUTPUT_ROOT}" ]] && break
    sleep 1
  done
  test -d "${OUTPUT_ROOT}"
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR="${nodes[0]}"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
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
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python

# Run the exact V12 math/config suite once on the head node. Other node
# launchers wait for its receipt so a failed unit contract cannot fall through
# into model loading or consume training steps.
if [[ "${SLURM_PROCID}" == "0" ]]; then
  "${PY}" -m pytest -q \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_fastgen_parity.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_fake_score_loss_space.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_rollout_carry.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_timestep_bounds.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_dmd2_vsd_normalizer.py" \
    "${CODE_ROOT}/fastvideo/tests/train/methods/test_minimax_h3_dmd2.py" \
    | tee "${OUTPUT_ROOT}/preflight-pytest.log"
  "${PY}" - "${CONFIG_PATH}" "${SELECTED_PARENT}" "${OUTPUT_ROOT}" <<'PY'
import sys
from fastvideo.train.utils.config import load_run_config

path, parent, output = sys.argv[1:]
cfg = load_run_config(path, [
    "--models.student.init_from", parent,
    "--training.checkpoint.output_dir", output,
])
assert cfg.training.distributed.num_gpus == 16
assert cfg.training.distributed.sp_size == 4
assert cfg.training.distributed.hsdp_shard_dim == 16
assert cfg.training.loop.gradient_accumulation_steps == 16
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
    --nnodes "${SLURM_JOB_NUM_NODES}" --nproc_per_node 4 \
    --node_rank "${SLURM_PROCID}" --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${port}" \
    -m fastvideo.train.entrypoint.train --config "${CONFIG_PATH}" \
    --models.student.init_from "${SELECTED_PARENT}" \
    --models.teacher.init_from "${TEACHER_PARENT}" \
    --models.critic.init_from "${TEACHER_PARENT}" \
    --training.loop.max_train_steps "${target}" \
    --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
    --training.checkpoint.training_state_checkpointing_steps "${checkpoint_every}" \
    --training.checkpoint.checkpointing_start_step "${checkpoint_start}" \
    --training.tracker.run_name "${name}-${SLURM_JOB_ID}" \
    "${resume_args[@]}"
}

train_phase 5 5 5 "" "${MASTER_PORT}" release20b-dmd2-v12-smoke
touch "${OUTPUT_ROOT}/.phase5-node-${SLURM_PROCID}"

if [[ "${SLURM_PROCID}" == "0" ]]; then
  for _ in $(seq 1 180); do
    [[ "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name '.phase5-node-*' | wc -l)" -eq "${SLURM_JOB_NUM_NODES}" ]] && break
    sleep 5
  done
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name '.phase5-node-*' | wc -l)" -eq "${SLURM_JOB_NUM_NODES}"
  test -s "${OUTPUT_ROOT}/checkpoint-5/dcp/.metadata"
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name 'dmd2_update_critic_rank*.json' | wc -l)" -eq 16
  test "$(find "${OUTPUT_ROOT}" -maxdepth 1 -name 'dmd2_update_student_rank*.json' | wc -l)" -eq 16
  "${PY}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipts = sorted(root.glob("dmd2_update_*_rank*.json"))
payloads = [json.loads(path.read_text()) for path in receipts]
assert len(payloads) == 32
assert all(row["passed"] and row["changed_probe_elements"] > 0 for row in payloads)
(root / "dmd2_smoke_passed.json").write_text(json.dumps({
    "phases": 5,
    "critic_updates": 4,
    "student_updates": 1,
    "ranks": 16,
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
train_phase 1000 100 100 "${OUTPUT_ROOT}/checkpoint-5" "$((MASTER_PORT + 1))" release20b-dmd2-v12-production
