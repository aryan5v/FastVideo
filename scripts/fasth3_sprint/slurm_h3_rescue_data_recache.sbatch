#!/usr/bin/env bash
#SBATCH --job-name=h3-rescue-data
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
SOURCE_ROOT="/mnt/nfs/vlm-aryan/fasth3-33b-20260806/data/h3_corpus"
RAW_MANIFEST="/mnt/nfs/vlm-aryan/fasth3-33b-20260806/data/vggsound_manifest_v2.csv"
MODEL_ROOT="/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe"
DENSE_V1="/mnt/nfs/vlm-aryan/hf-cache/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/f624f08c6c279ab43534c003e556fc5b295b6558"
OUTPUT="${SPRINT_ROOT}/data/h3-rescue-pilot-124f-207a"
PROMPT_POOL="${SPRINT_ROOT}/data/prompt_sources/derived/v1/short_teacher_cache_selection.jsonl"
PROMPT_SPLITS="${SPRINT_ROOT}/data/prompt_sources/derived/v1/splits"
SYNTH_OUTPUT="${SPRINT_ROOT}/data/h3-rescue-synthetic-train64-124f-207a"
RUN_ID="h36-rescue-data-124f-207a-v1"
LOG_DIR="${SPRINT_ROOT}/logs/${RUN_ID}"
mkdir -p "${OUTPUT}" "${LOG_DIR}" "${SPRINT_ROOT}/caches/triton/${RUN_ID}"

source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export HUGGINGFACE_HUB_CACHE=/mnt/nfs/vlm-aryan/hf-cache/hub
export WANDB_MODE=online
export WANDB_CACHE_DIR="${SPRINT_ROOT}/caches/wandb"
export WANDB_DIR="${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}:${SPRINT_ROOT}/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${SPRINT_ROOT}/caches/triton/${RUN_ID}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
echo "$(date -Is) run=${RUN_ID} node=${SLURM_JOB_NODELIST}"
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    "${REPO_ROOT}/scripts/fasth3_sprint/recache_h3_rescue_audio.py" select \
      --source-root "${SOURCE_ROOT}" \
      --raw-manifest "${RAW_MANIFEST}" \
      --output "${OUTPUT}" \
      --max-clips 512
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    "${REPO_ROOT}/scripts/fasth3_sprint/split_h3_prompt_pool.py" \
      --input "${PROMPT_POOL}" \
      --output-dir "${PROMPT_SPLITS}"
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${REPO_ROOT}/scripts/fasth3_sprint/recache_h3_rescue_audio.py" recache \
      --selection "${OUTPUT}/selection_train.jsonl" \
      --source-root "${SOURCE_ROOT}" \
      --output "${OUTPUT}" \
      --model-root "${MODEL_ROOT}"

srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python - "${OUTPUT}" "${RUN_ID}" <<'PY'
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import wandb

root = Path(sys.argv[1])
run_id = sys.argv[2]
records = []
failures = []
for path in sorted(glob.glob(str(root / "manifest_rank*.jsonl"))):
    records.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
for path in sorted(glob.glob(str(root / "failures_rank*.jsonl"))):
    failures.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
assert len(records) == 512, (len(records), failures[:5])
assert not failures, failures[:10]
assert all(r["num_latent_frames"] == 37 and r["num_audio_latents"] == 207 for r in records)
manifest_hash = hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(records, key=lambda x: x["id"])).encode()).hexdigest()
receipt = {
    "run_id": run_id,
    "sample_count": len(records),
    "unique_captions": len({r["caption"] for r in records}),
    "video_latents": 37,
    "audio_latents": 207,
    "manifest_sha256": manifest_hash,
    "persistent_output": str(root.resolve()),
}
(root / "recache_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
run = wandb.init(project="fasth3-14b-2step-qad-sprint", id=run_id, name=run_id, resume="allow", job_type="data-recache", config=receipt)
run.log({"data/sample_count": len(records), "data/unique_captions": receipt["unique_captions"], "data/temporal_alignment_passed": 1})
run.finish()
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

# Add a small, diverse teacher-generated supplement using only the immutable
# training split. Validation and evaluation prompts are never cached here.
srun \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="${REPO_ROOT}" \
  --export=ALL \
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python \
    "${REPO_ROOT}/scripts/fasth3_sprint/build_multishot_calibration_supplement.py" \
      --model-root "${DENSE_V1}" \
      --source-revision f624f08c6c279ab43534c003e556fc5b295b6558 \
      --source-commit "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
      --output-dir "${SYNTH_OUTPUT}" \
      --run-id h36-rescue-synthetic-train64-v1 \
      --prompts-jsonl "${PROMPT_SPLITS}/train.jsonl" \
      --max-prompts 64
