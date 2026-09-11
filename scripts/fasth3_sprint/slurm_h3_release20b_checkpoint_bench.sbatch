#!/bin/bash
#SBATCH --job-name=h3-20b-c50-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

# Export one immutable checkpoint from the 20B / 42-block long recovery and
# evaluate it on the exact prompt/seed contract used by the four-way benchmark.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export STEP="${1:?usage: sbatch $0 CHECKPOINT_STEP}"
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export RUN_ROOT="${SPRINT_ROOT}/runs/release20b-folded-long-4k-v3"
export CHECKPOINT="${RUN_ROOT}/checkpoint-${STEP}"
export EVAL_ROOT="${RUN_ROOT}/eval-step-${STEP}-bench-five"
export EXPORT="${EVAL_ROOT}/export-fp32"
export MEDIA="${EVAL_ROOT}/42block-recovery-7902-checkpoint-${STEP}"
export PROMPTS="${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/bench_five_new_prompts.json"
export SEED=20260912

test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${PROMPTS}"
test -s "${CHECKPOINT}/metadata.json"
test -s "${CHECKPOINT}/dcp/.metadata"
test ! -e "${EVAL_ROOT}"
mkdir -p "${EVAL_ROOT}"

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${EVAL_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python

# Materialize directly in the BF16 deployment precision used by the benchmark. The
# single-process exporter otherwise constructs both the 20.14B student and
# full teacher in FP32 and exceeds one H200; the accepted job-6929 export used
# this same BF16 path. Inference reloads the export in a fresh process, which
# is also the end-to-end format check.
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "${CHECKPOINT}" --output-dir "${EXPORT}" \
  --role student --load-precision bf16

"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "${EXPORT}" \
  --checkpoint-role "42block-recovery-7902-checkpoint-${STEP}" \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "${PROMPTS}" --output-dir "${MEDIA}" \
  --run-id "42block-recovery-7902-checkpoint-${STEP}-${SLURM_JOB_ID}" \
  --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" \
  --height 480 --width 832 --num-frames 124 --seed "${SEED}" --steps 50 \
  --num-gpus 4 --dit-precision bf16 --profile strict \
  --no-fa4 --no-compile --no-upload-videos

"${PY}" - <<"PY"
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
media = Path(os.environ["MEDIA"])
clips = sorted(media.glob("*.mp4"))
if len(clips) != 5:
    raise SystemExit(f"expected 5 clips, found {len(clips)}")
payload = {
    "schema_version": 1,
    "training_job_id": 7902,
    "evaluation_job_id": int(os.environ["SLURM_JOB_ID"]),
    "checkpoint_step": int(os.environ["STEP"]),
    "prompt_file": str(Path(os.environ["PROMPTS"])),
    "seed": int(os.environ["SEED"]),
    "height": 480,
    "width": 832,
    "num_frames": 124,
    "inference_steps": 50,
    "attention_backend": "TORCH_SDPA",
    "precision": "bf16",
    "clip_count": len(clips),
    "status": "complete",
}
(root / "evaluation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
date -Is > "${EVAL_ROOT}/completed_at.txt"
'
