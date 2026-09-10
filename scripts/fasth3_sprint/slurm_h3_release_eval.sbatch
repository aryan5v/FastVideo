#!/bin/bash
#SBATCH --job-name=h3-release-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export LANE="${1:?usage: sbatch $0 A|B|C STEP}"
export STEP="${2:?checkpoint step required}"
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
case "${LANE}" in
  A) export RUN_ROOT="${SPRINT_ROOT}/runs/release20b-folded-long-v1" INFERENCE_STEPS=50 ROLE=release20b ;;
  B) export RUN_ROOT="${SPRINT_ROOT}/runs/release14b-folded-long-v1" INFERENCE_STEPS=50 ROLE=release14b ;;
  C) export RUN_ROOT="${SPRINT_ROOT}/runs/release14b-4c-aware-recovery-v1" INFERENCE_STEPS=5 ROLE=release14b-4c ;;
  *) echo "LANE must be A, B, or C" >&2; exit 2 ;;
esac
export CHECKPOINT="${RUN_ROOT}/checkpoint-${STEP}"
export EVAL_ROOT="${RUN_ROOT}/eval-step-${STEP}"
export EXPORT="${EVAL_ROOT}/export-fp32"
test -s "${CHECKPOINT}/metadata.json"
test -d "${CHECKPOINT}/dcp"
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
# Keep folded modulation information intact. Blanket BF16 export loses basis
# precision before the production loader can apply its FP16 AdaLN policy.
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "${CHECKPOINT}" --output-dir "${EXPORT}" \
  --role student --load-precision fp32 --verify
for SEED in 2026091001 2026091002; do
  MEDIA="${EVAL_ROOT}/media-seed-${SEED}"
  "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
    --model-path "${EXPORT}" --checkpoint-role "${ROLE}-step${STEP}" \
    --attention dense --attention-backend TORCH_SDPA \
    --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/release_sentinel_24.json" \
    --output-dir "${MEDIA}" --run-id "${ROLE}-step${STEP}-seed${SEED}" \
    --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" \
    --height 480 --width 832 --num-frames 124 --seed "${SEED}" \
    --steps "${INFERENCE_STEPS}" --num-gpus 4 --dit-precision bf16 \
    --profile strict --no-fa4 --no-compile --no-upload-videos
  mkdir -p "${EVAL_ROOT}/review-seed-${SEED}"
  for MEDIA_FILE in "${MEDIA}"/*.mp4; do
    "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/extract_review_assets.py" \
      "${MEDIA_FILE}" --output-dir "${EVAL_ROOT}/review-seed-${SEED}" --frames 8
  done
  "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/verify_speech_asr.py" \
    --wav "${EVAL_ROOT}/review-seed-${SEED}/00_speech_en_exact.wav" \
    --expected "Fast video and clear audio arrive together." \
    --output "${EVAL_ROOT}/asr-seed-${SEED}.json" \
    --run-id "${ROLE}-step${STEP}-asr-seed${SEED}" --max-wer 0.25 || true
done
"${PY}" - "${EVAL_ROOT}" <<"PY"
import json, os, pathlib
root = pathlib.Path(os.environ["EVAL_ROOT"])
asr = [json.loads((root / f"asr-seed-{seed}.json").read_text()) for seed in (2026091001, 2026091002)]
review = sorted(root.glob("review-seed-*/*_review_manifest.json"))
payload = {
    "schema_version": 1,
    "lane": os.environ["LANE"],
    "step": int(os.environ["STEP"]),
    "seeds": [2026091001, 2026091002],
    "prompts_per_seed": 24,
    "media_count": 48,
    "review_manifest_count": len(review),
    "speech_wer": [item["word_error_rate"] for item in asr],
    "speech_gate_passed": all(item["passed"] for item in asr),
    "status": "awaiting_blind_media_review",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
(root / "evaluation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if len(review) != 48 or not payload["speech_gate_passed"]:
    raise SystemExit(3)
PY
'
