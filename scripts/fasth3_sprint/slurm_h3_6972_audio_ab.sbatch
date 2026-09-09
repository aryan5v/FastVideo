#!/bin/bash
# Matched 6972 audio A/B: control versus low-sigma interval sampling.
# Each array arm has a two-hour wall cap and four GB200s (<=16 GPU-hours total).
# Both arms restore the same checkpoint-200 model, Adam state, scheduler, and
# RNG snapshot; only the interval policy differs.
#SBATCH --job-name=h3-6972-audio-ab
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --mem=192G

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/code/ab-audio-b00c624c
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export FULL_INDEX="$SPRINT_ROOT/data/user-prompts-20260906/index-v2"
export SUBSET="$SPRINT_ROOT/data/ab-fresh50-6972"
export SOURCE_CHECKPOINT="$SPRINT_ROOT/runs/activation34-prompt58k-recovery/job-6972/checkpoint-200"
export START_MEDIA="$SPRINT_ROOT/runs/activation34-prompt58k-recovery/job-6972/media-200"
export OUTPUT_ROOT="$SPRINT_ROOT/runs/activation34-6972-audio-ab/job-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) export ARM=control; export LOW_SIGMA_FRACTION=0.0 ;;
  1) export ARM=audio-low-sigma; export LOW_SIGMA_FRACTION=0.5 ;;
  *) exit 2 ;;
esac

test -s "$CODE_ROOT/CODE_COMMIT"
test -s "$SOURCE_CHECKPOINT/dcp/.metadata"
test -s "$SOURCE_CHECKPOINT/metadata.json"
test -s "$START_MEDIA/run_manifest.json"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
  --container-workdir="$CODE_ROOT" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONPATH="$CODE_ROOT:$SPRINT_ROOT/python-packages"
export PYTHONDONTWRITEBYTECODE=1
export FASTVIDEO_MINIMAX_H3_FUSIONS=0
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online WANDB_DIR="$OUTPUT_ROOT"
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python

# Build one immutable 50-record index from records [200, 250), disjoint from
# the first 200 prompt IDs consumed by job 6972. Task 1 waits for task 0.
if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then
  if [[ ! -s "$SUBSET/receipt.json" ]]; then
    mkdir -p "$SUBSET"
    "$PY" - <<PYCODE
import hashlib, json
from pathlib import Path
full = Path("$FULL_INDEX")
subset = Path("$SUBSET")
rows = [json.loads(line) for line in (full / "prompt_index.jsonl").read_text().splitlines() if line.strip()]
consumed_path = Path("$SPRINT_ROOT/runs/activation34-prompt58k-recovery/job-6972/consumed_prompts_rank0.jsonl")
consumed = {json.loads(line)["id"] for line in consumed_path.read_text().splitlines() if line.strip()}
selected = rows[200:250]
ids = {str(row["id"]) for row in selected}
if len(selected) != 50 or len(ids) != 50 or ids & consumed:
    raise RuntimeError("fresh prompt selection is not exactly 50 disjoint records")
raw = "".join(json.dumps(row, sort_keys=True) + "\\n" for row in selected).encode()
(subset / "prompt_index.jsonl").write_bytes(raw)
receipt = json.loads((full / "receipt.json").read_text())
receipt.update({
    "requested": 50,
    "matched": 50,
    "missing": [],
    "conflicts": [],
    "index_sha256": hashlib.sha256(raw).hexdigest(),
    "subset_source": str(full),
    "subset_zero_based_range": [200, 250],
    "disjoint_from_job_6972_consumed": True,
})
(subset / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
print("fresh subset", len(selected), "records", selected[0]["id"], selected[-1]["id"])
PYCODE
  fi
else
  for _ in $(seq 1 120); do [[ -s "$SUBSET/receipt.json" ]] && break; sleep 1; done
fi
test -s "$SUBSET/prompt_index.jsonl"
test -s "$SUBSET/receipt.json"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
  --config "$CODE_ROOT/examples/train/configs/fasth3_6972_audio_ab.yaml" \
  --training.checkpoint.output_dir "$OUTPUT_ROOT" \
  --training.tracker.run_name "activation34-6972-${ARM}-${SLURM_ARRAY_JOB_ID}" \
  --training.data.data_path "$SUBSET" \
  --training.checkpoint.resume_from_checkpoint "$SOURCE_CHECKPOINT" \
  --training.loop.max_train_steps 250 \
  --method.low_sigma_interval_fraction "$LOW_SIGMA_FRACTION"

test -s "$OUTPUT_ROOT/checkpoint-250/dcp/.metadata"
"$PY" -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "$OUTPUT_ROOT/checkpoint-250" \
  --output-dir "$OUTPUT_ROOT/export-250" --role student --load-precision bf16 --verify

"$PY" "$CODE_ROOT/scripts/fasth3_sprint/run_baseline_matrix.py" \
  --model-path "$OUTPUT_ROOT/export-250" \
  --checkpoint-role "activation34-6972-${ARM}-step250" \
  --attention dense --attention-backend TORCH_SDPA \
  --prompts "$CODE_ROOT/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
  --output-dir "$OUTPUT_ROOT/media-250" \
  --run-id "activation34-6972-${ARM}-step250-${SLURM_ARRAY_JOB_ID}" \
  --source-commit "$(cat "$CODE_ROOT/CODE_COMMIT")" --max-prompts 5 \
  --height 480 --width 832 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 \
  --dit-precision bf16 --profile strict --no-fa4 --no-compile --no-upload-videos

ffmpeg -v error -y -i "$OUTPUT_ROOT/media-250/00_speech_exact_presenter.mp4" \
  -vn -ac 2 -ar 32000 -c:a pcm_s16le "$OUTPUT_ROOT/media-250/00_speech_exact_presenter.wav"
"$PY" "$CODE_ROOT/scripts/fasth3_sprint/audio_fidelity_gate.py" \
  --candidate-dir "$OUTPUT_ROOT/media-250" --reference-dir "$START_MEDIA" \
  --receipt-out "$OUTPUT_ROOT/audio-fidelity-vs-6972.json" || true
"$PY" "$CODE_ROOT/scripts/fasth3_sprint/verify_speech_asr.py" \
  --wav "$OUTPUT_ROOT/media-250/00_speech_exact_presenter.wav" \
  --expected "Fast video and clear audio arrive together." \
  --output "$OUTPUT_ROOT/asr-gate.json" \
  --run-id "activation34-6972-${ARM}-step250-${SLURM_ARRAY_JOB_ID}" || true

cat > "$OUTPUT_ROOT/ab-receipt.json" <<JSON
{
  "arm": "$ARM",
  "low_sigma_interval_fraction": $LOW_SIGMA_FRACTION,
  "source_checkpoint": "$SOURCE_CHECKPOINT",
  "source_step": 200,
  "endpoint_step": 250,
  "fresh_prompt_subset": "$SUBSET",
  "fresh_prompt_range_zero_based": [200, 250],
  "matched_noise_stream": true,
  "optimizer_state_restored": true,
  "starting_panel": "$START_MEDIA",
  "endpoint_panel": "$OUTPUT_ROOT/media-250"
}
JSON
date -Is > "$OUTPUT_ROOT/completed_at.txt"
'
