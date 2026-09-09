#!/bin/bash
#SBATCH --job-name=h3-base34-recut
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
# Re-cut 34-block map (constrained, audio-veto, late-half diversification):
# extract from Base H3, gate step 0, then the detail-band prompt-only recipe
# (arm C) for 200 updates. Compared against job 7002 (same recipe, old map)
# this isolates the map choice at fixed recipe and budget.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export BASE=/mnt/nfs/vlm-aryan/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export DEST="${SPRINT_ROOT}/checkpoints/base34-recut-v1"
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base34-recut-detail-band/job-${SLURM_JOB_ID}"
export REFERENCE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"
test ! -e "${DEST}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/select_h3_block_map.py" \
 --partials-dir "${SPRINT_ROOT}/eval/h6-block-score" --keep-blocks 34 \
 --output "${OUTPUT_ROOT}/recut-map.json"
"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/prune_minimax_h3_blocks.py" \
 --src "${BASE}/transformer" --dst "${DEST}/transformer" \
 --block-map "${OUTPUT_ROOT}/recut-map.json" --strategy constrained-audio-veto-recut \
 --source-model MiniMaxAI/MiniMax-H3 --source-revision 42ed227ee7df40d41602854ae760620d6eb651fe
for component in "${BASE}"/*; do
  name="$(basename "${component}")"
  [[ "${name}" == "transformer" ]] && continue
  ln -sfn "${component}" "${DEST}/${name}"
done
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${DEST}" --checkpoint-role base34-recut-step0 \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-step0" --run-id "base34-recut-step0-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/audio_fidelity_gate.py" \
 --candidate-dir "${OUTPUT_ROOT}/media-step0" --reference-dir "${REFERENCE}" \
 --receipt-out "${OUTPUT_ROOT}/fidelity-step0.json" || true
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" \
 --template "${CODE_ROOT}/examples/train/configs/fasth3_detail_band_recovery34.yaml" \
 --student "${DEST}" --depth 34 --output "${OUTPUT_ROOT}/config.yaml"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
 --config "${OUTPUT_ROOT}/config.yaml" \
 --training.checkpoint.output_dir "${OUTPUT_ROOT}" \
 --training.loop.max_train_steps 200 \
 --training.checkpoint.training_state_checkpointing_steps 100 \
 --method.allow_prompt_only true --method.denoising_weight 0.0 \
 --training.data.data_path "${SPRINT_ROOT}/data/user-prompts-20260906/index-v2" \
 --training.tracker.run_name "base34-recut-detail-band-C-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-200/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-200" \
 --output-dir "${OUTPUT_ROOT}/export-200" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-200" --checkpoint-role base34-recut-detail-band-200 \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-200" --run-id "base34-recut-200-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/audio_fidelity_gate.py" \
 --candidate-dir "${OUTPUT_ROOT}/media-200" --reference-dir "${REFERENCE}" \
 --receipt-out "${OUTPUT_ROOT}/fidelity-gate.json" || true
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
