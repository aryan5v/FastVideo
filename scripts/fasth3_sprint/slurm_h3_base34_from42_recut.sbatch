#!/bin/bash
#SBATCH --job-name=h3-34-from42-recut
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
# Re-cut the recovered 42-block winner (job-6929 export-200) to 34 blocks with
# the constrained selector (audio veto, no gap stretching, late-half quota),
# gate step 0, then the detail-band prompt-only 58k recipe for 200 updates.
# Salvages the 42-block recovery work instead of restarting from Base H3.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
export PARENT="${SPRINT_ROOT}/runs/base42-prompt58k-recovery/job-6929/export-200"
export DEST="${SPRINT_ROOT}/checkpoints/base34-from42-recut-v1"
export OUTPUT_ROOT="${SPRINT_ROOT}/runs/base34-from42-recut-detail-band/job-${SLURM_JOB_ID}"
export REFERENCE="${SPRINT_ROOT}/diagnostics/base-h3-controls/6871-TORCH_SDPA"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${PARENT}/transformer/block_map_manifest.json"
test ! -e "${DEST}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=online WANDB_DIR="${OUTPUT_ROOT}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
cat > "${OUTPUT_ROOT}/base42-map.json" <<JSON
{"block_map": [0,1,2,3,4,5,6,11,12,13,14,17,18,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]}
JSON
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/select_h3_block_map.py" \
 --partials-dir "${SPRINT_ROOT}/eval/h6-block-score" --keep-blocks 34 \
 --base-map "${OUTPUT_ROOT}/base42-map.json" \
 --output "${OUTPUT_ROOT}/recut-map.json"
"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/prune_minimax_h3_blocks.py" \
 --src "${PARENT}/transformer" --dst "${DEST}/transformer" \
 --block-map "${OUTPUT_ROOT}/recut-map.local.json" --strategy constrained-audio-veto-recut-from42 \
 --source-model MiniMaxAI/MiniMax-H3 --source-revision 42ed227ee7df40d41602854ae760620d6eb651fe
for component in "${PARENT}"/*; do
  name="$(basename "${component}")"
  [[ "${name}" == "transformer" ]] && continue
  ln -sfn "${component}" "${DEST}/${name}"
done
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${DEST}" --checkpoint-role base34-from42-recut-step0 \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-step0" --run-id "base34-from42-step0-${SLURM_JOB_ID}" \
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
 --training.tracker.run_name "base34-from42-recut-C-${SLURM_JOB_ID}"
test -s "${OUTPUT_ROOT}/checkpoint-200/dcp/.metadata"
"${PY}" -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "${OUTPUT_ROOT}/checkpoint-200" \
 --output-dir "${OUTPUT_ROOT}/export-200" --role student --load-precision bf16 --verify
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
 --model-path "${OUTPUT_ROOT}/export-200" --checkpoint-role base34-from42-recut-200 \
 --attention dense --attention-backend TORCH_SDPA \
 --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/comparison_five_prompts.json" \
 --output-dir "${OUTPUT_ROOT}/media-200" --run-id "base34-from42-200-${SLURM_JOB_ID}" \
 --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --height 480 --width 832 \
 --num-frames 124 --seed 2026082050 --steps 50 --num-gpus 4 --dit-precision bf16 \
 --profile strict --no-fa4 --no-compile --no-upload-videos
ffmpeg -v error -y -i "${OUTPUT_ROOT}/media-200/00_speech_exact_presenter.mp4" -vn -ac 2 -ar 32000 \
 -c:a pcm_s16le "${OUTPUT_ROOT}/media-200/00_speech_exact_presenter.wav"
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/audio_fidelity_gate.py" \
 --candidate-dir "${OUTPUT_ROOT}/media-200" --reference-dir "${REFERENCE}" \
 --receipt-out "${OUTPUT_ROOT}/fidelity-gate.json" || true
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/verify_speech_asr.py" \
 --wav "${OUTPUT_ROOT}/media-200/00_speech_exact_presenter.wav" \
 --expected "Fast video and clear audio arrive together." \
 --output "${OUTPUT_ROOT}/asr-gate.json" \
 --run-id "base34-from42-200-${SLURM_JOB_ID}" || true
date -Is > "${OUTPUT_ROOT}/awaiting_quality_review.txt"
'
