#!/bin/bash
#SBATCH --job-name=h3-asr-six
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
# Whisper WER on the exact-speech clip for the current candidate set.
# CPU-only; runs inside the container so the pinned venv is visible.
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
srun --export=ALL --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
 --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan \
 --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
R=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
for n in base 7048 r16 7101 6972 6988; do
  /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python "${CODE_ROOT}/scripts/fasth3_sprint/verify_speech_asr.py" \
    --wav "${R}/asr-wavs/${n}.wav" \
    --expected "Fast video and clear audio arrive together." \
    --output "${R}/asr-wavs/${n}.json" --run-id "local-${n}" > "${R}/asr-wavs/${n}.log" 2>&1 || true
done
'
