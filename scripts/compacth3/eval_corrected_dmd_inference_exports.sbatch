#!/bin/bash
#SBATCH --job-name=h3-dmd2-export-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/sbatch-h3-dmd2-export-eval-%j.log
#SBATCH --error=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829/sbatch-h3-dmd2-export-eval-%j.log

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
DMD_CODE=${SPRINT_ROOT}/code/release20b-dmd2-v17-fp32resume-v1
RUNNER_ROOT=${SPRINT_ROOT}/code/release-execution-dmd-ladderfix-v1
DMD_RUN=${SPRINT_ROOT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3/job-paired8972-8975-4000-v3
STEPS="800 1000 1200"
export SPRINT_ROOT DMD_CODE RUNNER_ROOT DMD_RUN STEPS

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${RUNNER_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
PROMPTS="${SPRINT_ROOT}/job-scripts/bench_five_new_prompts.json"
export HF_HOME=/mnt/nfs/vlm-aryan/hf-cache PYTHONDONTWRITEBYTECODE=1
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
export PYTHONPATH="${RUNNER_ROOT}:${SPRINT_ROOT}/python-packages:/mnt/nfs/vlm-aryan/fastvideo-wan-venv/lib/python3.12/site-packages"

for step in ${STEPS}; do
  model="${DMD_RUN}/inference/checkpoint-${step}"
  eval_dir="${DMD_RUN}/eval-exported-checkpoint-${step}-bench-five-exact4-v1"
  media="${eval_dir}/media"

  [[ -f "${model}/.complete" && -s "${model}/metadata.json" ]] || {
    echo "FATAL: incomplete inference export for step ${step}" >&2
    exit 2
  }
  [[ $(find "${model}/transformer" -maxdepth 1 -name "*.safetensors" ! -name "*.index.json" | wc -l) -eq 8 ]] || {
    echo "FATAL: step ${step} does not have eight transformer shards" >&2
    exit 3
  }

  "${PY}" - "${model}" <<"PY"
import collections
import json
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
counts = collections.Counter()
for shard in sorted((root / "transformer").glob("*.safetensors")):
    with shard.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    for name, value in header.items():
        if name != "__metadata__":
            counts[value["dtype"]] += 1
metadata = json.loads((root / "metadata.json").read_text())
print("step={} declared_dtype={} tensor_dtypes={}".format(
    metadata["step"], metadata["dtype"], dict(counts)))
if set(counts) != {"BF16"}:
    raise SystemExit(f"unexpected inference tensor dtypes: {dict(counts)}")
PY

  if [[ -f "${eval_dir}/completed_at.txt" ]] && [[ $(find "${media}" -maxdepth 1 -name "*.mp4" | wc -l) -eq 5 ]]; then
    echo "step ${step}: verified media already complete; skipping"
    continue
  fi
  if [[ -s "${media}/run_manifest.json" ]] && [[ $(find "${media}" -maxdepth 1 -name "*.mp4" | wc -l) -eq 5 ]]; then
    "${PY}" - "${media}" <<"PY"
import json
import sys
from pathlib import Path

media = Path(sys.argv[1])
manifest = json.loads((media / "run_manifest.json").read_text())
assert manifest["schedule"]["grid_points"] == 5, manifest["schedule"]
assert manifest["schedule"]["transformer_calls"] == 4, manifest["schedule"]
assert len(manifest["schedule"]["video"]["transformer_timesteps"]) == 4
assert len(manifest["schedule"]["audio"]["transformer_timesteps"]) == 4
PY
    date -Is > "${eval_dir}/completed_at.txt"
    echo "step ${step}: recovered and accepted five already-rendered videos"
    continue
  fi
  [[ ! -e "${eval_dir}" ]] || {
    echo "FATAL: refusing to overwrite partial evaluation ${eval_dir}" >&2
    exit 4
  }
  mkdir -p "${eval_dir}"

  cd "${RUNNER_ROOT}"
  "${PY}" "${RUNNER_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
    --model-path "${model}" --checkpoint-role "corrected-dmd2-step${step}-native-export-exact4" \
    --attention dense --attention-backend TORCH_SDPA --prompts "${PROMPTS}" \
    --output-dir "${media}" --run-id "corrected-dmd2-step${step}-native-export-exact4-${SLURM_JOB_ID}" \
    --source-commit "$(cat "${DMD_CODE}/CODE_COMMIT" 2>/dev/null || echo unknown)" --max-prompts 5 \
    --height 480 --width 832 --num-frames 124 --seed 20260912 \
    --steps 5 --num-gpus 4 --dit-precision bf16 --profile strict \
    --no-fa4 --no-compile --no-upload-videos

  "${PY}" - "${media}" <<"PY"
import json
import sys
from pathlib import Path

media = Path(sys.argv[1])
manifest = json.loads((media / "run_manifest.json").read_text())
assert len(list(media.glob("*.mp4"))) == 5
assert manifest["schedule"]["grid_points"] == 5, manifest["schedule"]
assert manifest["schedule"]["transformer_calls"] == 4, manifest["schedule"]
assert len(manifest["schedule"]["video"]["transformer_timesteps"]) == 4
assert len(manifest["schedule"]["audio"]["transformer_timesteps"]) == 4
print("verified exact four-call media", media)
PY
  date -Is > "${eval_dir}/completed_at.txt"
done
'
