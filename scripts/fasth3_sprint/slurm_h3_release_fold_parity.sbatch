#!/bin/bash
#SBATCH --job-name=h3-fold-parity
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export LANE="${1:?usage: sbatch $0 A|B|C}"
export CODE_ROOT="${PWD:?}"
export SPRINT_ROOT=/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829
case "${LANE}" in
  A)
    export SOURCE="${SPRINT_ROOT}/runs/base42-prompt58k-recovery/job-6929/export-200"
    export FOLDED="${SPRINT_ROOT}/release-candidates/20b-backbone-r768-job6929-v1"
    export RANK=768 DEPTH=42 INFERENCE_STEPS=50 ROLE=20b-backbone-r768
    export TEMPLATE="${CODE_ROOT}/examples/train/configs/fasth3_release_long_recovery.yaml"
    ;;
  B)
    export SOURCE="${SPRINT_ROOT}/runs/base34-recut-detail-band/job-7048/export-200"
    export FOLDED="${SPRINT_ROOT}/release-candidates/14b-backbone-r16-job7048-v1"
    export RANK=16 DEPTH=34 INFERENCE_STEPS=50 ROLE=14b-backbone-r16
    export TEMPLATE="${CODE_ROOT}/examples/train/configs/fasth3_release_long_recovery.yaml"
    ;;
  C)
    export SOURCE="${SPRINT_ROOT}/runs/base34-recut-dmd4/job-7102/export-500"
    export FOLDED="${SPRINT_ROOT}/release-candidates/14B-4C-FAST-BRANCH-v0"
    export RANK=16 DEPTH=34 INFERENCE_STEPS=5 ROLE=14b-4c-fast-r16
    export TEMPLATE="${CODE_ROOT}/examples/train/configs/fasth3_14b_four_call_recovery.yaml"
    ;;
  *) echo "LANE must be A, B, or C" >&2; exit 2 ;;
esac
export AUDIT_ROOT="${SPRINT_ROOT}/eval/release-fold-parity/lane-${LANE}-job-${SLURM_JOB_ID}"
test -s "${CODE_ROOT}/CODE_COMMIT"
test -s "${SOURCE}/transformer/config.json"
test ! -e "${FOLDED}"
test ! -e "${AUDIT_ROOT}"
mkdir -p "${AUDIT_ROOT}"

srun --export=ALL --kill-on-bad-exit=1 \
  --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
  --container-mounts=/mnt/nfs/vlm-aryan:/mnt/nfs/vlm-aryan,/mnt/lustre/vlm-shared:/mnt/lustre/vlm-shared:ro \
  --container-workdir="${CODE_ROOT}" bash -lc '
set -euo pipefail
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env
export PYTHONPATH="${CODE_ROOT}:${SPRINT_ROOT}/python-packages" HF_HOME=/mnt/nfs/vlm-aryan/hf-cache
export PYTHONDONTWRITEBYTECODE=1 FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA FASTVIDEO_MINIMAX_H3_FUSIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ENABLE_MONITORING=0
PY=/mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python
"${PY}" "${CODE_ROOT}/scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py" \
  --src "${SOURCE}/transformer" --dst "${FOLDED}/transformer" --rank "${RANK}" \
  --receipt "${FOLDED}/factorization_receipt.json"
for component in "${SOURCE}"/*; do
  name="$(basename "${component}")"
  [[ "${name}" == transformer ]] && continue
  ln -sfn "${component}" "${FOLDED}/${name}"
done
"${PY}" - "${FOLDED}/factorization_receipt.json" "${LANE}" <<"PY"
import json, sys
r = json.load(open(sys.argv[1]))
assert r["basis_relative_residual"] < 1e-6, r
assert r["modulation_relative_max_error"] < 1e-5, r
billion = r["parameters_after"] / 1e9
if sys.argv[2] == "A":
    assert 19.0 <= billion <= 21.5, r
else:
    assert 13.5 <= billion <= 14.5, r
PY
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/build_h3_recovery_config.py" \
  --template "${TEMPLATE}" --student "${FOLDED}" --depth "${DEPTH}" \
  --output "${FOLDED}/training-config.yaml"
for WHICH in source folded; do
  if [[ "${WHICH}" == source ]]; then MODEL="${SOURCE}"; else MODEL="${FOLDED}"; fi
  "${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/run_baseline_matrix.py" \
    --model-path "${MODEL}" --checkpoint-role "${ROLE}-${WHICH}" \
    --attention dense --attention-backend TORCH_SDPA \
    --prompts "${CODE_ROOT}/examples/training/fasth3_14b_2step_qad/rescue_gate_prompts.json" \
    --output-dir "${AUDIT_ROOT}/${WHICH}" --run-id "fold-parity-${LANE}-${WHICH}-${SLURM_JOB_ID}" \
    --source-commit "$(cat "${CODE_ROOT}/CODE_COMMIT")" --max-prompts 1 \
    --height 480 --width 832 --num-frames 124 --seed 2026091001 \
    --steps "${INFERENCE_STEPS}" --num-gpus 4 --dit-precision bf16 \
    --profile strict --no-fa4 --no-compile --no-upload-videos
done
"${PY}" "${CODE_ROOT}/scripts/fasth3_sprint/compare_h3_media_parity.py" \
  --reference "${AUDIT_ROOT}/source" --candidate "${AUDIT_ROOT}/folded" \
  --receipt "${FOLDED}/production_media_parity.json"
"${PY}" - "${FOLDED}" <<"PY"
import json, os, pathlib
root = pathlib.Path(os.environ["FOLDED"])
receipt = {
    "schema_version": 1,
    "status": "frozen_step0_parent",
    "lane": os.environ["LANE"],
    "source": os.environ["SOURCE"],
    "rank": int(os.environ["RANK"]),
    "depth": int(os.environ["DEPTH"]),
    "deployment_transformer_calls": int(os.environ["INFERENCE_STEPS"]) - 1,
    "parity_seed": 2026091001,
    "source_commit": pathlib.Path(os.environ["CODE_ROOT"], "CODE_COMMIT").read_text().strip(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
(root / "FROZEN_MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
'
