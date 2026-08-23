#!/bin/bash
#SBATCH --job-name=h3-dmd2-v10-64g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=120:00:00
#SBATCH --partition=hpc-rack-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/slurm-%x-%j.out

# One-allocation v10 launch: capacity gate, immutable-input preflight, then
# production. Submit this file directly with sbatch; do not wrap it in another
# shell payload. Requeue stays disabled until every pre-production gate passes.

set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: sbatch ... run_h3_v10_gated.sh <40-character-execution-commit>" >&2
  exit 2
fi

readonly REPO="/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10"
readonly VENV="${REPO}/.venv"
readonly PROD_CONFIG="${REPO}/examples/train/configs/distribution_matching/minimax_h3/"\
"dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml"
readonly MAXSHAPE_CONFIG="${REPO}/examples/train/configs/distribution_matching/minimax_h3/"\
"dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml"
readonly MAXSHAPE_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/v10_maxshape_64g"
readonly MAXSHAPE_RECEIPT="${MAXSHAPE_ROOT}/audit/job-${SLURM_JOB_ID:?run inside Slurm}/RESULT.json"
readonly LUSTRE_HOME="/mnt/lustre/vlm-wlsaidhi"
readonly KERNEL_PREFIX="/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix"
readonly FA4_OVERLAY="/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay"
readonly FA4_CUTLASS_PACKAGES="${FA4_OVERLAY}/nvidia_cutlass_dsl/python_packages"
readonly EXPECTED_V10_COMMIT="$1"

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export SLURM_EXPORT_ENV=ALL
export REPO VENV LUSTRE_HOME
export SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=64
export H3_V10_KERNEL_PREFIX="${KERNEL_PREFIX}"
export H3_V10_FA4_OVERLAY="${FA4_OVERLAY}"
export H3_V10_CUTLASS_PACKAGES="${FA4_CUTLASS_PACKAGES}"
export PYTHONPATH="${KERNEL_PREFIX}:${FA4_OVERLAY}:${FA4_CUTLASS_PACKAGES}"
export FASTVIDEO_VSA_SM100A=1
export H3_V10_COMPILE_LOGS=1

if [[ "${SLURM_JOB_NUM_NODES:-0}" != "16" ]]; then
  echo "V10 GATED LAUNCH FAILED: requires exactly 16 trays; got ${SLURM_JOB_NUM_NODES:-unset}" >&2
  exit 1
fi
gpus_per_node="${SLURM_GPUS_PER_NODE:-4}"
gpus_per_node="${gpus_per_node##*:}"
if [[ "${gpus_per_node}" != "4" ]]; then
  echo "V10 GATED LAUNCH FAILED: requires exactly four GPUs per tray; got ${SLURM_GPUS_PER_NODE:-unset}" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "V10 GATED LAUNCH FAILED: missing compute Python ${VENV}/bin/python" >&2
  exit 1
fi

# A requeued Slurm job retains its Requeue flag. Reset it on every entry so a
# stale/failed receipt or preflight can never form an automatic failure loop.
scontrol update JobId="${SLURM_JOB_ID}" Requeue=0

export EXPECTED_V10_COMMIT

require_exact_execution_checkout() {
  local execution_commit
  execution_commit="$(git -C "${REPO}" rev-parse HEAD)"
  if [[ "${EXPECTED_V10_COMMIT}" != "${execution_commit}" ]]; then
    echo "V10 GATED LAUNCH FAILED: requested commit ${EXPECTED_V10_COMMIT} != execution HEAD ${execution_commit}" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${REPO}" status --porcelain)" ]]; then
    echo "V10 GATED LAUNCH FAILED: execution checkout is dirty: ${REPO}" >&2
    git -C "${REPO}" status --short >&2
    exit 1
  fi
}

require_exact_execution_checkout

if [[ ! -f "${MAXSHAPE_RECEIPT}" ]]; then
  echo "=== v10 final-commit max-shape gate (job ${SLURM_JOB_ID}) ==="
  export CONFIG="${MAXSHAPE_CONFIG}"
  export H3_V10_MAXSHAPE_ROOT="${MAXSHAPE_ROOT}"
  export H3_V10_KERNEL_GATE=1
  export GATE_TEST=1
  bash "${REPO}/scripts/train/run_h3_v10_maxshape_gate.sh"
else
  echo "=== reusing same-job max-shape receipt ${MAXSHAPE_RECEIPT} ==="
fi
require_exact_execution_checkout

# Do not let a failed same-job receipt hide behind a successful receipt from a
# different allocation. The general preflight below performs the same check
# over the audit lineage; this check pins the gate to this Slurm job as well.
"${VENV}/bin/python" - \
  "${MAXSHAPE_RECEIPT}" "${MAXSHAPE_CONFIG}" "${EXPECTED_V10_COMMIT}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
execution_commit = sys.argv[3]
job_id = sys.argv[4]
try:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"V10 GATED LAUNCH FAILED: invalid same-job max-shape receipt: {error}") from error
checks = receipt.get("checks", {})
expected = {
    "schema_version": "fastvideo-h3-v10-maxshape-gate-v1",
    "success": True,
    "job_id": job_id,
    "execution_commit": execution_commit,
    "config": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "shape": {
        "width": 1760,
        "height": 768,
        "num_frames": 362,
        "video_latent_shape": [24, 107, 48, 110],
    },
    "steps": {"critic": 1, "student": 2},
}
observed = {key: receipt.get(key) for key in expected}
if observed != expected or not checks or not all(value is True for value in checks.values()):
    raise SystemExit(
        "V10 GATED LAUNCH FAILED: same-job max-shape receipt does not satisfy the exact final-commit contract; "
        f"receipt={receipt_path}"
    )
print(f"READY: exact same-job max-shape receipt {receipt_path}")
PY

echo "=== v10 immutable-input and resume preflight ==="
export CONFIG="${PROD_CONFIG}"
bash "${REPO}/examples/train/slurm/prepare_h3_dmd2_v10_slinky.sh"
require_exact_execution_checkout

# The max-shape run already exercised the kernel and launch-test gates with the
# exact environment in this allocation. Avoid repeating them in production.
export H3_V10_KERNEL_GATE=0
export GATE_TEST=0

# Only production is requeueable. On re-entry this script resets Requeue=0,
# validates the same-job receipt and all preflight contracts again, then turns
# it back on immediately before handing control to the production launcher.
scontrol update JobId="${SLURM_JOB_ID}" Requeue=1
echo "READY: gates passed; enabling Slurm requeue and starting v10 production"
exec bash "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
