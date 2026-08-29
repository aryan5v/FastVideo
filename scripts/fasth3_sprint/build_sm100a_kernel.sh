#!/usr/bin/env bash
# Build and verify the FastVideo SM100a VSA extension into a sprint-local overlay.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
KERNEL_ROOT="${REPO_ROOT}/fastvideo-kernel"
BUILD_VENV="${SPRINT_ROOT}/caches/kernel-build-venv"
WHEEL_DIR="${SPRINT_ROOT}/artifacts/kernel-wheels"
OVERLAY_DIR="${SPRINT_ROOT}/python-packages"
RECEIPT_DIR="${SPRINT_ROOT}/manifests"
LOG_DIR="${SPRINT_ROOT}/logs/kernel-build"
mkdir -p "${WHEEL_DIR}" "${OVERLAY_DIR}" "${RECEIPT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1
echo "$(date -Is) sm100a build start job=${SLURM_JOB_ID:-none} commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

if [[ ! -x "${BUILD_VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${BUILD_VENV}"
fi
source "${BUILD_VENV}/bin/activate"
python -m pip install --upgrade pip uv scikit-build-core cmake ninja

# CMake does not discover TorchConfig.cmake through Python's
# --system-site-packages path.  Resolve the prefix from the exact torch import
# used for the build and pass it explicitly instead of relying on a host-wide
# CMake search path.
TORCH_CMAKE_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
if [[ ! -f "${TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake" ]]; then
  echo "TorchConfig.cmake not found under ${TORCH_CMAKE_PREFIX}" >&2
  exit 1
fi
export CMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
echo "Using Torch CMake prefix: ${TORCH_CMAKE_PREFIX}"

git -C "${REPO_ROOT}" submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

export TORCH_CUDA_ARCH_LIST="10.0a"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=100a -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

cd "${KERNEL_ROOT}"
uv build --wheel -v --no-build-isolation --out-dir "${WHEEL_DIR}" .
WHEEL_PATH="$(find "${WHEEL_DIR}" -maxdepth 1 -type f \( -name 'fastvideo_kernel-*.whl' -o -name 'fastvideo-kernel-*.whl' \) | sort | tail -n 1)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "No fastvideo-kernel wheel was produced" >&2
  exit 1
fi
python -m pip install --target "${OVERLAY_DIR}" --upgrade --force-reinstall --no-deps "${WHEEL_PATH}"

PYTHONPATH="${REPO_ROOT}:${OVERLAY_DIR}" /mnt/nfs/vlm-aryan/fastvideo-wan-venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from fastvideo_kernel import block_sparse_attn_sm100a

if not block_sparse_attn_sm100a._HAS_VSA_SM100A:
    raise RuntimeError("Built fastvideo-kernel does not export the SM100a VSA forward")
props = torch.cuda.get_device_properties(0)
if (props.major, props.minor) != (10, 0):
    raise RuntimeError(f"SM100a receipt requires compute capability 10.0, got {props.major}.{props.minor}")
wheel_dir = Path(os.environ["WHEEL_DIR"])
wheel = sorted(wheel_dir.glob("fastvideo*kernel*.whl"))[-1]
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
receipt = {
    "source_commit": os.environ["SOURCE_COMMIT"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": props.name,
    "compute_capability": f"{props.major}.{props.minor}",
    "wheel": str(wheel),
    "wheel_sha256": digest,
    "module": str(Path(block_sparse_attn_sm100a.__file__).resolve()),
    "has_vsa_sm100a": bool(block_sparse_attn_sm100a._HAS_VSA_SM100A),
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "cmake_args": os.environ["CMAKE_ARGS"],
}
path = Path(os.environ["RECEIPT_PATH"])
path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "$(date -Is) sm100a build verified receipt=${RECEIPT_DIR}/sm100a_kernel_receipt.json"
