#!/bin/bash
# Build a single-arch GB200 fastvideo-kernel from the execution checkout and
# atomically publish it as the first entry in v10's production PYTHONPATH. The
# receipt binds the exact source commit, retained wheel, and stable installed
# prefix contents. Existing prefixes are retained as timestamped backups; this
# script never mutates the execution venv or the shared FA4 overlay.

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
KERNEL_ROOT="${KERNEL_ROOT:-/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel}"
KERNEL_PREFIX="${KERNEL_PREFIX:-${KERNEL_ROOT}/prefix}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/sm100a_fix}"
UV="${UV:-/home/vlm-wlsaidhi/.local/bin/uv}"

CUDA_VIEW="${TOOLCHAIN_ROOT}/cuda_view"
HOST_TOOLCHAIN="${TOOLCHAIN_ROOT}/hosttc"
BUILD_TOOLS="${TOOLCHAIN_ROOT}/buildtools"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file $1" >&2
    exit 1
  fi
}

require_file "${VENV}/bin/python"
require_file "${UV}"
require_file "${CUDA_VIEW}/bin/nvcc"
require_file "${HOST_TOOLCHAIN}/bin/aarch64-conda-linux-gnu-g++"
require_file "${BUILD_TOOLS}/scikit_build_core/__init__.py"
require_file "${REPO}/fastvideo-kernel/pyproject.toml"
require_file "${REPO}/scripts/train/h3_v10_kernel_receipt.py"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: the pinned host compiler targets aarch64; host is $(uname -m)" >&2
  exit 1
fi
if ! git -C "${REPO}" merge-base --is-ancestor 907f2100e HEAD; then
  echo "ERROR: source does not contain merged sm_100a forward PR #1719" >&2
  exit 1
fi
if ! git -C "${REPO}" merge-base --is-ancestor 56d4a6074 HEAD; then
  echo "ERROR: source does not contain corrected Triton backward PR #1730" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO}" status --porcelain -- fastvideo-kernel)" ]]; then
  echo "ERROR: fastvideo-kernel has uncommitted changes; build provenance would be ambiguous" >&2
  exit 1
fi

# The wheel needs CUTLASS's actual files; a git archive alone would contain an
# empty submodule directory. Initialize only the two kernel header submodules,
# then copy the complete committed kernel tree to an isolated build directory.
git -C "${REPO}" submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

mkdir -p "${KERNEL_ROOT}/wheels" "${KERNEL_ROOT}/logs"
source_commit="$(git -C "${REPO}" rev-parse HEAD)"
kernel_tree="$(git -C "${REPO}" rev-parse HEAD:fastvideo-kernel)"
build_dir="$(mktemp -d "${KERNEL_ROOT}/build.XXXXXX")"
wheel_dir="$(mktemp -d "${KERNEL_ROOT}/wheels/${source_commit}.XXXXXX")"
staged_prefix="$(mktemp -d "${KERNEL_ROOT}/prefix.${source_commit}.XXXXXX")"
cp -a "${REPO}/fastvideo-kernel/." "${build_dir}/"

build_log="${KERNEL_ROOT}/logs/rebuild_${source_commit}.log"
(
  cd "${build_dir}"
  env -u CONDA_PREFIX \
    TORCH_CUDA_ARCH_LIST=10.0a \
    GPU_BACKEND=CUDA \
    CC="${HOST_TOOLCHAIN}/bin/aarch64-conda-linux-gnu-gcc" \
    CXX="${HOST_TOOLCHAIN}/bin/aarch64-conda-linux-gnu-g++" \
    CUDACXX="${CUDA_VIEW}/bin/nvcc" \
    CUDAHOSTCXX="${HOST_TOOLCHAIN}/bin/aarch64-conda-linux-gnu-g++" \
    CUDAToolkit_ROOT="${CUDA_VIEW}" \
    CMAKE_ARGS="-DCUDAToolkit_ROOT=${CUDA_VIEW} -DCMAKE_CUDA_COMPILER=${CUDA_VIEW}/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=${HOST_TOOLCHAIN}/bin/aarch64-conda-linux-gnu-g++ -DCMAKE_CUDA_ARCHITECTURES=100a -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA" \
    PYTHONPATH="${BUILD_TOOLS}" \
    PATH="${CUDA_VIEW}/bin:${BUILD_TOOLS}/bin:${HOST_TOOLCHAIN}/bin:/usr/bin:/bin" \
    CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-64}" \
    "${VENV}/bin/python" - "${wheel_dir}" <<'PY'
import sys

from scikit_build_core.build import build_wheel

print("WHEEL_BUILT:", build_wheel(sys.argv[1]))
PY
) 2>&1 | tee "${build_log}"

wheel_path="$(find "${wheel_dir}" -maxdepth 1 -type f \( -name 'fastvideo_kernel-*.whl' -o -name 'fastvideo-kernel-*.whl' \) -print -quit)"
if [[ -z "${wheel_path}" ]]; then
  echo "ERROR: build produced no fastvideo-kernel wheel" >&2
  exit 1
fi
"${UV}" pip install --python "${VENV}/bin/python" --target "${staged_prefix}" --no-deps "${wheel_path}"

wheel_sha256="$(sha256sum "${wheel_path}" | cut -d' ' -f1)"
installed_prefix_tree_sha256="$(
  "${VENV}/bin/python" "${REPO}/scripts/train/h3_v10_kernel_receipt.py" "${staged_prefix}"
)"
"${VENV}/bin/python" - \
  "${staged_prefix}/FASTVIDEO_KERNEL_V10_RECEIPT.json" \
  "${source_commit}" "${kernel_tree}" "${wheel_path}" "${wheel_sha256}" \
  "${installed_prefix_tree_sha256}" <<'PY'
import datetime as dt
import json
import pathlib
import sys

receipt_path, source_commit, kernel_tree, wheel_path, wheel_sha256, installed_prefix_tree_sha256 = sys.argv[1:]
receipt = {
    "schema_version": "fastvideo-h3-v10-kernel-v1",
    "source_commit": source_commit,
    "kernel_tree": kernel_tree,
    "wheel": wheel_path,
    "wheel_sha256": wheel_sha256,
    "installed_prefix_tree_sha256": installed_prefix_tree_sha256,
    "cuda_arch": "10.0a",
    "contains_prs": [1719, 1730],
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
pathlib.Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

require_file "${staged_prefix}/FASTVIDEO_KERNEL_V10_RECEIPT.json"
if [[ ! -d "${staged_prefix}/fastvideo_kernel" ]]; then
  echo "ERROR: staged wheel has no fastvideo_kernel package" >&2
  exit 1
fi

if [[ -e "${KERNEL_PREFIX}" ]]; then
  backup="${KERNEL_PREFIX}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${KERNEL_PREFIX}" "${backup}"
  echo "Previous prefix retained at ${backup}"
fi
mv "${staged_prefix}" "${KERNEL_PREFIX}"

echo "Published ${KERNEL_PREFIX} from ${source_commit} (kernel tree ${kernel_tree})"
echo "Build log: ${build_log}"
echo "Production PYTHONPATH order:"
echo "${KERNEL_PREFIX}:/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay:/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay/nvidia_cutlass_dsl/python_packages"
echo "Hardware validation remains mandatory: scripts/train/gate_h3_v10_kernel.sh"
