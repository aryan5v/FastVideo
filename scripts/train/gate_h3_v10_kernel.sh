#!/bin/bash
# Compute-node launch gate for the v10 H3 sparse/dense attention environment.
# This must run on a GB200 before training: it verifies import provenance,
# corrected Triton-64 forward/backward gradients, sm_100a numerical parity,
# and the real no-grad H3 route (with the Triton fallback made fatal).

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
H3_V10_KERNEL_PREFIX="${H3_V10_KERNEL_PREFIX:-/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix}"
H3_V10_FA4_OVERLAY="${H3_V10_FA4_OVERLAY:-/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay}"
H3_V10_CUTLASS_PACKAGES="${H3_V10_CUTLASS_PACKAGES:-${H3_V10_FA4_OVERLAY}/nvidia_cutlass_dsl/python_packages}"
expected_pythonpath="${H3_V10_KERNEL_PREFIX}:${H3_V10_FA4_OVERLAY}:${H3_V10_CUTLASS_PACKAGES}"

if [[ "${PYTHONPATH:-}" != "${expected_pythonpath}" ]]; then
  echo "KERNEL GATE FAILED: PYTHONPATH must be exactly ${expected_pythonpath}" >&2
  echo "Observed: ${PYTHONPATH:-<unset>}" >&2
  exit 1
fi
if [[ "${FASTVIDEO_VSA_SM100A:-0}" != "1" || "${FASTVIDEO_FA4:-0}" != "1" ]]; then
  echo "KERNEL GATE FAILED: FASTVIDEO_VSA_SM100A=1 and FASTVIDEO_FA4=1 are required" >&2
  exit 1
fi
if [[ ! -f "${H3_V10_KERNEL_PREFIX}/FASTVIDEO_KERNEL_V10_RECEIPT.json" ]]; then
  echo "KERNEL GATE FAILED: missing immutable kernel receipt" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "KERNEL GATE FAILED: missing v10 Python ${VENV}/bin/python" >&2
  exit 1
fi

cd "${REPO}"
source "${VENV}/bin/activate"
export REPO H3_V10_KERNEL_PREFIX H3_V10_FA4_OVERLAY H3_V10_CUTLASS_PACKAGES

"${VENV}/bin/python" - <<'PY'
import importlib
import json
import os
from pathlib import Path
import subprocess

import torch

repo = Path(os.environ["REPO"]).resolve()
prefix = Path(os.environ["H3_V10_KERNEL_PREFIX"]).resolve()
fa4_overlay = Path(os.environ["H3_V10_FA4_OVERLAY"]).resolve()
cutlass_packages = Path(os.environ["H3_V10_CUTLASS_PACKAGES"]).resolve()
expected_sys_path = [str(prefix), str(fa4_overlay), str(cutlass_packages)]
observed_pythonpath = [str(Path(value).resolve()) for value in os.environ["PYTHONPATH"].split(":")]
if observed_pythonpath != expected_sys_path:
    raise SystemExit(f"PYTHONPATH entries do not match production order: {observed_pythonpath}")


def module_locations(module):
    locations = []
    if getattr(module, "__file__", None):
        locations.append(Path(module.__file__).resolve())
    locations.extend(Path(value).resolve() for value in getattr(module, "__path__", []))
    return locations


def require_under(module, root, label):
    locations = module_locations(module)
    if not locations or not all(location.is_relative_to(root) for location in locations):
        raise SystemExit(f"{label} resolved outside {root}: {locations}")
    print(f"RECEIPT {label}: {locations}")


kernel = importlib.import_module("fastvideo_kernel")
sm100a = importlib.import_module("fastvideo_kernel.block_sparse_attn_sm100a")
triton64 = importlib.import_module("fastvideo_kernel.triton_kernels.block_sparse_attn_triton")
fa4_cute = importlib.import_module("flash_attn.cute")
cutlass = importlib.import_module("cutlass")
require_under(kernel, prefix, "fastvideo_kernel")
require_under(sm100a, prefix, "sm100a")
require_under(triton64, prefix, "triton64")
require_under(fa4_cute, fa4_overlay, "flash_attn.cute")
require_under(cutlass, cutlass_packages, "cutlass")

receipt = json.loads((prefix / "FASTVIDEO_KERNEL_V10_RECEIPT.json").read_text(encoding="utf-8"))
if receipt.get("schema_version") != "fastvideo-h3-v10-kernel-v1":
    raise SystemExit(f"unexpected kernel receipt: {receipt}")
source_commit = str(receipt["source_commit"])
subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", source_commit, "HEAD"], check=True)
current_kernel_tree = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD:fastvideo-kernel"], text=True).strip()
if current_kernel_tree != receipt["kernel_tree"]:
    raise SystemExit(
        f"installed kernel tree {receipt['kernel_tree']} != execution tree {current_kernel_tree}")
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0):
    raise SystemExit("kernel gate requires a GB200 with compute capability (10, 0)")
if not sm100a._HAS_VSA_SM100A:
    raise SystemExit("the imported extension has no sm_100a block-sparse symbols")
print(f"RECEIPT source_commit={source_commit} kernel_tree={current_kernel_tree} gpu={torch.cuda.get_device_name(0)}")
PY

"${VENV}/bin/python" -m pytest -q -s \
  fastvideo-kernel/tests/test_vsa_triton_backward_scale.py
"${VENV}/bin/python" -m pytest -q -s \
  'tests/test_block_sparse_sm100a.py::test_forward_matches_reference[64]'
timeout --signal=TERM --kill-after=30s 300s \
  "${VENV}/bin/python" -m pytest -q -s \
  fastvideo/tests/attention/test_vsa_h3_sm100a_route.py::test_real_sm100a_no_grad_route_receipt

echo "H3_V10_KERNEL_GATE=PASSED"
