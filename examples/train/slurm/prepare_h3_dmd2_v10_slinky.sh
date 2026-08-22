#!/bin/bash
# Validate the shared v10 inputs and print the exact eight-tray Slinky submit
# command. This helper never calls sbatch; copy the final line deliberately
# after every gate reports READY.

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
CONFIG="${CONFIG:-${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp32_v10_dataonly_mixed_vsa64.yaml}"
DATA_ROOT="${DATA_ROOT:-/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v1}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-${DATA_ROOT}/validation/heldout64.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/lustre/vlm-wlsaidhi/fastvideo/outputs/minimax_h3_dmd2_sp1_v10_dataonly_mixed_vsa64}"
LOG_DIR="${LOG_DIR:-/mnt/lustre/vlm-wlsaidhi/fastvideo/logs}"
# rack-3 is deliberate: Slinky provisions pods from the submitted eight-node
# request, and the retired v8 allocation was on this rack. Do not submit
# speculative warm-up jobs; override PARTITION if the operator selects rack-2.
PARTITION="${PARTITION:-hpc-rack-3}"
LUSTRE_HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
KERNEL_PREFIX="${KERNEL_PREFIX:-/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix}"
FA4_OVERLAY="${FA4_OVERLAY:-/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay}"
FA4_CUTLASS_PACKAGES="${FA4_CUTLASS_PACKAGES:-${FA4_OVERLAY}/nvidia_cutlass_dsl/python_packages}"
V10_PYTHONPATH="${KERNEL_PREFIX}:${FA4_OVERLAY}:${FA4_CUTLASS_PACKAGES}"

sources=(
  h3_t2av_video_nuva_50k_720_mixed_len
  h3_t2av_video_5s_768p
  h3_t2av_video_nuva_10k_720_mixed_len
  h3_t2av_video_nuva_10k_mixed_res_len
  h3_t2av_fastgen_vidprom_150k
)

failures=0
require_file() {
  if [[ ! -f "$1" ]]; then
    echo "NOT READY: missing file $1" >&2
    failures=$((failures + 1))
  fi
}

require_file "${CONFIG}"
require_file "${VENV}/bin/python"
require_file "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
require_file "${REPO}/scripts/train/gate_h3_v10_kernel.sh"
require_file "${REPO}/scripts/preprocess/minimax_h3_native_t2va/finalize_dataset.py"
require_file "${KERNEL_PREFIX}/FASTVIDEO_KERNEL_V10_RECEIPT.json"
require_file "${DATA_ROOT}/FROZEN_MANIFEST.json"
require_file "${DATA_ROOT}/READY.json"
if [[ ! -d "${FA4_OVERLAY}/flash_attn/cute" ]]; then
  echo "NOT READY: missing FA4 CuTe package under ${FA4_OVERLAY}" >&2
  failures=$((failures + 1))
fi
if [[ ! -d "${FA4_CUTLASS_PACKAGES}/cutlass" ]]; then
  echo "NOT READY: missing pinned CUTLASS DSL package under ${FA4_CUTLASS_PACKAGES}" >&2
  failures=$((failures + 1))
fi

for source in "${sources[@]}"; do
  source_root="${DATA_ROOT}/${source}"
  require_file "${source_root}/READY.json"
  require_file "${source_root}/MANIFEST.json"
  require_file "${source_root}/MANIFEST_rows.jsonl"
  require_file "${source_root}/data/map_style_cache/file_info.pkl"
  if [[ ! -d "${source_root}/data" ]]; then
    echo "NOT READY: missing production parquet root ${source_root}/data" >&2
    failures=$((failures + 1))
  elif [[ -z "$(find "${source_root}/data" -type f -name '*.parquet' -print -quit)" ]]; then
    echo "NOT READY: no parquet under ${source_root}/data" >&2
    failures=$((failures + 1))
  fi
done

# The finalizer is the authoritative immutable-data verifier. It re-audits
# every done marker/parquet row, bucket geometry, schema/hash, the exact
# map-style cache tuple, source MANIFEST/READY receipts, and aggregate count.
if (( failures == 0 )); then
  if ! "${VENV}/bin/python" \
    "${REPO}/scripts/preprocess/minimax_h3_native_t2va/finalize_dataset.py" \
    --root "${DATA_ROOT}" --verify-only; then
    echo "NOT READY: finalized dataset verification failed" >&2
    failures=$((failures + 1))
  fi
fi

require_file "${VALIDATION_MANIFEST}"
if [[ -x "${VENV}/bin/python" && -f "${VALIDATION_MANIFEST}" ]]; then
  if ! "${VENV}/bin/python" - "${VALIDATION_MANIFEST}" "${sources[@]}" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
required_sources = set(sys.argv[2:])
document = json.loads(manifest.read_text(encoding="utf-8"))
rows = document.get("data") if isinstance(document, dict) else document
if not isinstance(rows, list) or len(rows) != 64:
    raise SystemExit(f"validation manifest must contain exactly 64 data rows; got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}")

seen_sources = set()
seen_refs = set()
for index, row in enumerate(rows):
    required = ("caption", "ref_video", "source", "sample_id", "width", "height", "num_frames")
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"validation row {index} is missing {missing}")
    width, height, frames = (int(row["width"]), int(row["height"]), int(row["num_frames"]))
    if min(width, height, frames) <= 0 or width % 8 or height % 8:
        raise SystemExit(f"validation row {index} has invalid shape {width}x{height}x{frames}f")
    ref = pathlib.Path(str(row["ref_video"]))
    if not ref.is_absolute():
        ref = manifest.parent / ref
    ref = ref.resolve()
    if not ref.is_file():
        raise SystemExit(f"validation row {index} reference does not exist: {ref}")
    if ref in seen_refs:
        raise SystemExit(f"validation reference is duplicated: {ref}")
    seen_refs.add(ref)
    seen_sources.add(str(row["source"]))

missing_sources = sorted(required_sources - seen_sources)
if missing_sources:
    raise SystemExit(f"validation manifest does not cover sources: {missing_sources}")
print(f"READY: validation manifest has 64 unique references across {len(seen_sources)} sources")
PY
  then
    failures=$((failures + 1))
  fi
fi

if ! git -C "${REPO}" merge-base --is-ancestor 907f2100e HEAD; then
  echo "NOT READY: execution commit does not contain merged sm100a forward PR #1719" >&2
  failures=$((failures + 1))
fi
if ! git -C "${REPO}" merge-base --is-ancestor 56d4a6074 HEAD; then
  echo "NOT READY: execution commit does not contain corrected Triton backward PR #1730" >&2
  failures=$((failures + 1))
fi
if [[ -n "$(git -C "${REPO}" status --porcelain)" ]]; then
  echo "NOT READY: execution checkout has uncommitted or untracked files: ${REPO}" >&2
  git -C "${REPO}" status --short >&2
  failures=$((failures + 1))
fi

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "NOT READY: fresh v10 output directory is non-empty: ${OUTPUT_DIR}" >&2
  failures=$((failures + 1))
fi

echo "Execution commit: $(git -C "${REPO}" rev-parse HEAD)"
echo "Partition:        ${PARTITION}"
echo "Config:           ${CONFIG}"
echo "Validation:       ${VALIDATION_MANIFEST}"
echo "Output:           ${OUTPUT_DIR}"

if (( failures > 0 )); then
  echo "V10 launch preflight failed with ${failures} issue(s); no command emitted." >&2
  exit 1
fi

execution_commit="$(git -C "${REPO}" rev-parse HEAD)"
printf -v payload \
  'export REPO=%q VENV=%q CONFIG=%q LUSTRE_HOME=%q EXPECTED_V10_COMMIT=%q H3_V10_KERNEL_PREFIX=%q H3_V10_FA4_OVERLAY=%q H3_V10_CUTLASS_PACKAGES=%q PYTHONPATH=%q SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=32 FASTVIDEO_VSA_SM100A=1 H3_V10_KERNEL_GATE=1 PATH=%q SLURM_EXPORT_ENV=ALL; exec bash %q' \
  "${REPO}" "${VENV}" "${CONFIG}" "${LUSTRE_HOME}" "${execution_commit}" \
  "${KERNEL_PREFIX}" "${FA4_OVERLAY}" "${FA4_CUTLASS_PACKAGES}" "${V10_PYTHONPATH}" \
  /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"

printf 'READY: review, then submit exactly:\n'
printf 'sbatch --export=NIL --chdir=%q --nodes=8 --ntasks-per-node=1 --gpus-per-node=4 --exclusive ' "${REPO}"
printf '%q ' -p "${PARTITION}" -t 120:00:00 --requeue -J h3-dmd2-v10 \
  -o "${LOG_DIR}/slurm-%x-%j.out" --wrap="${payload}"
printf '\n'
