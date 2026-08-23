#!/bin/bash
# Validate the shared v10 inputs and print the exact sixteen-tray Slinky submit
# command. This helper never calls sbatch; copy the final line deliberately
# after every gate reports READY.

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
# These four paths are one committed recipe contract. They are deliberately not
# environment overrides: preflight must inspect exactly what the launched YAML
# will consume and write.
readonly CONFIG="${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml"
readonly MAXSHAPE_CONFIG="${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml"
readonly MAXSHAPE_AUDIT_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/v10_maxshape_64g/audit"
readonly BASE_DATA_ROOT="/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v2"
readonly DATA_ROOT="/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3"
readonly MIN_RESOLUTION_COUNT=10
readonly EXPECTED_TRAINING_ROWS=60549
readonly EXPECTED_SHAPE_BUCKETS=87
readonly EXPECTED_SCHEDULED_ROWS=63424
readonly EXPECTED_PADDED_ROWS=2875
readonly EXPECTED_STEPS_PER_EPOCH=991
readonly EXPECTED_VALIDATION_ROWS=60
readonly EXPECTED_VALIDATION_DP_PADDING=4
readonly VALIDATION_MANIFEST="${DATA_ROOT}/validation/heldout60.json"
readonly VALIDATION_MAX_RECORD_NUM_FRAMES=345
readonly OUTPUT_DIR="/mnt/lustre/vlm-wlsaidhi/fastvideo/outputs/minimax_h3_dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64"
readonly NUM_NODES=16
readonly GPUS_PER_NODE=4
readonly WORLD_SIZE=64
readonly SP_SIZE=1
readonly HSDP_REPLICATE=1
readonly HSDP_SHARD=64
readonly TRAIN_BATCH_SIZE=1
readonly GRADIENT_ACCUMULATION_STEPS=1
readonly GLOBAL_BATCH_SIZE=64
readonly MIN_OUTPUT_FREE_BYTES=$((6 * 1024 * 1024 * 1024 * 1024))
readonly REVIEWED_V10_COMMIT="7635a5295b027000a00f6d70789c5cb5886218c3"
LOG_DIR="${LOG_DIR:-/mnt/lustre/vlm-wlsaidhi/fastvideo/logs}"
# rack-3 is the selected v10 recovery lane. Slinky provisions pods from the
# submitted sixteen-node request. This helper never
# submits primers; use unique job names and the documented Slinky warm/race
# procedure when the rack is cold. Override PARTITION only if the operator
# explicitly selects another rack.
PARTITION="${PARTITION:-hpc-rack-3}"
LUSTRE_HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"
KERNEL_PREFIX="${KERNEL_PREFIX:-/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix}"
KERNEL_RECEIPT="${KERNEL_PREFIX}/FASTVIDEO_KERNEL_V10_RECEIPT.json"
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
require_file "${MAXSHAPE_CONFIG}"
require_file "${VENV}/bin/python"
require_file "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
require_file "${REPO}/scripts/train/gate_h3_v10_kernel.sh"
require_file "${REPO}/scripts/preprocess/minimax_h3_native_t2va/finalize_dataset.py"
require_file "${REPO}/scripts/preprocess/minimax_h3_native_t2va/derive_filtered_dataset.py"
require_file "${KERNEL_RECEIPT}"
require_file "${DATA_ROOT}/FROZEN_MANIFEST.json"
require_file "${DATA_ROOT}/DERIVATION_RECEIPT.json"
require_file "${DATA_ROOT}/READY.json"
require_file "${BASE_DATA_ROOT}/FROZEN_MANIFEST.json"
require_file "${BASE_DATA_ROOT}/READY.json"
if [[ ! -d "${FA4_OVERLAY}/flash_attn/cute" ]]; then
  echo "NOT READY: missing FA4 CuTe package under ${FA4_OVERLAY}" >&2
  failures=$((failures + 1))
fi
if [[ ! -d "${FA4_CUTLASS_PACKAGES}/cutlass" ]]; then
  echo "NOT READY: missing pinned CUTLASS DSL package under ${FA4_CUTLASS_PACKAGES}" >&2
  failures=$((failures + 1))
fi

# Parse the committed YAML without importing FastVideo (which probes GPU
# backends at package import). This guards future edits from making preflight
# validate different data, validation, or output paths than training uses.
if [[ -x "${VENV}/bin/python" && -f "${CONFIG}" ]]; then
  if ! "${VENV}/bin/python" - \
    "${CONFIG}" "${DATA_ROOT}" "${VALIDATION_MANIFEST}" "${VALIDATION_MAX_RECORD_NUM_FRAMES}" \
    "${OUTPUT_DIR}" "${NUM_NODES}" "${GPUS_PER_NODE}" "${WORLD_SIZE}" \
    "${SP_SIZE}" "${HSDP_REPLICATE}" "${HSDP_SHARD}" "${TRAIN_BATCH_SIZE}" \
    "${GRADIENT_ACCUMULATION_STEPS}" "${GLOBAL_BATCH_SIZE}" "${sources[@]}" <<'PY'
import pathlib
import sys

import yaml

config_path = pathlib.Path(sys.argv[1])
data_root = pathlib.Path(sys.argv[2])
validation_manifest = sys.argv[3]
validation_max_record_num_frames = int(sys.argv[4])
output_dir = sys.argv[5]
num_nodes = int(sys.argv[6])
gpus_per_node = int(sys.argv[7])
world_size = int(sys.argv[8])
sp_size = int(sys.argv[9])
hsdp_replicate = int(sys.argv[10])
hsdp_shard = int(sys.argv[11])
train_batch_size = int(sys.argv[12])
gradient_accumulation_steps = int(sys.argv[13])
global_batch_size = int(sys.argv[14])
sources = sys.argv[15:]
document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
try:
    training = document["training"]
    distributed = training["distributed"]
    data = training["data"]
    loop = training["loop"]
    model = training["model"]
    actual_data_paths = training["data"]["data_path"]
    validation = document["callbacks"]["validation"]
    actual_validation = validation["dataset_file"]
    actual_validation_max_record_num_frames = validation["max_record_num_frames"]
    actual_output = training["checkpoint"]["output_dir"]
    checkpoint = training["checkpoint"]
except (KeyError, TypeError) as error:
    raise SystemExit(f"v10 YAML is missing a required launch path: {error}") from error

expected_data_paths = [str(data_root / source / "data") for source in sources]
if actual_data_paths != expected_data_paths:
    raise SystemExit(f"v10 YAML data_path {actual_data_paths!r} != fixed preflight roots {expected_data_paths!r}")
if actual_validation != validation_manifest:
    raise SystemExit(f"v10 YAML validation dataset {actual_validation!r} != {validation_manifest!r}")
if actual_validation_max_record_num_frames != validation_max_record_num_frames:
    raise SystemExit("v10 YAML validation max_record_num_frames "
                     f"{actual_validation_max_record_num_frames!r} != {validation_max_record_num_frames}")
if actual_output != output_dir:
    raise SystemExit(f"v10 YAML output directory {actual_output!r} != {output_dir!r}")
expected_checkpoint = {
    "save_inference_checkpoint_on_validation": True,
    "inference_checkpoint_role": "student",
    "inference_checkpoint_dtype": "bfloat16",
    "training_state_checkpointing_steps": 100,
    "require_complete_training_checkpoint": True,
    "checkpointing_start_step": 100,
    "checkpoints_total_limit": 3,
}
actual_checkpoint = {key: checkpoint.get(key) for key in expected_checkpoint}
if actual_checkpoint != expected_checkpoint:
    raise SystemExit(f"v10 checkpoint policy {actual_checkpoint!r} != {expected_checkpoint!r}")
if int(validation.get("every_steps", 0)) != 100 or validation.get("run_at_start") is not True:
    raise SystemExit("v10 validation must run at step zero and every 100 steps so each event receives an "
                     "unlimited-retention inference checkpoint")
if validation.get("sampling_steps") != [4]:
    raise SystemExit(f"v10 validation must use the exact four-forward schedule; got {validation.get('sampling_steps')!r}")
if num_nodes * gpus_per_node != world_size:
    raise SystemExit(f"preflight allocation {num_nodes}x{gpus_per_node} != world size {world_size}")
expected_topology = {
    "num_gpus": world_size,
    "sp_size": sp_size,
    "tp_size": 1,
    "hsdp_replicate_dim": hsdp_replicate,
    "hsdp_shard_dim": hsdp_shard,
}
actual_topology = {key: int(distributed[key]) for key in expected_topology}
if actual_topology != expected_topology:
    raise SystemExit(f"v10 YAML topology {actual_topology!r} != {expected_topology!r}")
actual_train_batch_size = int(data["train_batch_size"])
actual_gradient_accumulation_steps = int(loop["gradient_accumulation_steps"])
if actual_train_batch_size != train_batch_size:
    raise SystemExit(f"v10 YAML train batch {actual_train_batch_size} != {train_batch_size}")
if actual_gradient_accumulation_steps != gradient_accumulation_steps:
    raise SystemExit("v10 YAML accumulation "
                     f"{actual_gradient_accumulation_steps} != {gradient_accumulation_steps}")
actual_global_batch_size = (world_size // sp_size) * train_batch_size * actual_gradient_accumulation_steps
if actual_global_batch_size != global_batch_size:
    raise SystemExit(f"v10 effective global batch {actual_global_batch_size} != {global_batch_size}")
if hsdp_replicate * hsdp_shard != world_size:
    raise SystemExit(f"HSDP mesh {hsdp_replicate}x{hsdp_shard} != world size {world_size}")
if model.get("enable_torch_compile") is not True:
    raise SystemExit("v10 must enable the #1718 modular regional-compile path")
compile_kwargs = model.get("torch_compile_kwargs", {}) or {}
if compile_kwargs.get("fullgraph", True) is not True or "mode" in compile_kwargs:
    raise SystemExit("v10 regional compile requires fullgraph=True and forbids torch_compile_kwargs.mode")
if compile_kwargs.get("dynamic") is not True:
    raise SystemExit("v10 native-shape regional compile requires torch_compile_kwargs.dynamic=true")
if compile_kwargs.get("recompile_limit") != 32:
    raise SystemExit("v10 regional compile requires the reviewed per-call recompile_limit=32")
print("READY: committed v10 YAML paths/topology match the fixed 64-GPU preflight contract; "
      f"global batch={actual_global_batch_size}, regional compile enabled")
PY
  then
    failures=$((failures + 1))
  fi
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

# Pin the exact filtered corpus and the world-64 sampler arithmetic. The
# finalizer verifies row contents; this lightweight check verifies the launch
# recipe will see exactly the intended retained buckets and padding envelope.
if (( failures == 0 )); then
  if ! "${VENV}/bin/python" - \
    "${DATA_ROOT}" "${MIN_RESOLUTION_COUNT}" "${EXPECTED_TRAINING_ROWS}" \
    "${EXPECTED_SHAPE_BUCKETS}" "${EXPECTED_SCHEDULED_ROWS}" \
    "${EXPECTED_PADDED_ROWS}" "${EXPECTED_STEPS_PER_EPOCH}" "${WORLD_SIZE}" \
    "${sources[@]}" <<'PY'
import collections
import hashlib
import json
import pathlib
import pickle
import re
import sys

data_root = pathlib.Path(sys.argv[1])
threshold, expected_rows, expected_buckets, expected_scheduled = map(int, sys.argv[2:6])
expected_padding, expected_steps, world_size = map(int, sys.argv[6:9])
sources = sys.argv[9:]
receipt = json.loads((data_root / "DERIVATION_RECEIPT.json").read_text(encoding="utf-8"))
expected_receipt = {
    "min_resolution_count": threshold,
    "excluded_resolutions": ["576x576", "640x480", "832x480"],
    "excluded_frozen_rows": 7,
    "excluded_training_rows": 3,
    "base_frozen_rows": 60629,
    "derived_training_rows": expected_rows,
    "base_validation_rows": 64,
    "derived_validation_rows": 60,
    "excluded_validation_rows": 4,
    "excluded_validation_conditioning_ids": [
        "t2va-0020260818-000002",
        "t2va-0020260818-000004",
        "t2va-0020260818-000007",
        "t2va-0020260818-000008",
    ],
    "training_holdout_policy": "preserve_base_validation_conditioning_ids",
    "validation_payload_path": "validation/heldout60.json",
}
observed_receipt = {
    "min_resolution_count": receipt.get("filter", {}).get("min_resolution_count"),
    **{key: receipt.get(key) for key in expected_receipt if key != "min_resolution_count"},
}
if observed_receipt != expected_receipt:
    raise SystemExit(f"v3 derivation receipt {observed_receipt!r} != {expected_receipt!r}")
if receipt.get("schema_version") != "minimax-h3-native-t2va-filtered-derivation-v2":
    raise SystemExit(f"unexpected v3 derivation schema: {receipt.get('schema_version')!r}")
expected_source_rows = dict(zip(sources, [29052, 16728, 9980, 2884, 1905], strict=True))
expected_current_validation_exclusions = dict(zip(sources, [12, 16, 16, 9, 16], strict=True))
expected_base_validation_exclusions = dict(zip(sources, [12, 16, 20, 13, 16], strict=True))
for source in sources:
    source_receipt = receipt.get("sources", {}).get(source, {})
    observed_source = {
        "derived_training_rows": source_receipt.get("derived_training_rows"),
        "validation_exclusions": source_receipt.get("validation_exclusions"),
        "base_validation_exclusions": source_receipt.get("base_validation_exclusions"),
    }
    expected_source = {
        "derived_training_rows": expected_source_rows[source],
        "validation_exclusions": expected_current_validation_exclusions[source],
        "base_validation_exclusions": expected_base_validation_exclusions[source],
    }
    if observed_source != expected_source:
        raise SystemExit(f"v3 source receipt {source}: {observed_source!r} != {expected_source!r}")

ready = json.loads((data_root / "READY.json").read_text(encoding="utf-8"))
validation_manifest_path = data_root / "validation" / "manifest.jsonl"
validation_payload_path = data_root / receipt["validation_payload_path"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected_validation_hashes = {
    "validation_manifest_sha256": sha256(validation_manifest_path),
    "validation_summary_sha256": sha256(data_root / "validation" / "manifest.json"),
    "validation_payload_sha256": sha256(validation_payload_path),
}
for field, actual_sha256 in expected_validation_hashes.items():
    if receipt.get(field) != actual_sha256 or ready.get(field) != actual_sha256:
        raise SystemExit(
            f"v3 {field} is not jointly anchored by DERIVATION_RECEIPT and READY: "
            f"file={actual_sha256} receipt={receipt.get(field)} ready={ready.get(field)}"
        )
if ready.get("validation_payload_path") != receipt["validation_payload_path"]:
    raise SystemExit("v3 READY validation payload path does not match the derivation receipt")
if ready.get("training_rows") != expected_rows or ready.get("validation_rows") != 60:
    raise SystemExit(
        f"v3 READY row counts training/validation={ready.get('training_rows')}/{ready.get('validation_rows')} "
        f"!= {expected_rows}/60"
    )

bucket_pattern = re.compile(r"^bucket=([1-9][0-9]*)x([1-9][0-9]*)-([1-9][0-9]*)f$")
bucket_rows = collections.Counter()
for source in sources:
    cache_path = data_root / source / "data" / "map_style_cache" / "file_info.pkl"
    with cache_path.open("rb") as handle:
        paths, lengths = pickle.load(handle)
    if len(paths) != len(lengths):
        raise SystemExit(f"corrupt map-style cache: {cache_path}")
    for path, length in zip(paths, lengths, strict=True):
        match = bucket_pattern.fullmatch(pathlib.Path(path).parent.name)
        if match is None:
            raise SystemExit(f"invalid bucket path in {cache_path}: {path}")
        width, height, _frames = match.groups()
        if f"{width}x{height}" in expected_receipt["excluded_resolutions"]:
            raise SystemExit(f"excluded resolution leaked into v3 cache: {path}")
        bucket_rows[pathlib.Path(path).parent.name] += int(length)

rows = sum(bucket_rows.values())
scheduled = sum(((count + world_size - 1) // world_size) * world_size for count in bucket_rows.values())
padding = scheduled - rows
observed = (rows, len(bucket_rows), scheduled, padding, scheduled // world_size)
expected = (expected_rows, expected_buckets, expected_scheduled, expected_padding, expected_steps)
if observed != expected:
    raise SystemExit(f"v3 sampler contract {observed} != {expected}")
print(
    f"READY: filtered v3 has {rows} rows in {len(bucket_rows)} buckets; "
    f"world-{world_size} schedules {scheduled} rows/{expected_steps} steps with {padding} repeats"
)
PY
  then
    failures=$((failures + 1))
  fi
fi

# Verify the derivation policy and its hash-anchored ancestry to v2, including
# the whole-resolution threshold, filtered heldout60 split, and inherited
# base-validation training exclusions. The derived
# verifier invokes the ordinary finalizer internally to re-audit every v3 done
# marker, parquet row, bucket geometry, cache tuple, source receipt, and count.
if (( failures == 0 )); then
  if ! "${VENV}/bin/python" \
    "${REPO}/scripts/preprocess/minimax_h3_native_t2va/derive_filtered_dataset.py" \
    --base-root "${BASE_DATA_ROOT}" --output-root "${DATA_ROOT}" \
    --min-resolution-count "${MIN_RESOLUTION_COUNT}" --verify-only; then
    echo "NOT READY: filtered v3 derivation verification failed" >&2
    failures=$((failures + 1))
  fi
fi

require_file "${VALIDATION_MANIFEST}"
if [[ -x "${VENV}/bin/python" && -f "${VALIDATION_MANIFEST}" ]]; then
  if ! "${VENV}/bin/python" - "${VALIDATION_MANIFEST}" "${VALIDATION_MAX_RECORD_NUM_FRAMES}" \
    "${EXPECTED_VALIDATION_ROWS}" "${EXPECTED_VALIDATION_DP_PADDING}" "${WORLD_SIZE}" "${sources[@]}" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
max_record_num_frames = int(sys.argv[2])
expected_rows = int(sys.argv[3])
expected_padding = int(sys.argv[4])
world_size = int(sys.argv[5])
required_sources = set(sys.argv[6:])
document = json.loads(manifest.read_text(encoding="utf-8"))
rows = document.get("data") if isinstance(document, dict) else document
if not isinstance(rows, list) or len(rows) != expected_rows:
    raise SystemExit(
        f"validation manifest must contain exactly {expected_rows} data rows; "
        f"got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}"
    )

seen_sources = set()
seen_refs = set()
sample_ids = []
capped_rows = 0
excluded_resolutions = {"576x576", "640x480", "832x480"}
for index, row in enumerate(rows):
    required = ("caption", "ref_video", "source", "sample_id", "width", "height", "num_frames")
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"validation row {index} is missing {missing}")
    width, height, frames = (int(row["width"]), int(row["height"]), int(row["num_frames"]))
    if min(width, height, frames) <= 0 or width % 8 or height % 8:
        raise SystemExit(f"validation row {index} has invalid shape {width}x{height}x{frames}f")
    if f"{width}x{height}" in excluded_resolutions:
        raise SystemExit(f"rare resolution leaked into filtered validation row {index}: {width}x{height}")
    requested_frames = min(frames, max_record_num_frames)
    aligned_frames = requested_frames
    while aligned_frames % 17 != 5:
        aligned_frames += 1
    if not 5 <= aligned_frames / 24 <= 15:
        raise SystemExit(f"validation row {index} resolves to unsupported H3 target {aligned_frames}f")
    capped_rows += int(requested_frames != frames)
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
    sample_ids.append(str(row["sample_id"]))

if len(set(sample_ids)) != expected_rows:
    raise SystemExit(f"validation manifest has duplicate sample IDs: {len(set(sample_ids))}/{expected_rows}")
missing_sources = sorted(required_sources - seen_sources)
if missing_sources:
    raise SystemExit(f"validation manifest does not cover sources: {missing_sources}")
padding = (-len(rows)) % world_size
if padding != expected_padding:
    raise SystemExit(f"validation DP-{world_size} padding {padding} != {expected_padding}")
padded_ids = sample_ids + sample_ids[:padding]
if len(padded_ids) != world_size or padded_ids[-padding:] != sample_ids[:padding]:
    raise SystemExit("validation DP padding does not repeat exactly the first four retained records")
if any(record_id in {
    "t2va-0020260818-000002",
    "t2va-0020260818-000004",
    "t2va-0020260818-000007",
    "t2va-0020260818-000008",
} for record_id in padded_ids):
    raise SystemExit("filtered validation DP padding reintroduced an excluded conditioning ID")
print(
    f"READY: validation manifest has {len(rows)} unique references across {len(seen_sources)} sources; "
    f"DP-{world_size} repeats {padding} retained records; "
    f"{capped_rows} generation requests cap at {max_record_num_frames}f"
)
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
if ! git -C "${REPO}" merge-base --is-ancestor "${REVIEWED_V10_COMMIT}" HEAD; then
  echo "NOT READY: execution commit does not contain reviewed v10 marker ${REVIEWED_V10_COMMIT}" >&2
  failures=$((failures + 1))
fi
if [[ -x "${VENV}/bin/python" && -f "${KERNEL_RECEIPT}" ]]; then
  if ! "${VENV}/bin/python" - "${KERNEL_RECEIPT}" "${REPO}" <<'PY'
import json
import pathlib
import subprocess
import sys

receipt_path = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("schema_version") != "fastvideo-h3-v10-kernel-v1":
    raise SystemExit(f"NOT READY: unexpected kernel receipt schema in {receipt_path}")
execution_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if receipt.get("source_commit") != execution_commit:
    raise SystemExit("NOT READY: kernel receipt source "
                     f"{receipt.get('source_commit')} != execution HEAD {execution_commit}; "
                     "rebuild from the final commit")
kernel_tree = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD:fastvideo-kernel"], text=True).strip()
if receipt.get("kernel_tree") != kernel_tree:
    raise SystemExit("NOT READY: kernel receipt tree "
                     f"{receipt.get('kernel_tree')} != execution tree {kernel_tree}")
print(f"READY: kernel receipt is bound to execution HEAD {execution_commit}")
PY
  then
    failures=$((failures + 1))
  fi
fi

# Production is allowed only after the final execution commit has completed
# the exact 64-rank, largest-shape critic/student capacity gate. Receipts from
# older commits or a modified gate YAML are deliberately ignored.
if [[ -x "${VENV}/bin/python" && -f "${MAXSHAPE_CONFIG}" ]]; then
  if ! "${VENV}/bin/python" - "${MAXSHAPE_AUDIT_ROOT}" "${MAXSHAPE_CONFIG}" "${REPO}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

audit_root = pathlib.Path(sys.argv[1])
config = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3])
execution_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
matches = []
for result_path in sorted(audit_root.glob("job-*/RESULT.json")):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    checks = result.get("checks", {})
    if (
        result.get("schema_version") == "fastvideo-h3-v10-maxshape-gate-v1"
        and result.get("success") is True
        and checks
        and all(value is True for value in checks.values())
        and result.get("execution_commit") == execution_commit
        and result.get("config") == str(config)
        and result.get("config_sha256") == config_sha256
        and result.get("shape") == {
            "width": 1760,
            "height": 768,
            "num_frames": 362,
            "video_latent_shape": [24, 107, 48, 110],
        }
        and result.get("steps") == {"critic": 1, "student": 2}
    ):
        matches.append((result_path, result))
if not matches:
    raise SystemExit(
        "NOT READY: no successful final-commit 64-GPU 1760x768x362 capacity receipt under "
        f"{audit_root} for {execution_commit}"
    )
path, result = matches[-1]
observed = result.get("observed_gpu_memory", {})
print(
    f"READY: max-shape capacity gate {path} passed on job {result.get('job_id')} with "
    f"observed external peak {observed.get('max_used_mib')} MiB"
)
PY
  then
    failures=$((failures + 1))
  fi
fi
if [[ -n "$(git -C "${REPO}" status --porcelain)" ]]; then
  echo "NOT READY: execution checkout has uncommitted or untracked files: ${REPO}" >&2
  git -C "${REPO}" status --short >&2
  failures=$((failures + 1))
fi

# A pre-step-100 failure has no resumable training state, but it may have
# already published the immutable step-zero inference export and all 64
# DP-padded validation videos (60 retained records plus four repeats). Accept
# only that exact, fully validated namespace. This
# makes an incident restart idempotent without deleting a good 66-GiB export
# or accidentally resuming incompatible optimizer/RNG state.
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if ! "${VENV}/bin/python" - \
    "${OUTPUT_DIR}" "${WORLD_SIZE}" "${CONFIG}" "${DATA_ROOT}" "${VALIDATION_MANIFEST}" <<'PY'
import json
import pathlib
import sys

import yaml

output_dir = pathlib.Path(sys.argv[1])
world_size = int(sys.argv[2])
config_path = pathlib.Path(sys.argv[3])
data_root = pathlib.Path(sys.argv[4])
validation_manifest = sys.argv[5]
current_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected_data_paths = current_config["training"]["data"]["data_path"]
expected_validation_manifest = current_config["callbacks"]["validation"]["dataset_file"]
if any(not str(path).startswith(f"{data_root}/") for path in expected_data_paths):
    raise SystemExit("NOT READY: current v10 config data paths are outside the filtered v3 root")
if expected_validation_manifest != validation_manifest:
    raise SystemExit("NOT READY: current v10 validation path differs from the fixed v3 launch contract")
validation_names = {
    f"validation_step_0_inference_steps_4_rank_{rank}_video_0.mp4"
    for rank in range(world_size)
}
allowed_top_level = {"inference", "tracker", *validation_names}
actual_top_level = {path.name for path in output_dir.iterdir()}
unexpected = sorted(actual_top_level - allowed_top_level)
missing = sorted(({"inference", *validation_names}) - actual_top_level)
if unexpected or missing:
    raise SystemExit(
        "NOT READY: v10 restart output is not the exact step-zero namespace; "
        f"unexpected={unexpected} missing={missing}"
    )

training_checkpoints = sorted(path.name for path in output_dir.glob("checkpoint-*"))
training_staging = sorted(path.name for path in output_dir.glob(".checkpoint-*"))
if training_checkpoints or training_staging:
    raise SystemExit(
        "NOT READY: step-zero restart namespace contains resumable/staging training state; "
        f"checkpoints={training_checkpoints} staging={training_staging}"
    )

for name in validation_names:
    video = output_dir / name
    if not video.is_file() or video.stat().st_size <= 0:
        raise SystemExit(f"NOT READY: missing or empty step-zero validation video: {video}")

inference_root = output_dir / "inference"
checkpoint = inference_root / "checkpoint-0"
inference_entries = sorted(path.name for path in inference_root.iterdir())
if inference_entries != ["checkpoint-0"]:
    raise SystemExit(
        "NOT READY: inference namespace must contain only checkpoint-0; "
        f"found={inference_entries}"
    )
if (checkpoint / ".complete").read_text(encoding="utf-8") != "complete\n":
    raise SystemExit(f"NOT READY: invalid inference completion marker: {checkpoint / '.complete'}")
metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
expected_metadata = {
    "format_version": 1,
    "kind": "inference",
    "step": 0,
    "role": "student",
    "dtype": "bfloat16",
    "module": "transformer",
}
observed_metadata = {key: metadata.get(key) for key in expected_metadata}
if observed_metadata != expected_metadata:
    raise SystemExit(
        "NOT READY: checkpoint-0 metadata is not the v10 student inference contract; "
        f"observed={observed_metadata}"
    )
saved_config = metadata.get("config", {})
saved_training = saved_config.get("training", {})
saved_distributed = saved_training.get("distributed", {})
saved_method = saved_config.get("method", {})
saved_data_paths = saved_training.get("data", {}).get("data_path")
saved_validation_manifest = saved_config.get("callbacks", {}).get("validation", {}).get("dataset_file")
if saved_data_paths != expected_data_paths or saved_validation_manifest != expected_validation_manifest:
    raise SystemExit(
        "NOT READY: checkpoint-0 belongs to a different data/validation recipe; "
        f"saved_data={saved_data_paths!r} expected_data={expected_data_paths!r} "
        f"saved_validation={saved_validation_manifest!r} expected_validation={expected_validation_manifest!r}"
    )
if saved_training.get("checkpoint", {}).get("output_dir") != str(output_dir):
    raise SystemExit("NOT READY: checkpoint-0 metadata belongs to a different output namespace")
if saved_distributed != {
    "num_gpus": 64,
    "sp_size": 1,
    "tp_size": 1,
    "hsdp_replicate_dim": 1,
    "hsdp_shard_dim": 64,
}:
    raise SystemExit(f"NOT READY: checkpoint-0 topology is not v10 fsdp64: {saved_distributed}")
if saved_method.get("dmd_denoising_steps") != [999, 749, 500, 250]:
    raise SystemExit("NOT READY: checkpoint-0 does not use the trained four-forward ladder")

module_dir = checkpoint / "transformer"
index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(f"NOT READY: checkpoint-0 has an invalid shard index: {index_path}")
expected_shards = {filename for filename in weight_map.values()}
actual_shards = {path.name for path in module_dir.glob("*.safetensors") if path.is_file()}
if expected_shards != actual_shards or any((module_dir / name).stat().st_size <= 0 for name in expected_shards):
    raise SystemExit(
        "NOT READY: checkpoint-0 shard set differs from its index; "
        f"missing={sorted(expected_shards - actual_shards)} extra={sorted(actual_shards - expected_shards)}"
    )
if metadata.get("shard_count") != len(expected_shards) or metadata.get("tensor_count") != len(weight_map):
    raise SystemExit("NOT READY: checkpoint-0 shard/tensor counts differ from its index")
print(
    "READY: validated same-recipe pre-step-100 restart namespace: no training state, "
    "complete step-zero bf16 student export, and 64 four-forward validation videos"
)
PY
  then
    failures=$((failures + 1))
  fi
else
  echo "READY: fresh v10 output namespace"
fi
output_parent="$(dirname "${OUTPUT_DIR}")"
if [[ -d "${output_parent}" ]]; then
  available_bytes="$(df -B1 --output=avail "${output_parent}" | tail -n 1 | tr -d ' ')"
  if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || (( available_bytes < MIN_OUTPUT_FREE_BYTES )); then
    echo "NOT READY: output filesystem has ${available_bytes:-unknown} free bytes; " \
      "v10 checkpoint policy requires at least ${MIN_OUTPUT_FREE_BYTES} before launch" >&2
    failures=$((failures + 1))
  else
    echo "READY: output filesystem has ${available_bytes} free bytes"
  fi
else
  echo "NOT READY: output parent does not exist: ${output_parent}" >&2
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
  'export REPO=%q VENV=%q CONFIG=%q LUSTRE_HOME=%q EXPECTED_V10_COMMIT=%q H3_V10_KERNEL_PREFIX=%q H3_V10_FA4_OVERLAY=%q H3_V10_CUTLASS_PACKAGES=%q PYTHONPATH=%q SP_SIZE=%q HSDP_REPLICATE=%q HSDP_SHARD=%q FASTVIDEO_VSA_SM100A=1 H3_V10_KERNEL_GATE=1 H3_V10_COMPILE_LOGS=1 PATH=%q SLURM_EXPORT_ENV=ALL; exec bash %q' \
  "${REPO}" "${VENV}" "${CONFIG}" "${LUSTRE_HOME}" "${execution_commit}" \
  "${KERNEL_PREFIX}" "${FA4_OVERLAY}" "${FA4_CUTLASS_PACKAGES}" "${V10_PYTHONPATH}" \
  "${SP_SIZE}" "${HSDP_REPLICATE}" "${HSDP_SHARD}" \
  /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"

printf 'READY: review, then submit exactly:\n'
printf 'sbatch --export=NIL --chdir=%q --nodes=%q --ntasks-per-node=1 --gpus-per-node=%q --exclusive ' \
  "${REPO}" "${NUM_NODES}" "${GPUS_PER_NODE}"
printf '%q ' -p "${PARTITION}" -t 120:00:00 --requeue -J h3-dmd2-v10-64g \
  -o "${LOG_DIR}/slurm-%x-%j.out" --wrap="${payload}"
printf '\n'
