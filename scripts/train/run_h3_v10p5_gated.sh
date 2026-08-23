#!/bin/bash
#SBATCH --job-name=h3-dmd2-v10p5-32g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=120:00:00
#SBATCH --partition=hpc-rack-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/slurm-%x-%j.out

# V10.5 launch: immutable recipe audit, an exact max-shape critic/student
# capacity gate for one topology, then requeueable production.

set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: sbatch run_h3_v10p5_gated.sh <40-character-execution-commit>" >&2
  exit 2
fi

readonly REPO="/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10p5"
readonly VENV="${REPO}/.venv"
readonly CONFIG_DIR="${REPO}/examples/train/configs/distribution_matching/minimax_h3"
readonly PROD_CONFIG="${CONFIG_DIR}/dmd2_sp1_fsdp32_v10p5_datafree_mixed_vsa64.yaml"
readonly MAXSHAPE_CONFIG="${CONFIG_DIR}/dmd2_sp1_fsdp32_v10p5_datafree_maxshape_gate_vsa64.yaml"
readonly SP4_PROD_CONFIG="${CONFIG_DIR}/dmd2_sp4_fsdp32_v10p5_datafree_mixed_vsa64.yaml"
readonly SP4_MAXSHAPE_CONFIG="${CONFIG_DIR}/dmd2_sp4_fsdp32_v10p5_datafree_maxshape_gate_vsa64.yaml"
readonly DATA_ROOT="/mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3"
readonly OUTPUT_DIR="/mnt/lustre/vlm-wlsaidhi/fastvideo/outputs/minimax_h3_dmd2_sp1_fsdp32_v10p5_datafree_mixed_vsa64"
readonly MAXSHAPE_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/v10p5_datafree_maxshape_sp1"
readonly MAXSHAPE_RECEIPT="${MAXSHAPE_ROOT}/READY.json"
readonly SP4_MAXSHAPE_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/v10p5_datafree_maxshape_sp4"
readonly SP4_MAXSHAPE_RECEIPT="${SP4_MAXSHAPE_ROOT}/READY.json"
readonly LUSTRE_HOME="/mnt/lustre/vlm-wlsaidhi"
readonly KERNEL_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/v10p5_kernel"
readonly KERNEL_PREFIX="${KERNEL_ROOT}/prefix"
readonly FA4_OVERLAY="/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay"
readonly FA4_CUTLASS_PACKAGES="${FA4_OVERLAY}/nvidia_cutlass_dsl/python_packages"
readonly EXPECTED_V10_COMMIT="$1"
readonly START_TOPOLOGY="${V10P5_START_TOPOLOGY:-auto}"

if [[ "${START_TOPOLOGY}" != "auto" && "${START_TOPOLOGY}" != "sp4" ]]; then
  echo "V10.5 GATE FAILED: V10P5_START_TOPOLOGY must be auto or sp4; got ${START_TOPOLOGY}" >&2
  exit 2
fi

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export SLURM_EXPORT_ENV=ALL
export REPO VENV LUSTRE_HOME EXPECTED_V10_COMMIT
export SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=32
export H3_V10_KERNEL_PREFIX="${KERNEL_PREFIX}"
export H3_V10_FA4_OVERLAY="${FA4_OVERLAY}"
export H3_V10_CUTLASS_PACKAGES="${FA4_CUTLASS_PACKAGES}"
export PYTHONPATH="${KERNEL_PREFIX}:${FA4_OVERLAY}:${FA4_CUTLASS_PACKAGES}"
export FASTVIDEO_VSA_SM100A=1
export H3_V10_COMPILE_LOGS=1

if [[ "${SLURM_JOB_NUM_NODES:-0}" != "8" ]]; then
  echo "V10.5 GATE FAILED: requires exactly 8 trays; got ${SLURM_JOB_NUM_NODES:-unset}" >&2
  exit 1
fi
gpus_per_node="${SLURM_GPUS_PER_NODE:-4}"
gpus_per_node="${gpus_per_node##*:}"
if [[ "${gpus_per_node}" != "4" ]]; then
  echo "V10.5 GATE FAILED: requires four GPUs per tray; got ${SLURM_GPUS_PER_NODE:-unset}" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "V10.5 GATE FAILED: missing compute Python ${VENV}/bin/python" >&2
  exit 1
fi

# Gates are never automatically requeued. Requeue is enabled only immediately
# before production, after this exact job has written a passing receipt.
scontrol update JobId="${SLURM_JOB_ID}" Requeue=0

require_exact_checkout() {
  local observed
  observed="$(git -C "${REPO}" rev-parse HEAD)"
  if [[ "${observed}" != "${EXPECTED_V10_COMMIT}" ]]; then
    echo "V10.5 GATE FAILED: execution HEAD ${observed} != ${EXPECTED_V10_COMMIT}" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${REPO}" status --porcelain)" ]]; then
    echo "V10.5 GATE FAILED: execution checkout is dirty" >&2
    git -C "${REPO}" status --short >&2
    exit 1
  fi
}

require_exact_checkout

echo "=== V10.5 immutable recipe/data audit ==="
"${VENV}/bin/python" - \
  "${PROD_CONFIG}" "${MAXSHAPE_CONFIG}" "${SP4_PROD_CONFIG}" "${SP4_MAXSHAPE_CONFIG}" \
  "${DATA_ROOT}" "${OUTPUT_DIR}" <<'PY'
import hashlib
import json
import math
import pathlib
import pickle
import re
import sys

import pyarrow.parquet as pq
import yaml

prod_path, gate_path, sp4_prod_path, sp4_gate_path, data_root, output_dir = map(pathlib.Path, sys.argv[1:])
prod = yaml.safe_load(prod_path.read_text(encoding="utf-8"))
gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
sp4_prod = yaml.safe_load(sp4_prod_path.read_text(encoding="utf-8"))
sp4_gate = yaml.safe_load(sp4_gate_path.read_text(encoding="utf-8"))
sources = [
    "h3_t2av_video_nuva_50k_720_mixed_len",
    "h3_t2av_video_5s_768p",
    "h3_t2av_video_nuva_10k_720_mixed_len",
    "h3_t2av_video_nuva_10k_mixed_res_len",
    "h3_t2av_fastgen_vidprom_150k",
]
expected_paths = [str(data_root / source / "data") for source in sources]

method = prod["method"]
training = prod["training"]
distributed = training["distributed"]
data = training["data"]
loop = training["loop"]
checkpoint = training["checkpoint"]
validation = prod["callbacks"]["validation"]
expected_method = {
    "rollout_mode": "simulate",
    "rollout_carry": True,
    "rollout_carry_slots": 2,
    "rollout_sample_type": "ode",
    "generator_update_interval": 5,
    "dmd_denoising_steps": [999, 749, 500, 250],
    "score_timestep_shift": 2.4,
    "fake_score_loss_space": "x0",
}
for key, value in expected_method.items():
    if method.get(key) != value:
        raise SystemExit(f"V10.5 method {key}={method.get(key)!r} != {value!r}")
if method.get("rollout_data_forcing", False):
    raise SystemExit("V10.5 must not enable data forcing")
expected_topology = {
    "num_gpus": 32,
    "sp_size": 1,
    "tp_size": 1,
    "hsdp_replicate_dim": 1,
    "hsdp_shard_dim": 32,
}
if {key: int(distributed[key]) for key in expected_topology} != expected_topology:
    raise SystemExit(f"V10.5 topology mismatch: {distributed}")
if data.get("data_path") != expected_paths:
    raise SystemExit(f"V10.5 data roots differ from filtered V10: {data.get('data_path')}")
if data.get("preprocessed_data_type") != "text_only" or data.get("native_shape_bucketing") is not True:
    raise SystemExit("V10.5 requires projected text-only reads with native-shape bucketing")
global_batch = (32 // 1) * int(data["train_batch_size"]) * int(loop["gradient_accumulation_steps"])
if global_batch != 64:
    raise SystemExit(f"V10.5 global batch is {global_batch}, expected 64")
if float(training["optimizer"]["learning_rate"]) != 2e-6 or float(method["fake_score_learning_rate"]) != 2e-6:
    raise SystemExit("V10.5 student and critic learning rates must both be 2e-6")
if pathlib.Path(checkpoint["output_dir"]) != output_dir:
    raise SystemExit(f"V10.5 output mismatch: {checkpoint['output_dir']}")
expected_checkpoint = {
    "save_inference_checkpoint_on_validation": True,
    "inference_checkpoint_role": "student",
    "inference_checkpoint_dtype": "bfloat16",
    "training_state_checkpointing_steps": 100,
    "require_complete_training_checkpoint": True,
    "checkpointing_start_step": 100,
    "checkpoints_total_limit": 3,
}
if {key: checkpoint.get(key) for key in expected_checkpoint} != expected_checkpoint:
    raise SystemExit(f"V10.5 checkpoint policy mismatch: {checkpoint}")
if validation.get("sampling_steps") != [4] or validation.get("run_at_start") is not True:
    raise SystemExit("V10.5 validation must run the four-forward schedule at step zero")
heldout = data_root / "validation/heldout60.json"
if validation.get("dataset_file") != str(heldout):
    raise SystemExit("V10.5 validation must use V10 heldout60")
if hashlib.sha256(heldout.read_bytes()).hexdigest() != "66618beb196f9a68d3de983c6349f655c14339324600398bd9aad2293965f5d9":
    raise SystemExit("heldout60 hash changed")
compile_config = training["model"]
if compile_config.get("enable_torch_compile") is not True:
    raise SystemExit("V10.5 regional compilation is disabled")
if compile_config.get("torch_compile_kwargs") != {"dynamic": True, "recompile_limit": 32}:
    raise SystemExit(f"V10.5 compile policy mismatch: {compile_config.get('torch_compile_kwargs')}")

text_columns = {
    "id",
    "text_embedding_bytes",
    "text_embedding_shape",
    "text_embedding_dtype",
    "caption",
}
bucket_pattern = re.compile(r"^bucket=(\d+)x(\d+)-(\d+)f$")
bucket_counts = {}
row_count = 0
for source, root_text in zip(sources, expected_paths, strict=True):
    root = pathlib.Path(root_text)
    for required in (data_root / source / "READY.json", data_root / source / "MANIFEST.json",
                     root / "map_style_cache/file_info.pkl"):
        if not required.is_file():
            raise SystemExit(f"missing immutable V10 data receipt: {required}")
    with (root / "map_style_cache/file_info.pkl").open("rb") as handle:
        files, lengths = pickle.load(handle)
    if not files or len(files) != len(lengths):
        raise SystemExit(f"invalid parquet cache under {root}")
    if not text_columns.issubset(set(pq.ParquetFile(files[0]).schema_arrow.names)):
        raise SystemExit(f"T2VA parquet is not a text-only schema superset: {files[0]}")
    for file_name, length in zip(files, lengths, strict=True):
        matches = [part for part in pathlib.Path(file_name).parts if part.startswith("bucket=")]
        if len(matches) != 1 or bucket_pattern.fullmatch(matches[0]) is None:
            raise SystemExit(f"invalid exact-shape path: {file_name}")
        bucket_counts[matches[0]] = bucket_counts.get(matches[0], 0) + int(length)
        row_count += int(length)
if row_count != 60_549 or len(bucket_counts) != 87:
    raise SystemExit(f"V10.5 corpus mismatch: rows={row_count}, buckets={len(bucket_counts)}")
padding = sum((-count) % 32 for count in bucket_counts.values())
if padding != 1_403 or row_count + padding != 61_952:
    raise SystemExit(f"world32 bucket schedule mismatch: padding={padding}, scheduled={row_count + padding}")

gate_method = gate["method"]
gate_training = gate["training"]
if gate_method.get("generator_update_interval") != 2:
    raise SystemExit("max-shape gate must exercise critic then student in two steps")
if gate_method.get("rollout_carry_slots") != 2 or gate_training["loop"].get("gradient_accumulation_steps") != 2:
    raise SystemExit("max-shape gate needs two synchronized carry streams for interval-2 coverage")
if gate_training["data"].get("num_width") != 1760 or gate_training["data"].get("num_frames") != 362:
    raise SystemExit("max-shape gate is not 1760x768x362")
sp4_distributed = sp4_prod["training"]["distributed"]
sp4_loop = sp4_prod["training"]["loop"]
if {key: int(sp4_distributed[key]) for key in expected_topology} != {
    "num_gpus": 32,
    "sp_size": 4,
    "tp_size": 1,
    "hsdp_replicate_dim": 1,
    "hsdp_shard_dim": 32,
}:
    raise SystemExit(f"V10.5 SP4 topology mismatch: {sp4_distributed}")
if sp4_prod["method"].get("rollout_carry_slots") != 8 or int(sp4_loop["gradient_accumulation_steps"]) != 8:
    raise SystemExit("V10.5 SP4 production requires carry_slots=accumulation=8")
sp4_global_batch = (32 // 4) * int(sp4_prod["training"]["data"]["train_batch_size"]) * int(
    sp4_loop["gradient_accumulation_steps"])
if sp4_global_batch != 64:
    raise SystemExit(f"V10.5 SP4 global batch is {sp4_global_batch}, expected 64")
if sp4_prod["training"]["data"]["data_path"] != expected_paths:
    raise SystemExit("V10.5 SP4 fallback changed the prompt population")
sp4_gate_training = sp4_gate["training"]
if sp4_gate["method"].get("rollout_carry_slots") != 2 or sp4_gate_training["loop"].get(
        "gradient_accumulation_steps") != 2:
    raise SystemExit("SP4 max-shape gate needs two synchronized carry streams")
if int(sp4_gate_training["distributed"].get("sp_size", 0)) != 4:
    raise SystemExit("SP4 max-shape gate does not enable SP=4")
print("READY: V10.5 data-free/native recipe, global batch 64, corpus, validation, compile, and gate contracts")
PY

require_exact_checkout

# Build an independently receipted prefix for the exact V10.5 source commit.
receipt_commit=""
if [[ -f "${KERNEL_PREFIX}/FASTVIDEO_KERNEL_V10_RECEIPT.json" ]]; then
  receipt_commit="$("${VENV}/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' \
    "${KERNEL_PREFIX}/FASTVIDEO_KERNEL_V10_RECEIPT.json")"
fi
if [[ "${receipt_commit}" != "${EXPECTED_V10_COMMIT}" ]]; then
  echo "=== building exact V10.5 kernel prefix ==="
  srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)" \
    env REPO="${REPO}" VENV="${VENV}" KERNEL_ROOT="${KERNEL_ROOT}" \
    UV_CACHE_DIR="${KERNEL_ROOT}/uv-cache" \
    UV=/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/tools/uv \
    bash "${REPO}/scripts/train/rebuild_h3_v10_kernel.sh"
fi

readonly SP4_OUTPUT_DIR="/mnt/lustre/vlm-wlsaidhi/fastvideo/outputs/minimax_h3_dmd2_sp4_fsdp32_v10p5_datafree_mixed_vsa64"

validate_gate_receipt() {
  local receipt_path="$1"
  local config_path="$2"
  local topology="$3"
  "${VENV}/bin/python" - "${receipt_path}" "${EXPECTED_V10_COMMIT}" "${config_path}" "${topology}" <<'PY'
import hashlib
import json
import pathlib
import sys
receipt_path = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
config_path = pathlib.Path(sys.argv[3])
topology = sys.argv[4]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected = {
    "schema_version": "fastvideo-h3-v10p5-maxshape-v1",
    "success": True,
    "execution_commit": commit,
    "config": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "topology": topology,
    "shape": {"width": 1760, "height": 768, "num_frames": 362},
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"stale V10.5 max-shape receipt {key}: {receipt.get(key)!r} != {value!r}")
checks = receipt.get("checks", {})
if not checks or not all(value is True for value in checks.values()):
    raise SystemExit(f"failed V10.5 max-shape checks in {receipt_path}: {checks}")
print(f"READY: reusing exact {topology} max-shape receipt {receipt_path}")
PY
}

run_capacity_gate() {
  local topology="$1"
  local config_path="$2"
  local gate_root="$3"
  local receipt_path="$4"
  local sp_size="$5"
  local run_kernel_tests="$6"

  if [[ -f "${receipt_path}" ]]; then
    validate_gate_receipt "${receipt_path}" "${config_path}" "${topology}"
    return 0
  fi
  if [[ -d "${gate_root}/output" && -n "$(find "${gate_root}/output" -mindepth 1 -print -quit)" ]]; then
    echo "V10.5 ${topology} GATE FAILED: unreceipted output is not fresh: ${gate_root}/output" >&2
    return 1
  fi

  echo "=== V10.5 ${topology} exact max-shape critic/student gate ==="
  export CONFIG="${config_path}"
  export H3_V10_TRAIN_LOG_ROOT="${gate_root}/train_logs"
  export SP_SIZE="${sp_size}" HSDP_REPLICATE=1 HSDP_SHARD=32
  export H3_V10_KERNEL_GATE="${run_kernel_tests}"
  export GATE_TEST="${run_kernel_tests}"
  mkdir -p "${gate_root}/train_logs"
  set +e
  bash "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
  local training_rc=$?
  set -e
  if (( training_rc != 0 )); then
    return "${training_rc}"
  fi

  "${VENV}/bin/python" - \
    "${gate_root}/train_logs/${SLURM_JOB_ID}-node0" \
    "${receipt_path}" "${EXPECTED_V10_COMMIT}" "${config_path}" "${topology}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
receipt_path = pathlib.Path(sys.argv[2])
commit = sys.argv[3]
config_path = pathlib.Path(sys.argv[4])
topology = sys.argv[5]
logs = sorted(log_dir.glob("*.log"))
text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in logs).replace("\r", "\n")
def last(metric):
    values = re.findall(rf"{re.escape(metric)}\s+([-+0-9.eE]+)", text)
    return float(values[-1]) if values else None
critic = last("grad_norm/critic")
student = last("grad_norm/student")
checks = {
    "training_completed": "Training completed" in text,
    "critic_grad_finite_positive": critic is not None and math.isfinite(critic) and critic > 0,
    "student_grad_finite_positive": student is not None and math.isfinite(student) and student > 0,
    "dense_teacher_and_critic_compiled": text.count("Enabled regional torch.compile for 52 submodules") >= 2,
    "fa4_selected": "Using FlashAttention-4 backend" in text,
    "vsa_student_selected": "attention backend resolved to VIDEO_SPARSE_ATTN_H3" in text,
    "vsa_training_backward_route": "inputs require grad and the sm_100a kernel is forward-only" in text,
}
if not all(checks.values()):
    raise SystemExit(f"V10.5 max-shape checks failed: {checks}")
receipt = {
    "schema_version": "fastvideo-h3-v10p5-maxshape-v1",
    "success": True,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "execution_commit": commit,
    "config": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "topology": topology,
    "shape": {"width": 1760, "height": 768, "num_frames": 362},
    "grad_norm": {"critic": critic, "student": student},
    "checks": checks,
    "logs": [str(path) for path in logs],
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"READY: {receipt_path}")
PY
}

selected_topology=""
if [[ "${START_TOPOLOGY}" == "sp4" ]]; then
  echo "READY: fresh-allocation SP4 recovery selected"
  if [[ -f "${SP4_MAXSHAPE_RECEIPT}" ]]; then
    validate_gate_receipt "${SP4_MAXSHAPE_RECEIPT}" "${SP4_MAXSHAPE_CONFIG}" "sp4"
  else
    if [[ -d "${SP4_OUTPUT_DIR}" && -n "$(find "${SP4_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
      echo "V10.5 GATE FAILED: SP4 production output exists before a capacity receipt: ${SP4_OUTPUT_DIR}" >&2
      exit 1
    fi
    run_capacity_gate "sp4" "${SP4_MAXSHAPE_CONFIG}" "${SP4_MAXSHAPE_ROOT}" \
      "${SP4_MAXSHAPE_RECEIPT}" 4 1
  fi
  selected_topology="sp4"
elif [[ -f "${MAXSHAPE_RECEIPT}" ]]; then
  validate_gate_receipt "${MAXSHAPE_RECEIPT}" "${MAXSHAPE_CONFIG}" "sp1"
  selected_topology="sp1"
elif [[ -f "${SP4_MAXSHAPE_RECEIPT}" ]]; then
  validate_gate_receipt "${SP4_MAXSHAPE_RECEIPT}" "${SP4_MAXSHAPE_CONFIG}" "sp4"
  selected_topology="sp4"
else
  if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "V10.5 GATE FAILED: SP1 production output exists before a capacity receipt: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  if [[ -d "${SP4_OUTPUT_DIR}" && -n "$(find "${SP4_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "V10.5 GATE FAILED: SP4 production output exists before a capacity receipt: ${SP4_OUTPUT_DIR}" >&2
    exit 1
  fi
  if run_capacity_gate "sp1" "${MAXSHAPE_CONFIG}" "${MAXSHAPE_ROOT}" "${MAXSHAPE_RECEIPT}" 1 1; then
    sp1_rc=0
  else
    sp1_rc=$?
  fi
  if (( sp1_rc == 0 )); then
    selected_topology="sp1"
  elif grep -RqsE "CUDA out of memory|OutOfMemoryError|out of memory" \
      "${MAXSHAPE_ROOT}/train_logs/${SLURM_JOB_ID}-node"*; then
    echo "V10.5 SP1 max-shape OOM confirmed; this CUDA allocation cannot be reused safely" >&2
    echo "Relaunch a clean allocation with V10P5_START_TOPOLOGY=sp4" >&2
    exit 75
  else
    echo "V10.5 SP1 gate failed without an OOM signature; refusing an automatic topology change" >&2
    exit "${sp1_rc}"
  fi
fi

require_exact_checkout
if [[ "${selected_topology}" == "sp4" ]]; then
  export CONFIG="${SP4_PROD_CONFIG}"
  export SP_SIZE=4 HSDP_REPLICATE=1 HSDP_SHARD=32
else
  export CONFIG="${PROD_CONFIG}"
  export SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=32
fi
export H3_V10_TRAIN_LOG_ROOT="/mnt/lustre/vlm-wlsaidhi/fastvideo/train_logs/v10p5"
export H3_V10_KERNEL_GATE=0
export GATE_TEST=0
mkdir -p "${H3_V10_TRAIN_LOG_ROOT}"

scontrol update JobId="${SLURM_JOB_ID}" Requeue=1
echo "READY: V10.5 gates passed; enabling requeue and starting ${selected_topology} production"
exec bash "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
