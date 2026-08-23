#!/bin/bash
# Run the two-step v10 capacity gate inside an existing 16-tray allocation.
# Persistent logs and receipts stay under the isolated gate root.

set -euo pipefail

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10}"
VENV="${VENV:-${REPO}/.venv}"
GATE_ROOT="${H3_V10_MAXSHAPE_ROOT:-/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/v10_maxshape_64g}"
CONFIG="${CONFIG:-${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml}"
TRAIN_LOG_ROOT="${H3_V10_TRAIN_LOG_ROOT:-${GATE_ROOT}/train_logs}"
GATE_DATA="${GATE_ROOT}/data"
GATE_OUTPUT="${GATE_ROOT}/output"
AUDIT_DIR="${GATE_ROOT}/audit/job-${SLURM_JOB_ID:?run inside Slurm}"
STOP_FILE="${AUDIT_DIR}/STOP"
SAMPLE_SECONDS="${H3_V10_MEMORY_SAMPLE_SECONDS:-1}"

expected_config="${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp64_v10_maxshape_gate_vsa64.yaml"
if [[ "${CONFIG}" != "${expected_config}" ]]; then
  echo "MAX-SHAPE GATE FAILED: config ${CONFIG} != ${expected_config}" >&2
  exit 1
fi
if [[ "${TRAIN_LOG_ROOT}" != "${GATE_ROOT}/train_logs" ]]; then
  echo "MAX-SHAPE GATE FAILED: persistent train logs must stay under ${GATE_ROOT}" >&2
  exit 1
fi
if [[ -e "${AUDIT_DIR}" ]]; then
  echo "MAX-SHAPE GATE FAILED: audit directory already exists: ${AUDIT_DIR}" >&2
  exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:-0}" != "16" ]]; then
  echo "MAX-SHAPE GATE FAILED: requires exactly 16 trays; got ${SLURM_JOB_NUM_NODES:-unset}" >&2
  exit 1
fi
if [[ -z "${EXPECTED_V10_COMMIT:-}" || "$(git -C "${REPO}" rev-parse HEAD)" != "${EXPECTED_V10_COMMIT}" ]]; then
  echo "MAX-SHAPE GATE FAILED: execution checkout is not pinned to EXPECTED_V10_COMMIT" >&2
  exit 1
fi
if [[ -d "${GATE_OUTPUT}" && -n "$(find "${GATE_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "MAX-SHAPE GATE FAILED: isolated output is not fresh: ${GATE_OUTPUT}" >&2
  exit 1
fi

bucket="${GATE_DATA}/bucket=1760x768-362f"
sources=(
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_10k_720_mixed_len/data/bucket=1760x768-362f/c00343.parquet
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data/bucket=1760x768-362f/c00936.parquet
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data/bucket=1760x768-362f/c00937.parquet
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data/bucket=1760x768-362f/c00938.parquet
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data/bucket=1760x768-362f/c00939.parquet
  /mnt/lustre/vlm-shared/h3_t2av_preprocessed/v10_mixed_native_v3/h3_t2av_video_nuva_50k_720_mixed_len/data/bucket=1760x768-362f/c00940.parquet
)
for source in "${sources[@]}"; do
  staged="${bucket}/$(basename "${source}")"
  if [[ ! -f "${staged}" || ! "${staged}" -ef "${source}" ]]; then
    echo "MAX-SHAPE GATE FAILED: ${staged} is not a hardlink to ${source}" >&2
    exit 1
  fi
done
if [[ "$(find "${GATE_DATA}" -type f -name '*.parquet' | wc -l)" != "${#sources[@]}" ]]; then
  echo "MAX-SHAPE GATE FAILED: isolated data must contain exactly ${#sources[@]} parquets" >&2
  exit 1
fi

mkdir -p "${AUDIT_DIR}" "${TRAIN_LOG_ROOT}"
export REPO VENV CONFIG AUDIT_DIR STOP_FILE SAMPLE_SECONDS
export H3_V10_TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT}"

# Sample every GPU on every tray. This is an external observed peak, so the
# receipt names it accordingly rather than claiming allocator-perfect timing.
srun --overlap --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    node="$(hostname -s)"
    output="${AUDIT_DIR}/gpu-memory-node${SLURM_NODEID}-${node}.csv"
    printf "%s\n" "timestamp_unix,node,gpu_index,gpu_uuid,memory_used_mib,memory_total_mib,utilization_gpu_pct" > "${output}"
    while [[ ! -e "${STOP_FILE}" ]]; do
      stamp="$(date +%s.%N)"
      nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
        while IFS= read -r row; do
          printf "%s,%s,%s\n" "${stamp}" "${node}" "${row}" >> "${output}"
        done
      sleep "${SAMPLE_SECONDS}"
    done
  ' &
monitor_pid=$!

set +e
bash "${REPO}/examples/train/slurm/dmd2_32xgb200.sbatch"
training_rc=$?
set -e

touch "${STOP_FILE}"
set +e
wait "${monitor_pid}"
monitor_rc=$?
set -e

# Produce one machine-readable receipt even when training failed. The audit
# requires both finite, positive grad norms, exact two-step completion, both
# dense compile receipts, FA4 selection, and the expected VSA eager/Triton
# backward receipt.
set +e
"${VENV}/bin/python" - \
  "${AUDIT_DIR}" "${TRAIN_LOG_ROOT}" "${SLURM_JOB_ID}" "${training_rc}" "${monitor_rc}" \
  "${REPO}" "${CONFIG}" "${EXPECTED_V10_COMMIT}" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

audit_dir = pathlib.Path(sys.argv[1])
train_log_root = pathlib.Path(sys.argv[2])
job_id = sys.argv[3]
training_rc = int(sys.argv[4])
monitor_rc = int(sys.argv[5])
repo = pathlib.Path(sys.argv[6])
config = pathlib.Path(sys.argv[7])
expected_execution_commit = sys.argv[8]

ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
log_paths = sorted((train_log_root / f"{job_id}-node0").glob("*.log"))
text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in log_paths)
text = ansi.sub("", text).replace("\r", "\n")


def last_number(metric: str):
    matches = re.findall(rf"{re.escape(metric)}\s+([-+0-9.eE]+)", text)
    return float(matches[-1]) if matches else None


critic_grad = last_number("grad_norm/critic")
student_grad = last_number("grad_norm/student")
fake_score_loss = last_number("fake_score_loss")
generator_loss = last_number("generator_loss")

peaks = {}
samples = 0
for path in sorted(audit_dir.glob("gpu-memory-node*.csv")):
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['node'].strip()}/{row['gpu_uuid'].strip()}"
            used = int(row["memory_used_mib"].strip())
            total = int(row["memory_total_mib"].strip())
            previous = peaks.get(key)
            if previous is None or used > previous["used_mib"]:
                peaks[key] = {"used_mib": used, "total_mib": total}
            samples += 1

compile_receipts = text.count("Enabled regional torch.compile for 52 submodules")
fa4_receipts = text.count("Using FlashAttention-4 backend")
vsa_eager_receipts = text.count("attention backend resolved to VIDEO_SPARSE_ATTN_H3")
vsa_triton_backward_receipts = text.count("inputs require grad and the sm_100a kernel is forward-only")
observed_execution_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
execution_checkout_clean = not subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()


def finite_positive(value):
    return value is not None and math.isfinite(value) and value > 0.0


def finite(value):
    return value is not None and math.isfinite(value)


checks = {
    "training_exit_zero": training_rc == 0,
    "memory_monitor_exit_zero": monitor_rc == 0,
    "training_completed": "Training completed" in text,
    "critic_grad_finite_positive": finite_positive(critic_grad),
    "student_grad_finite_positive": finite_positive(student_grad),
    "fake_score_loss_finite": finite(fake_score_loss),
    "generator_loss_finite": finite(generator_loss),
    "dense_teacher_and_critic_compiled": compile_receipts >= 2,
    "fa4_selected": fa4_receipts > 0,
    "vsa_student_kept_eager": vsa_eager_receipts > 0,
    "vsa_grad_used_triton64": vsa_triton_backward_receipts > 0,
    "all_64_gpus_sampled": len(peaks) == 64,
    "memory_samples_present": samples > 0,
    "execution_commit_unchanged": observed_execution_commit == expected_execution_commit,
    "execution_checkout_clean": execution_checkout_clean,
}
receipt = {
    "schema_version": "fastvideo-h3-v10-maxshape-gate-v1",
    "success": all(checks.values()),
    "checks": checks,
    "job_id": job_id,
    "execution_commit": expected_execution_commit,
    "observed_execution_commit_at_receipt": observed_execution_commit,
    "config": str(config),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "shape": {"width": 1760, "height": 768, "num_frames": 362, "video_latent_shape": [24, 107, 48, 110]},
    "steps": {"critic": 1, "student": 2},
    "metrics": {
        "grad_norm/critic": critic_grad,
        "grad_norm/student": student_grad,
        "fake_score_loss": fake_score_loss,
        "generator_loss": generator_loss,
    },
    "route_receipts": {
        "regional_compile_52_count": compile_receipts,
        "fa4_count": fa4_receipts,
        "vsa_eager_count": vsa_eager_receipts,
        "vsa_triton_backward_count": vsa_triton_backward_receipts,
    },
    "observed_gpu_memory": {
        "sample_count": samples,
        "max_used_mib": max((entry["used_mib"] for entry in peaks.values()), default=None),
        "per_gpu": peaks,
        "note": "one-second external nvidia-smi samples; observed peak, not allocator-perfect peak",
    },
    "node0_logs": [str(path) for path in log_paths],
}
receipt_path = audit_dir / "RESULT.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(0 if receipt["success"] else 1)
PY
audit_rc=$?
set -e

if (( training_rc != 0 )); then
  exit "${training_rc}"
fi
exit "${audit_rc}"
