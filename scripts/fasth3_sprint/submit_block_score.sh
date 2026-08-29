#!/usr/bin/env bash
# Submit four independent 4-GPU scoring jobs and one aggregate dependency.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/slurm"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_DIR}"
submit_dependency=()
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  submit_dependency+=(--dependency="afterok:${AFTEROK_JOB_ID}" --kill-on-invalid-dep=yes)
fi
jobs=()
for job_index in 0 1 2 3; do
  job_id="$(sbatch --parsable --export=NIL \
    "${submit_dependency[@]}" \
    --partition=all --nodes=1 --ntasks=1 --gres=gpu:4 --time=06:00:00 \
    --job-name="h3-score-${job_index}" \
    --output="${LOG_DIR}/h3-score-${job_index}-%j.out" \
    --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' JOB_INDEX='${job_index}' NUM_JOBS=4 SOURCE_COMMIT='${SOURCE_COMMIT}' /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_block_score.sbatch'")"
  jobs+=("${job_id}")
done

dependency="$(IFS=:; printf '%s' "${jobs[*]}")"
aggregate_job="$(sbatch --parsable --export=NIL \
  --dependency="afterok:${dependency}" --kill-on-invalid-dep=yes \
  --partition=all --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:30:00 \
  --job-name=h3-score-aggregate \
  --output="${LOG_DIR}/h3-score-aggregate-%j.out" \
  --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_block_score_aggregate.sbatch'")"

printf 'score_jobs=%s\naggregate_job=%s\n' "${jobs[*]}" "${aggregate_job}"
