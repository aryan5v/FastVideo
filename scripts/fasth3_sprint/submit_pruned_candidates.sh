#!/usr/bin/env bash
# Submit the four Dense/VSA × activation/uniform 20-block candidates.
set -euo pipefail

: "${SPRINT_ROOT:?SPRINT_ROOT is required}"
: "${AFTEROK_JOB_ID:?AFTEROK_JOB_ID is required}"

REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}"
jobs=()
for source_kind in dense vsa; do
  for map_strategy in activation uniform; do
    job_id="$(sbatch --parsable --export=NIL \
      --dependency="afterok:${AFTEROK_JOB_ID}" --kill-on-invalid-dep=yes \
      --partition=all --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 \
      --job-name="h3-prune-${source_kind}-${map_strategy}" \
      --output="${LOG_DIR}/h3-prune-${source_kind}-${map_strategy}-%j.out" \
      --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' SOURCE_KIND='${source_kind}' MAP_STRATEGY='${map_strategy}' /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_prune_candidate.sbatch'")"
    jobs+=("${job_id}")
  done
done
printf 'candidate_jobs=%s\n' "${jobs[*]}"

