#!/usr/bin/env bash
# Queue 24-block maps, candidates, and locked step-zero diagnostics.
set -euo pipefail

SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
REPO_ROOT="${SPRINT_ROOT}/repo"
LOG_DIR="${SPRINT_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}"

aggregate_job="$(sbatch --parsable --export=NIL \
  --partition=all --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:30:00 \
  --output="${LOG_DIR}/h3-map-24b-%j.out" \
  --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' KEEP_BLOCKS=24 /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_block_score_aggregate.sbatch'")"

candidate_jobs=()
for source_kind in dense vsa; do
  for map_strategy in activation uniform; do
    job="$(sbatch --parsable --export=NIL \
      --dependency="afterok:${aggregate_job}" --kill-on-invalid-dep=yes \
      --partition=all --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 \
      --output="${LOG_DIR}/h3-prune-${source_kind}-${map_strategy}-24b-%j.out" \
      --wrap="/usr/bin/env SPRINT_ROOT='${SPRINT_ROOT}' SOURCE_KIND='${source_kind}' MAP_STRATEGY='${map_strategy}' KEEP_BLOCKS=24 /bin/bash '${REPO_ROOT}/scripts/fasth3_sprint/slurm_prune_candidate.sbatch'")"
    candidate_jobs+=("${job}")
  done
done

dependency="$(IFS=:; echo "${candidate_jobs[*]}")"
diagnostic_job="$(sbatch --parsable --export=NIL \
  --dependency="afterok:${dependency}" --kill-on-invalid-dep=yes \
  --output="${LOG_DIR}/h3-24b-step0-diag-%j.out" \
  "${REPO_ROOT}/scripts/fasth3_sprint/slurm_h3_24b_step0_diagnostics.sbatch")"

printf 'aggregate_job=%s\n' "${aggregate_job}"
printf 'candidate_jobs=%s\n' "${candidate_jobs[*]}"
printf 'diagnostic_job=%s\n' "${diagnostic_job}"
