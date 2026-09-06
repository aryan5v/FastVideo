#!/usr/bin/env bash
# Initializer search only: deliberately contains no recovery/training command.
#SBATCH --job-name=h3-conservative-step0
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -euo pipefail
export SPRINT_ROOT="${SPRINT_ROOT:-/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829}"
export CANDIDATE_DEPTHS="${CANDIDATE_DEPTHS:-40 36 32}"
REPO_ROOT="${SPRINT_ROOT}/repo"
export RUN_TAG="conservative-${SLURM_JOB_ID}"
export OUTPUT_ROOT="${SPRINT_ROOT}/videos/h36-conservative-step0/${RUN_TAG}"
mkdir -p "${OUTPUT_ROOT}" "${SPRINT_ROOT}/manifests/h6-block-score"

# Uniform depth coverage is a control for pruning severity. No single-block
# importance score is used as a claim that simultaneous removal is harmless.
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SPRINT_ROOT"]) / "manifests/h6-block-score"
depths = [int(value) for value in os.environ["CANDIDATE_DEPTHS"].split()]
assert depths and len(depths) == len(set(depths))
assert all(count in (32, 36, 40, 48) for count in depths)
for count in depths:
    block_map = [round(i * 49 / (count - 1)) for i in range(count)]
    assert len(set(block_map)) == count
    assert block_map[0] == 0 and block_map[-1] == 49
    assert max(b - a for a, b in zip(block_map, block_map[1:])) <= 2
    payload = {"strategy": "uniform_depth_control", "source_num_layers": 50, "block_map": block_map}
    path = root / f"uniform_{count}_block_map.json"
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"refusing to overwrite {path}"
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload), flush=True)
PY

for keep_blocks in ${CANDIDATE_DEPTHS}; do
  for source_kind in dense vsa; do
    model="${SPRINT_ROOT}/checkpoints/h18-candidates/${source_kind}-uniform-${keep_blocks}b"
    if [[ -e "${model}" ]]; then
      echo "refusing to reuse an existing candidate without a separate audit: ${model}" >&2
      exit 2
    fi
    echo "$(date -Is) constructing ${source_kind} ${keep_blocks} blocks"
    SOURCE_KIND="${source_kind}" MAP_STRATEGY=uniform KEEP_BLOCKS="${keep_blocks}" \
      bash "${REPO_ROOT}/scripts/fasth3_sprint/slurm_prune_candidate.sbatch"
  done
  KEEP_BLOCKS="${keep_blocks}" MAP_STRATEGIES=uniform \
    bash "${REPO_ROOT}/scripts/fasth3_sprint/slurm_h3_24b_step0_diagnostics.sbatch"
done
date -Is > "${OUTPUT_ROOT}/all_candidates_completed_at.txt"
