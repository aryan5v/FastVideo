# SPDX-License-Identifier: Apache-2.0
"""Write a QAD step-0 cache (``method.step0_cache``) from an earlier run's log.

For runs that predate the cache: their step-0 metrics are in the rank-0 ``QAD eval @0`` log line (rounded to five
decimals, < 0.1% relative for every normalizer). The fingerprint is computed from that run's ``cfg.yaml``; a rerun
with the same init checkpoint, calibration table, quantization plan, eval set and tolerances will accept it.

    python step0_cache_from_log.py --cfg runs/x/cfg.yaml --log logs/x.log --out runs/y/step0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

from fastvideo.train.methods.knowledge_distillation.pdd_qad_step0 import save_step0, step0_fingerprint_inputs


def step0_metrics(log_text: str) -> dict[str, float]:
    lines = [line for line in log_text.splitlines() if "QAD eval @0:" in line]
    if len(lines) != 1:
        raise ValueError(f"expected one 'QAD eval @0' line, found {len(lines)}")
    return json.loads(lines[0][lines[0].index("{"):])


def num_eval_rows(log_text: str) -> int:
    found = set(re.findall(r"OmniRef QAD rows: .* (\d+) held-out eval", log_text))
    if len(found) != 1:
        raise ValueError(f"cannot determine the eval row count from the log: {sorted(found)}")
    return int(found.pop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cfg", required=True, help="the earlier run's cfg.yaml")
    parser.add_argument("--log", required=True, help="the earlier run's training log")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    log_text = Path(args.log).read_text(errors="replace")
    student = cfg["models"]["student"]
    amax = student["qad"]["linears"]["amax_json"]
    sha = student["qad"]["linears"].get("amax_sha256") or hashlib.sha256(Path(amax).read_bytes()).hexdigest()
    inputs = step0_fingerprint_inputs(cfg["method"], student["data"], student["qad"], sha, num_eval_rows(log_text))
    metrics = step0_metrics(log_text)
    save_step0(args.out, inputs, metrics)
    print(json.dumps({"out": args.out, "metrics": len(metrics), "num_eval_rows": inputs["num_eval_rows"]}))


if __name__ == "__main__":
    main()
