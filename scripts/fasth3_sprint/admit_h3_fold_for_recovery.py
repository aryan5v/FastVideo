#!/usr/bin/env python3
"""Admit a numerically valid H3 fold to recovery, never directly to release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lane", choices=("A", "B", "C"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--deployment-transformer-calls", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()

    factor_path = args.candidate / "factorization_receipt.json"
    parity_path = args.candidate / "production_media_parity.json"
    config_path = args.candidate / "training-config.yaml"
    factor = json.loads(factor_path.read_text())
    parity = json.loads(parity_path.read_text())

    assert config_path.is_file(), config_path
    assert factor["basis_relative_residual"] < 1e-6, factor
    assert factor["modulation_relative_max_error"] < 1e-5, factor
    billion = factor["parameters_after"] / 1e9
    if args.lane == "A":
        assert 19.0 <= billion <= 21.5, factor
    else:
        assert 13.5 <= billion <= 14.5, factor

    payload = {
        "schema_version": 1,
        "status": "recovery_only",
        "production_release_approved": False,
        "lane": args.lane,
        "role": args.role,
        "source": args.source,
        "parameters_after": factor["parameters_after"],
        "rank": factor["rank"],
        "deployment_transformer_calls": args.deployment_transformer_calls,
        "factorization_gate_passed": True,
        "production_media_parity_passed": bool(parity["passed"]),
        "production_media_parity": parity,
        "admission_reason": (
            "The fold is algebraically valid but is not production-parity. "
            "It is admitted only as the initialization for gated recovery training."
        ),
        "source_commit": args.source_commit,
        "slurm_job_id": args.slurm_job_id,
    }
    output = args.candidate / "RECOVERY_ADMISSION.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
