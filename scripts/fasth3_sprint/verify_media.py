#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode and verify a FastH3 joint video/audio artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.fasth3_sprint.run_baseline_matrix import _probe_media


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    probe = _probe_media(args.media)
    payload = json.dumps(probe, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
