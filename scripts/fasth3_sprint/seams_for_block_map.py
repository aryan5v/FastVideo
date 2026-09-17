#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive seam (post-gap) local block indices from an H3 block map.

The tokenwise seam loss supervises the first retained block after every run of
deleted source blocks: that block must now absorb the deleted interval's work.
Given a block map (local index -> source index) this prints the local indices
whose predecessor in the map is non-contiguous.

Usage::

    python scripts/fasth3_sprint/seams_for_block_map.py --block-map map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def seams_for_block_map(block_map: list[int]) -> list[int]:
    seams = []
    for local, source in enumerate(block_map):
        if local == 0:
            continue
        if source != block_map[local - 1] + 1:
            seams.append(local)
    return seams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-map", type=Path, required=True,
                        help="JSON list, or object with a block_map field")
    args = parser.parse_args()
    payload = json.loads(args.block_map.read_text())
    block_map = payload["block_map"] if isinstance(payload, dict) else payload
    print(json.dumps(seams_for_block_map(block_map)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
