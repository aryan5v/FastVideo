#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Weight-interpolate two same-architecture H3 transformer exports (model soup).

7048 and 7101 share lineage and architecture (7101 was initialized from 7048),
so interpolating their weights is a valid no-training quality knob: 7101 carries
the higher-contrast/higher-motion direction, 7048 the better audio brightness.
Writes one diffusers-style transformer directory per alpha.

Usage::

    python scripts/fasth3_sprint/soup_h3_exports.py \
        --left export-200/transformer --right export-100/transformer \
        --alphas 0.3,0.5,0.7 --out-dir /path/to/soups
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--alphas", required=True, help="comma-separated weight of RIGHT in [0,1]")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    left = load_file(str(args.left / "model.safetensors"))
    right = load_file(str(args.right / "model.safetensors"))
    if left.keys() != right.keys():
        raise SystemExit("soup endpoints have different parameter sets")
    for alpha in [float(value) for value in args.alphas.split(",")]:
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit(f"alpha out of range: {alpha}")
        dest = args.out_dir / f"alpha-{alpha:.2f}" / "transformer"
        dest.mkdir(parents=True, exist_ok=True)
        mixed = {
            key: ((1.0 - alpha) * left[key].float() + alpha * right[key].float()).to(left[key].dtype)
            for key in left
        }
        save_file(mixed, str(dest / "model.safetensors"))
        for extra in args.left.glob("*.json"):
            shutil.copy2(extra, dest / extra.name)
        print(json.dumps({"alpha": alpha, "dest": str(dest), "tensors": len(mixed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
