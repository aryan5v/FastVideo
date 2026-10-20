#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export or check FastVideo's OpenAI-compatible HTTP schema.

This reads the mounted FastAPI routes; it does not start the server, load
weights, initialize CUDA/MLX, or contact a model host. Run from any directory:

    python scripts/export_openapi.py
    python scripts/export_openapi.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastvideo.entrypoints.openai.openapi_export import SPEC_PATH, check_openapi, write_openapi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in spec is stale instead of rewriting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SPEC_PATH,
        help=f"Destination path (default: {SPEC_PATH.relative_to(ROOT)}).",
    )
    args = parser.parse_args()

    if args.check:
        if not check_openapi(args.output):
            print(
                f"OpenAPI spec is stale or missing: {args.output}\n"
                "Run `python scripts/export_openapi.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"OpenAPI spec is current: {args.output}")
        return 0

    write_openapi(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
