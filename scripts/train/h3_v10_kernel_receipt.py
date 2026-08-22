#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic content receipt for the installed H3 v10 kernel prefix."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


RECEIPT_FILENAME = "FASTVIDEO_KERNEL_V10_RECEIPT.json"
_CHUNK_SIZE = 8 * 1024 * 1024


def _is_receipted_file(prefix: Path, path: Path) -> bool:
    relative = path.relative_to(prefix)
    return (path.is_file() and relative.as_posix() != RECEIPT_FILENAME and "__pycache__" not in relative.parts
            and path.suffix != ".pyc")


def installed_prefix_tree_sha256(prefix: str | Path) -> str:
    """Hash every stable installed file as sorted ``path + size + contents``.

    The root receipt is excluded because it stores this digest. Python bytecode
    caches are also excluded because importing the installed wheel may create or
    rewrite them after publication without changing the installed artifact.
    """
    root = Path(prefix).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    digest = hashlib.sha256()
    paths = sorted(
        (path for path in root.rglob("*") if _is_receipted_file(root, path)),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative_bytes = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(b"F")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    print(installed_prefix_tree_sha256(args.prefix))


if __name__ == "__main__":
    main()
