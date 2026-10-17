# SPDX-License-Identifier: Apache-2.0
"""Stage the OmniRef NVFP4 QAD v2 release folders (stdlib only, CPU, never uploads).

Inputs: the packed transformer written by ``convert_minimax_h3_modelopt_nvfp4_dit.py`` (all 50 blocks NVFP4), the old
release folder (``FastH3-Omni-NVFP4-QAD`` layout) and the model card. Writes

  <out>/               full folder: old release components hard-linked, transformer + transformer_ref replaced
  <out>-transformer-only/   README + the replaced transformer and transformer_ref only

Both transformers are the same tensors (the old release also shipped identical ``transformer`` and
``transformer_ref``; QAD trains one student). Verifies, byte for byte, that every dense (non-packed) tensor of the
export (norms, modulation tables, embeddings, ...) equals the original pre-QAD transformer_ref, and that the packed
file records no ``skip_blocks``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

PACKED = "nvfp4_weights.safetensors"
SKIP_NAMES = {"README.md", ".cache", ".gitattributes", "transformer", "transformer_ref", "fastvideo_inference.json"}


def header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return 8 + size, json.loads(handle.read(size))


def dense_entries(directory: Path) -> dict[str, tuple[Path, int, dict]]:
    out = {}
    for shard in sorted(directory.glob("*.safetensors")):
        if shard.name == PACKED:
            continue
        base, meta = header(shard)
        out.update({k: (shard, base, v) for k, v in meta.items() if k != "__metadata__"})
    return out


def read(entry: tuple[Path, int, dict]) -> tuple[bytes, str, list]:
    path, base, meta = entry
    start, end = meta["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + start)
        return handle.read(end - start), meta["dtype"], meta["shape"]


def link_or_copy(src: Path, dst: Path) -> str:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def link_tree(src: Path, dst: Path, tally: dict[str, int]) -> None:
    real = src.resolve()
    if real.is_dir():
        for child in sorted(real.iterdir()):
            link_tree(child, dst / child.name, tally)
    else:
        mode = link_or_copy(real, dst)
        tally[mode] = tally.get(mode, 0) + 1


def verify(transformer: Path, original: Path) -> dict:
    meta = header(transformer / PACKED)[1].get("__metadata__", {}) or {}
    skip = json.loads(meta.get("skip_blocks", "[]"))
    packed = [k for k in header(transformer / PACKED)[1] if k != "__metadata__"]
    packed_blocks = {int(k.split(".")[1]) for k in packed if k.startswith("transformer_blocks.")}
    dense, orig = dense_entries(transformer), dense_entries(original)
    missing = [k for k in dense if k not in orig]
    differing = [k for k in dense if k in orig and read(dense[k]) != read(orig[k])]
    return {"skip_blocks": skip, "packed_tensors": len(packed), "packed_blocks": len(packed_blocks),
            "dense_tensors": len(dense), "dense_not_in_original": missing[:5],
            "dense_differing_from_original": differing[:5], "dense_differing_count": len(differing),
            "ok": skip == [] and len(packed_blocks) == 50 and not missing and not differing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transformer", required=True, type=Path, help="packed transformer dir from the converter")
    parser.add_argument("--old-release", required=True, type=Path, help="old FastH3-Omni-NVFP4-QAD folder")
    parser.add_argument("--original", required=True, type=Path, help="pre-QAD transformer_ref dir (dense bf16)")
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="full folder; <out>-transformer-only is derived")
    args = parser.parse_args()

    report = verify(args.transformer, args.original)
    print(json.dumps({"verify": report}), flush=True)
    if not report["ok"]:
        return 1
    only = args.out.with_name(args.out.name + "-transformer-only")
    tally: dict[str, int] = {}
    for entry in sorted(args.old_release.iterdir()):
        if entry.name not in SKIP_NAMES:
            link_tree(entry, args.out / entry.name, tally)
    contract = args.old_release / "fastvideo_inference.json"
    link_tree(contract, args.out / contract.name, tally)
    for root in (args.out, only):
        for name in ("transformer", "transformer_ref"):
            link_tree(args.transformer, root / name, tally)
        shutil.copyfile(args.card, root / "README.md")
    print(json.dumps({"staged": [str(args.out), str(only)], "files": tally}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
