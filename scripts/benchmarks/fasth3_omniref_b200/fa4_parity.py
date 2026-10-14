"""FA4 VSA-H3 fine attention vs the QAD FP4 emulator in the same numeric mode, on captured OmniRef layers.

One GPU. For every capture (``q``/``k``/``v`` ``[1, H, L, D]``, VSA tile ``mask``, ``vbs``) and head group, runs
``FA4VSAFineAttention(mode)`` and ``FP4VSAFineAttention`` with the matching numerics on identical inputs and the
same metadata, and reports rel-L2 over the valid query rows (all / dense-prefix / video). Gate: <= 0.01.

  python fa4_parity.py --glob '<captures>/*.pt' --modes mixmx qk bf16 --out parity.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from types import SimpleNamespace

import torch

EMULATOR = {
    "mix8": dict(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision="fp8"),
    "mixmx": dict(quantize=True, quantize_pv=False, smooth_k=True, prefix_precision="mxfp8"),
    "qk": dict(quantize=True, quantize_pv=False, smooth_k=True),
    "bf16": dict(quantize=False),
}


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))


def leading_dense_tiles(tile_mask: torch.Tensor) -> int:
    """Leading query tiles whose rows keep every key tile in every head (VSA-H3's dense prefix)."""
    dense = tile_mask.all(dim=-1).all(dim=0).tolist()
    return next((i for i, d in enumerate(dense) if not d), len(dense))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--modes", nargs="+", default=["mixmx", "qk", "bf16"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate", type=float, default=0.01)
    args = parser.parse_args()

    from fastvideo.attention.backends.fa4_vsa_h3 import FA4VSAFineAttention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics, FP4VSAFineAttention

    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for path in sorted(glob.glob(args.glob)):
        cap = torch.load(path, map_location="cpu")
        heads = min(args.heads, cap["q"].shape[1])
        q, k, v = (cap[n][:, :heads].cuda().transpose(1, 2).contiguous() for n in "qkv")  # [1, L, H, D]
        mask = cap["mask"][:, :heads].cuda()
        vbs = cap["vbs"].cuda()
        prefix = leading_dense_tiles(mask[0])
        meta = SimpleNamespace(tile_elems=128, variable_block_sizes=vbs, num_prefix_tiles=prefix)
        length = q.shape[1]
        valid = (torch.arange(length, device="cuda") % 128) < vbs.repeat_interleave(128)
        is_prefix = torch.arange(length, device="cuda") < prefix * 128
        groups = {"all": valid, "prefix": valid & is_prefix, "video": valid & ~is_prefix}
        row = {"capture": os.path.basename(path), "tokens": length, "prefix_tiles": prefix,
               "tiles": int(mask.shape[-1])}
        for mode in args.modes:
            emu = FP4VSAFineAttention(FP4AttentionNumerics(**EMULATOR[mode]))(q, k, v, mask, meta)
            fa4 = FA4VSAFineAttention(mode)(q, k, v, mask, meta)
            row[f"{mode}/finite"] = bool(torch.isfinite(fa4[:, valid]).all())
            for name, rows_ok in groups.items():
                if bool(rows_ok.any()):
                    row[f"{mode}/{name}"] = round(rel(fa4[:, rows_ok], emu[:, rows_ok]), 5)
        print("PARITY", json.dumps(row), flush=True)
        rows.append(row)
    worst = {mode: max(r[f"{mode}/all"] for r in rows) for mode in args.modes}
    mean = {mode: sum(r[f"{mode}/all"] for r in rows) / len(rows) for mode in args.modes}
    summary = {"captures": len(rows), "worst_all": worst, "mean_all": mean,
               "pass": {m: worst[m] <= args.gate and all(r[f"{m}/finite"] for r in rows) for m in args.modes}}
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=1)
    print("PARITY_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
