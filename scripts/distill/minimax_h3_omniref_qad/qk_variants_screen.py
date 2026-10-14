# SPDX-License-Identifier: Apache-2.0
"""Screen the FP4 attention variants of ``eval_attention_variants`` on captured OmniRef attention inputs (one GPU).

Each capture (``fidelity_check.py capture``) holds one layer's tile-ordered Q/K/V for a few heads plus the VSA-128
tile mask and tile sizes at the deployed settings. Per capture and head this computes the exact fp32 sparse attention
and every variant's per-call rel-L2 against it, split by query-row type: dense-prefix rows (the leading tiles whose
mask rows are all kept: text, reference and audio tokens) vs video rows. It also reports the share of listed key
blocks (kernel FLOPs) that the dense-prefix query tiles account for.

    python qk_variants_screen.py --glob 'captures/82_*.pt' --out screen-82.json [--variants qk qk_had ...]
"""
from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_KERNELS = _ROOT / "fastvideo-kernel" / "python" / "fastvideo_kernel" / "triton_kernels"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_attention_variants import VARIANTS  # noqa: E402


def _kernels():
    spec = importlib.util.spec_from_loader("_qk_screen_kernels", loader=None, is_package=True)
    spec.submodule_search_locations.append(str(_KERNELS))
    sys.modules["_qk_screen_kernels"] = importlib.util.module_from_spec(spec)
    return (importlib.import_module("_qk_screen_kernels.attn_qat_vsa_reference"),
            importlib.import_module("_qk_screen_kernels.attn_qat_vsa_train"))


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))


def exact_reference(q, k, v, tile_mask, vbs, chunk: int = 16) -> torch.Tensor:
    """fp32 softmax(q k^T / sqrt(D)) v over the tile mask, pad keys excluded; ``[L, D]``."""
    length, dim = q.shape
    n_tiles = length // 128
    key_ok = (torch.arange(length, device=q.device) % 128) < vbs.repeat_interleave(128)
    out = torch.empty(length, dim, device=q.device)
    qf, kf, vf = q.float(), k.float(), v.float()
    for t0 in range(0, n_tiles, chunk):
        t1 = min(n_tiles, t0 + chunk)
        s = (qf[t0 * 128:t1 * 128] @ kf.T) / math.sqrt(dim)
        ok = tile_mask[t0:t1].repeat_interleave(128, 0).repeat_interleave(128, 1) & key_ok[None]
        out[t0 * 128:t1 * 128] = torch.softmax(s.masked_fill(~ok, float("-inf")), -1) @ vf
    return out


def leading_dense_tiles(tile_mask: torch.Tensor) -> int:
    """Number of leading query tiles whose mask rows keep every key tile (the dense prefix)."""
    dense = tile_mask.all(dim=-1).tolist()
    return next((i for i, d in enumerate(dense) if not d), len(dense))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["bf16"] + list(VARIANTS))
    args = parser.parse_args()
    ref, train = _kernels()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for path in sorted(glob.glob(args.glob)):
        capture = torch.load(path, map_location="cpu")
        vbs = capture["vbs"].cuda()
        for head in range(min(args.heads, capture["q"].shape[1])):
            q, k, v = (capture[n][0, head].cuda() for n in "qkv")
            tile_mask = capture["mask"][0, head].cuda()
            n_tiles = tile_mask.shape[0]
            prefix = leading_dense_tiles(tile_mask)
            valid = (torch.arange(q.shape[0], device="cuda") % 128) < vbs.repeat_interleave(128)
            is_prefix = torch.arange(q.shape[0], device="cuda") < prefix * 128
            groups = {"all": valid, "prefix": valid & is_prefix, "video": valid & ~is_prefix}
            exact = exact_reference(q, k, v, tile_mask, vbs)
            q2k_idx, q2k_num, kv_valid = (t.cuda() for t in ref.tile128_mask_to_fp4_blocks(tile_mask[None, None], vbs))
            blocks = q2k_num[0, 0].float()
            row = {
                "capture": os.path.basename(path),
                "head": head,
                "tiles": n_tiles,
                "prefix_tiles": prefix,
                "prefix_tile_share": prefix / n_tiles,
                "prefix_block_share": float(blocks[:prefix].sum() / blocks.sum()),
            }
            for name in args.variants:
                options = {"quantize": False} if name == "bf16" else {"quantize": True, **VARIANTS[name]}
                out = train.fp4_vsa_attn_qat(q[None, None], k[None, None], v[None, None], q2k_idx, q2k_num, kv_valid,
                                             num_prefix_tiles=prefix, **options)[0, 0]
                for group, rows_ok in groups.items():
                    if bool(rows_ok.any()):
                        row[f"{name}/{group}"] = rel(out[rows_ok], exact[rows_ok])
            print(json.dumps({k2: round(v2, 5) if isinstance(v2, float) else v2 for k2, v2 in row.items()}), flush=True)
            rows.append(row)
        del capture
    keys = sorted({key for row in rows for key in row if "/" in key or key.endswith("share")})
    summary = {key: sum(r[key] for r in rows if key in r) / max(1, sum(key in r for r in rows)) for key in keys}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("SUMMARY " + json.dumps({k2: round(v2, 5) for k2, v2 in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
