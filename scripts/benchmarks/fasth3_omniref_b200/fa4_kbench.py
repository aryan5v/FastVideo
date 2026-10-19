"""Fine-attention timing on one GPU: today's sm_100a BF16 VSA-128 kernel vs the FA4 mixed path, on real VSA lists.

Inputs are capture files (``mask [1, h, T, T]``, ``vbs``, ``q`` for the length; ``lists-*.pt`` from the pipeline
also work). The mask is tiled to ``--heads`` heads (14 = one SP4 rank, 56 = the whole model); Q/K/V are random
BF16. FA4 times include everything the override does per call (K smoothing, FlashInfer quantization, Q256 list
construction, the kernels and the output merge); ``quant``/``lists`` time those parts alone.

  python fa4_kbench.py --inputs cap1.pt cap2.pt --heads 14 56
"""
from __future__ import annotations

import argparse
import json
import math

import torch


def bench(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return round(times[len(times) // 2], 3)


def leading_dense_tiles(tile_mask: torch.Tensor) -> int:
    dense = tile_mask.all(dim=-1).all(dim=0).tolist()
    return next((i for i, d in enumerate(dense) if not d), len(dense))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--heads", nargs="+", type=int, default=[14, 56])
    parser.add_argument("--modes", nargs="+", default=["mixmx", "qk"])
    args = parser.parse_args()

    from fastvideo.attention.backends import fa4_vsa_h3 as fa4
    from fastvideo.attention.backends import video_sparse_attn_h3 as vsa

    for path in args.inputs:
        data = torch.load(path, map_location="cpu")
        vbs = data["vbs"].cuda().to(torch.int32)
        base = data["mask"][0].cuda()
        if "heads_on_rank" in data:  # pipeline-saved lists: column 0 was clobbered by the scatter padding
            base[..., 0] = True
        n_tiles = base.shape[-1]
        prefix = leading_dense_tiles(base)
        for heads in args.heads:
            mask = base.repeat(math.ceil(heads / base.shape[0]), 1, 1)[:heads][None]
            torch.manual_seed(0)
            length = n_tiles * 128
            q, k, v = (torch.randn(1, length, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            listed = (mask & (vbs > 0)[None, None, None]).sum(-1).float()
            row = {"input": path.split("/")[-1], "tokens": int(vbs.sum()), "tiles": n_tiles, "heads": heads,
                   "prefix_tiles": prefix, "listed_frac": round(float(listed.sum() / (heads * n_tiles * n_tiles)), 4),
                   "prefix_block_share": round(float(listed[..., :prefix].sum() / listed.sum()), 4)}
            # sm_100a needs an even tile count (the pipeline pads one partner tile the same way).
            m2, vbs2, qb, kb, vb = mask, vbs, q, k, v
            if n_tiles % 2:
                m2 = torch.nn.functional.pad(mask, (0, 1, 0, 1), value=False)
                vbs2 = torch.nn.functional.pad(vbs, (0, 1), value=0)
                qb, kb, vb = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 128)) for t in (q, k, v))
            qb, kb, vb = (t.transpose(1, 2).contiguous() for t in (qb, kb, vb))
            idx, num = vsa.map_to_index(m2)
            row["sm100a_bf16_ms"] = bench(
                lambda: vsa._sm100a.block_sparse_attn_sm100a(qb, kb, vb, idx, num, vbs2, need_lse=False))
            for mode in args.modes:
                row[f"fa4_{mode}_ms"] = bench(lambda: fa4.fa4_vsa_attention(q, k, v, mask, vbs, prefix, mode=mode))
                row[f"fa4_{mode}_speedup"] = round(row["sm100a_bf16_ms"] / row[f"fa4_{mode}_ms"], 3)
            row["quant_nvfp4_qk_ms"] = bench(lambda: (fa4._quantize(q, "nvfp4"), fa4._quantize(k, "nvfp4")))
            row["quant_mxfp8_qk_ms"] = bench(lambda: (fa4._quantize(q, "mxfp8"), fa4._quantize(k, "mxfp8")))
            limit = torch.where(mask, vbs[None, None, None, :], 0).to(torch.int32)
            row["lists_ms"] = bench(lambda: fa4._pair_lists(limit[:, :, :n_tiles - n_tiles % 2]))
            print("KBENCH", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
