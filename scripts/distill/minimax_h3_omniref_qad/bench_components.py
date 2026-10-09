# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark of the QAD hot paths at FastH3 480p shapes on one GPU (no model load).

NVFP4 QAD linear (first call vs steady state, per backend) and the sparse FP4 attention emulator
(fake quant, forward, backward) versus the BF16 sm_100a VSA-128 kernel, per-rank heads at SP 4.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def _time(fn, repeat: int = 3) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return first, (time.perf_counter() - start) / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=15360)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--prefix-tiles", type=int, default=8)
    args = parser.parse_args()
    report: dict[str, object] = {}

    from fastvideo.layers.quantization import nvfp4_qad
    from fastvideo.layers.quantization.nvfp4_config import _mm_fp4_backend
    rows = args.tokens // 4
    for name, (n, k) in {"qkv": (5376, 5376), "fc_in": (28672, 5376), "fc_out": (5376, 14336)}.items():
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
        sf = torch.tensor(1.0, device="cuda")
        first, steady = _time(lambda w=w, x=x: nvfp4_qad.nvfp4_gemm(x, w, None, sf))
        x2 = torch.randn(rows + 37, k, device="cuda").to(torch.bfloat16)  # a new M, as a new row brings
        first_new_m, _ = _time(lambda w=w, x2=x2: nvfp4_qad.nvfp4_gemm(x2, w, None, sf), repeat=1)
        _, dense = _time(lambda w=w, x=x: x @ w.t())
        report[f"gemm/{name}"] = dict(first=round(first, 3), steady_ms=round(steady * 1e3, 2),
                                      new_m_first=round(first_new_m, 3), bf16_ms=round(dense * 1e3, 2))
    report["mm_backend"] = _mm_fp4_backend()

    from fastvideo.attention.backends.fp4_vsa_qat import load_qat_vsa_modules
    reference, train = load_qat_vsa_modules()
    n_tiles = args.tokens // 128
    q, k, v = (torch.randn(1, args.heads, args.tokens, 128, device="cuda").to(torch.bfloat16) for _ in range(3))
    mask = torch.rand(1, args.heads, n_tiles, n_tiles, device="cuda") < args.density
    mask[..., :args.prefix_tiles] = True
    mask[:, :, :args.prefix_tiles] = True
    vbs = torch.full((n_tiles, ), 128, device="cuda", dtype=torch.int32)
    lists = reference.tile128_mask_to_fp4_blocks(mask, vbs)
    report["attn/blocks_listed_frac"] = round(float(mask.float().mean()), 3)
    report["attn/lists_ms"] = round(_time(lambda: reference.tile128_mask_to_fp4_blocks(mask, vbs))[1] * 1e3, 2)
    report["attn/fake_quant_ms"] = round(_time(lambda: reference.fake_quant_qkv(q, k, v))[1] * 1e3, 2)
    first, steady = _time(lambda: train.fp4_vsa_attn_qat(q, k, v, *lists))
    report["attn/emulator_fwd"] = dict(first=round(first, 2), steady_ms=round(steady * 1e3, 2))
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))

    def fwd_bwd():
        out = train.fp4_vsa_attn_qat(qg, kg, vg, *lists)
        out.backward(torch.ones_like(out))

    first, steady = _time(fwd_bwd)
    report["attn/emulator_fwd_bwd"] = dict(first=round(first, 2), steady_ms=round(steady * 1e3, 2))
    try:
        from fastvideo_kernel import block_sparse_attn_sm100a as sm100a
        from fastvideo_kernel.triton_kernels.index import map_to_index
        idx, num = map_to_index(mask)
        _, steady = _time(lambda: sm100a.block_sparse_attn_sm100a(q, k, v, idx, num, vbs, need_lse=False))
        report["attn/sm100a_bf16_ms"] = round(steady * 1e3, 2)
    except Exception as error:  # noqa: BLE001 - optional comparison only
        report["attn/sm100a_bf16_ms"] = repr(error)[:200]
    print("BENCH " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
