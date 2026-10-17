# SPDX-License-Identifier: Apache-2.0
"""Compare FlashInfer's NVFP4 GEMM backends on representative 480p H3 shapes."""

import json
import os

os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")

import flashinfer
import torch

from fastvideo.layers.quantization.nvfp4_config import _mm_fp4, _nvfp4_quantize


@torch.inference_mode()
def main():
    torch.manual_seed(1234)
    for rows, inner, columns in ((13312, 5376, 7168), (13312, 5376, 28672), (13312, 14336, 5376)):
        a = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(columns, inner, device="cuda", dtype=torch.bfloat16)
        sa = (448.0 * 6.0) / a.abs().max().float()
        sb = (448.0 * 6.0) / b.abs().max().float()
        ap, ass = _nvfp4_quantize(a, sa, sfLayout=flashinfer.SfLayout.layout_128x4)
        bp, bs = _nvfp4_quantize(b, sb, sfLayout=flashinfer.SfLayout.layout_128x4)
        alpha = 1.0 / (sa * sb)
        del a, b
        results = {"shape_mkn": [rows, inner, columns], "ms": {}, "errors": {}}
        reference = None
        for backend in ("cudnn", "cutlass"):
            try:
                def call():
                    return _mm_fp4(ap, bp.T, ass, bs.T, alpha, torch.bfloat16, None, backend=backend)
                for _ in range(3):
                    output = call()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    output = call()
                end.record()
                end.synchronize()
                results["ms"][backend] = start.elapsed_time(end) / 10
                if reference is None:
                    reference = output
                else:
                    results["different_fraction"] = (output != reference).float().mean().item()
            except Exception as exc:
                results["errors"][backend] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(results), flush=True)
        del ap, ass, bp, bs, alpha, reference
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
