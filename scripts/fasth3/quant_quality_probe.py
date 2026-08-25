# SPDX-License-Identifier: Apache-2.0
"""FastH3 quantization quality probe — hardware-independent error table.

Runs on the CUDA cluster (torch) over the real FastH3-Preview 33B weights and
computes, per candidate grid, the weight-level reconstruction error of the
matrices the MLX runtime will quantize (attention QKV/out + FFN, the
converter's quantizable set). The same arithmetic MLX applies, so these
numbers transfer directly to the Mac — the Mac bake-off then only measures
speed/memory and the eyeball quality of the finalists.

Grids (group 64 where applicable, matching the MLX deploy specs):
- affine int8 / int6 / int4: per-group (max-min) scale, zero point, round+clip
- mxfp8 / mxfp4: E8M0 power-of-two block scale (32), E4M3/E2M1 elements, RNE
- nvfp4: E4M3 block scale (16) — included for reference, CUDA-side only

Also reports the projected resident sizes for the 33B with the AdaLN cache
(19.3B resident-param estimate for attention+FFN, ~33.2B total).

Usage (cluster):
    python scripts/fasth3/quant_quality_probe.py \
        --model-root <H3 snapshot dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# The converter's quantizable set (suffixes; everything else stays high-prec).
QUANTIZABLE_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
)

GROUP = 64
BITS_PER_PARAM = {"int8": 8.5, "int6": 6.5, "int4": 4.5, "mxfp8": 8.25, "mxfp4": 4.25, "nvfp4": 4.5}
RESIDENT_PARAMS_B = 19.3  # attention+FFN after the AdaLN cache; total 33.2B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--formats", default="int8 int6 int4 mxfp8 mxfp4 nvfp4")
    return parser.parse_args()


def affine_quantize(w: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    """Group-wise affine int quantization on the last dim (scale+zero, round+clip)."""
    levels = 2**bits
    orig = w.shape
    w2 = w.reshape(-1, group)
    wmin = w2.min(dim=1, keepdim=True).values
    wmax = w2.max(dim=1, keepdim=True).values
    scale = (wmax - wmin) / (levels - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = -wmin / scale
    q = torch.round(w2 / scale + zero).clamp(0, levels - 1)
    wq = (q - zero) * scale
    return wq.reshape(orig)


def _e8m0_scale(x: torch.Tensor) -> torch.Tensor:
    """E8M0: power-of-two max-abs per block (round up to the representable grid)."""
    m = x.abs().amax(dim=-1, keepdim=True)
    # E8M0 grid: 2^e for e in [-127, 127]; round-up to the next pow2.
    exponent = torch.ceil(torch.log2(m.clamp_min(1e-30)))
    return 2.0**exponent


def mx_quantize(w: torch.Tensor, mode: str, group: int = GROUP) -> torch.Tensor:
    """Approximate the MLX float formats: block power-of-two scale, element RNE.

    E4M3: 3 mantissa bits -> step 2^-2 of the scale (values quantized to
    nearest multiple of scale/4, max 448*scale); E2M1: 1 mantissa bit ->
    multiples of scale/2. Close enough for the error table (the MX grids'
    dominant error is the coarse shared scale, which is exact here).
    """
    w2 = w.reshape(-1, group)
    scale = _e8m0_scale(w2)
    if mode == "mxfp8":
        # E4M3: 3 mantissa bits -> 8 levels per exponent, step = scale/8.
        step = scale / 8.0
        q = torch.round(w2 / step) * step
        q = q.clamp(-448 * scale, 448 * scale)
    elif mode in ("mxfp4", "nvfp4"):
        # E2M1: 1 mantissa bit -> 2 levels per exponent, step = scale/2.
        step = scale / 2.0
        q = torch.round(w2 / step) * step
        q = q.clamp(-6 * scale, 6 * scale)
    else:
        raise ValueError(mode)
    return q.reshape(w.shape)


def rel_l2(q: torch.Tensor, w: torch.Tensor) -> float:
    return float(((q - w).float().square().sum() / w.float().square().sum()).sqrt())


def main() -> None:
    args = parse_args()
    from safetensors import safe_open

    model_root = Path(args.model_root)
    index = json.loads((model_root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    errors: dict[str, list[float]] = {fmt: [] for fmt in args.formats.split()}
    total = 0.0
    params = 0
    for key in sorted(weight_map):
        if not key.endswith(QUANTIZABLE_SUFFIXES):
            continue
        shard = model_root / "transformer" / weight_map[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            w = f.get_tensor(key).float()
        w = w.reshape(w.shape[0], -1)
        if w.shape[-1] % GROUP:
            continue
        total += float(w.square().sum())
        params += w.numel()
        for fmt in errors:
            if fmt.startswith("int"):
                bits = int(fmt[3])
                q = affine_quantize(w, bits)
            else:
                q = mx_quantize(w, fmt)
            errors[fmt].append(float((q - w).float().square().sum()))

    print(f"\nFastH3-Preview 33B quantization quality probe")
    print(f"quantizable params: {params/1e9:.1f}B (attention + FFN, group-64-compatible)")
    print(f"\n{'format':<8} {'mean rel-L2':>12} {'M5 survey (Wan, for scale)':>28}")
    m5 = {"int8": 0.00546, "int6": 0.022, "int4": 0.0919, "mxfp8": 0.0679, "mxfp4": 0.1209, "nvfp4": 0.1029}
    for fmt in errors:
        err = float(sum(errors[fmt])) / float(total)
        print(f"{fmt:<8} {err:>12.5f}   {m5.get(fmt, ''):>28}")
    print("\nprojected 33B resident (attention+FFN int-grid at group 64, AdaLN cache applied):")
    for fmt in errors:
        bits = BITS_PER_PARAM[fmt]
        print(f"  {fmt:<8} {RESIDENT_PARAMS_B * bits / 8.0:>7.1f} GB (~{RESIDENT_PARAMS_B * bits / 8.0 / 1.0737:.1f} GiB)")
    print("\nPROBE DONE")


if __name__ == "__main__":
    main()