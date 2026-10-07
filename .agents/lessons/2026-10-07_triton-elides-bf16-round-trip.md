---
date: 2026-10-07
experiment: fused NVFP4 MiniMax-H3 decoder kernels (fastvideo/models/vaes/minimax_h3_nvfp4_fused.py)
category: porting
severity: important
---

# Triton Can Skip an Explicit bf16 Rounding Before an Add

## What Happened

A Triton kernel that had to reproduce an eager bf16 RoPE bit for bit
(`bf16(bf16(q * cos) + bf16(rot * sin))`) differed from eager on ~25% of the
rotary elements by one bf16 ulp. The norm before it and every other fused
kernel were bit-exact.

## Root Cause

The intermediate roundings were written as `x.to(tl.bfloat16).to(tl.float32)`.
When the result feeds a float add, the compiler folded the `truncf`/`extf`
round trip, so the add consumed the unrounded fp32 product. Values consumed
by inline PTX or a store kept their rounding, which hid the problem in the
other kernels.

## Fix / Workaround

Round with inline PTX that cannot be folded:
`cvt.rn.bf16.f32` followed by `cvt.f32.bf16` via `tl.inline_asm_elementwise`
(see `_bf16` in the fused decoder kernels). In the same way, use `mul.rn.f32`
for a product that eager rounds before a following add, because Triton
contracts `a * b + c` into an FMA.

## Prevention

When a Triton kernel must match an eager op sequence exactly, compare it with
`torch.equal` against the eager ops on random inputs for every output, not only
the final one. Never rely on dtype round trips or operation order to force
rounding.
