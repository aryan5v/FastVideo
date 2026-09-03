---
date: 2026-09-03
experiment: MiniMax H3 hybrid attention on existing FastVideo H3
category: porting
severity: important
---

# Hybrid H3 Mapping and Module Tree Must Match Converted Sibling Names

## What Happened

Hybrid MiniMax H3 is an opt-in attention body on the existing FastVideo H3 DiT
(packed layout, QKV, RoPE, `to_out`). Early wiring nested `HybridAttention` as
`MiniMaxH3Attention.hybrid` and used a generic `attn.orig.*` rewrite. Converted
VDN keys (`attn.to_out_linear`, `attn.orig.to_out.0`) would not land on the
native `state_dict()`, and CPU tests that built the full DiT died in
`get_attn_backend` (`Invalid attention backend for CPU`; this VM also reports
`device=mps` on Linux).

## Root Cause

1. `param_names_mapping` is first-match. A generic `attn.orig.*` rule rewrites
   `attn.orig.to_out.0.weight` to `attn.to_out.0.weight` and stops. FastVideo's
   linear is `attn.to_out.weight`.
2. Registering `HybridAttention` as an `nn.Module` child emits
   `attn.hybrid.to_out_linear`. The converter, FP8 suffix list, FSDP
   `ALLOWED_NEW_PARAM_PATTERNS`, and MLX key detectors all expect
   `attn.to_out_linear`.
3. `MiniMaxH3Attention.__init__` always constructs `DistributedAttention`, even
   when hybrid is on, so a full DiT is not a valid CPU unit-test fixture.

## Fix / Workaround

- Map `attn.orig.to_out.0` in its own rule **above** `attn.orig.*`. Skip
  dropout `to_out.1`.
- Assign `linear_attention` / `to_out_linear` / `softmax_gate` onto
  `MiniMaxH3Attention` and keep `HybridAttention` off the module tree
  (`object.__setattr__(self, "hybrid", hybrid)`).
- Test `HybridAttention` with a stub parent (`ReplicatedLinear` QKV/`to_out`)
  or patch `get_attn_backend`. Do not instantiate `MiniMaxH3Transformer3DModel`
  on CPU for hybrid unit tests.

## Prevention

Read `fastvideo/models/dits/minimax_h3_hybrid/AGENTS.md` (Already had / On top /
Learned) before changing mappings, module names, or hybrid tests. Keep a
state_dict assertion that native keys are siblings (`to_out_linear.`, not
`hybrid.`).
