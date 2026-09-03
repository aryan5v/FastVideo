# `fastvideo/models/dits/minimax_h3_hybrid/` — Hybrid MiniMax H3 attention

Opt-in window softmax + linear far branch **on top of** the existing FastVideo
H3 stack. Dense and VSA FastH3 stay the default. Do not vendor FlexAttention,
VDN fused AdaLN, or a second packing scheme.

If you are changing this package, read **Already had**, **On top**, and
**Learned** before editing mappings, module names, or tests.

## Already had (reused — do not reimplement)

These surfaces already existed on `main` and stay authoritative:

| Surface | Where |
|---------|-------|
| Packed document layout `[text \| condition \| audio \| video]` | `pipelines/basic/minimax_h3/packing.py` |
| DiT QKV, per-head QK-norm, 3-axis MM-RoPE, `to_out` | `models/dits/minimax_h3.py` `MiniMaxH3Attention` |
| AdaLN, SwiGLU FFN, dual video/audio heads, text refiner | same DiT |
| Dense `DistributedAttention` and VSA-H3 | `attention/layer.py` |
| Sol-Engine fusions (QK-norm+RoPE, SwiGLU, RMSNorm modulate) | DiT + `FASTVIDEO_MINIMAX_H3_FUSIONS` |
| Tensorwise FP8 on wide linears (`to_q` / `to_k` / `to_v` / `to_out`, plus hybrid `to_out_linear` / `beta_proj` / gates) | `layers/quantization/fp8_config.py` |
| Sequence parallel (shard / all-gather-unpad) | `distributed/` |
| FSDP load + `ALLOWED_NEW_PARAM_PATTERNS` | `models/loader/fsdp_load.py` |
| MLX T2VA DiT (dense SDPA, INT8/6/4, AdaLN cache) | `mlx_runtime/minimax_h3.py` |
| Video/audio VAEs, Qwen3-VL, schedulers (shift 12 / 3) | `models/vaes/`, `encoders/`, `schedulers/` |
| Composed T2VA / FL2VA / Ref2VA pipeline | `pipelines/basic/minimax_h3/` |
| Dense Diffusers → FastVideo `param_names_mapping` | `configs/models/dits/minimax_h3.py` |

Component and official E2E latent parity for that stack is recorded in
`tests/local_tests/minimax_h3/PORT_STATUS.md`. Hybrid does not reopen that port.

## On top (what this package adds)

| Piece | Role |
|-------|------|
| `layout.py` | `HybridSequenceLayout` derived from the packed layout; `window_bounds` / `windows_cover_all_frames`. No torch import at module level (MLX reuses the geometry helpers). |
| `window.py` | Chunk-aligned softmax as a union of dense SDPA rectangles. c1 default: `radius=1`, `chunk=5`, `anchor_frames=both`. |
| `linear.py` | Bidirectional delta-rule scan (`vdn_solve`), optional K/V short conv, text state, output gates. |
| `attention.py` | `HybridAttention` body. Reuses parent QKV / RoPE / `to_out`. Owns `linear_attention`, `to_out_linear`, `softmax_gate`. |
| `checkpoint.py` | VDN key remap, LoRA `W += scale * (B @ A)`, `hybrid_arch_fields_from_spec`. |
| Wiring | `arch.hybrid_attention` (default `False`). Denoising passes `hybrid_layout`. FP8 suffixes `to_out_linear`, `beta_proj`, `softmax_gate`, `output_gate` (not KDA `alpha`). FSDP allowlist for the new siblings. |
| Converter | `scripts/checkpoint_conversion/convert_vdn_h3_to_fastvideo.py` overlays `linear_branch/` + adapters onto a dense `transformer/`. |
| MLX | `mlx_runtime/minimax_h3_hybrid.py`, imported only when hybrid keys are present. |
| CLI / docs | `basic_fasth3.py --no-vsa` (and `--fp8` to quantize tagged linears); MLX auto-detect; `docs/inference/optimizations.md`. |

`MiniMaxH3Attention` registers the extra modules as **siblings** of `to_q`
(`attn.to_out_linear`, not `attn.hybrid.to_out_linear`) so converted
checkpoints match `state_dict()`. `HybridAttention` itself is kept off the
module tree via `object.__setattr__`.

Two sequence-parallel ranks (`hybrid_branch_parallel`, default True) split
softmax vs linear and all-reduce. That split is a no-op unless SP == 2.
SP > 1 still all-gathers the packed sequence into this module first.

## Learned

1. **`param_names_mapping` first match wins.** `attn.orig.to_out.0` must be a
   more specific rule than `attn.orig.*`, or the rewrite stops at `to_out.0`
   and misses FastVideo's `to_out`.
2. **Dropout `attn.orig.to_out.1` / `attn.to_out.1` is not a parameter.** Skip
   it in remap; FastVideo has no matching module.
3. **Do not nest `HybridAttention` as `self.hybrid` in `nn.Module`.** Native
   keys would become `attn.hybrid.to_out_linear`; the converter, FP8 suffixes,
   FSDP allowlist, and MLX detectors all use sibling names.
4. **`torch_key_to_mlx` `.attn.to_out.` (trailing dot) does not rewrite
   `to_out_linear`.** Keep it that way — MLX looks up `attn.to_out_linear.weight`.
5. **Full-cover windows skip the linear branch, but the softmax gate still
   applies.** Gate-off + full-cover matches dense SDPA through `to_out`. Default
   constant init is ~0.99, not 1.0.
6. **Hybrid `forward` wins over VSA** when `hybrid_config` is set. Converted
   checkpoints must run `--no-vsa`.
7. **The converter writes `transformer/` only.** VAE, text encoder, and
   schedulers stay the dense snapshot.
8. **Do not construct the full DiT on CPU to test hybrid.**
   `MiniMaxH3Attention.__init__` still builds `DistributedAttention` and
   `get_attn_backend` raises on CPU (this VM also reports `device=mps` on
   Linux). Exercise `HybridAttention` with a stub parent, or patch the backend.
9. **`layout.py` must stay importable without torch** so the MLX dense path
   does not pull CUDA PyTorch when hybrid keys are absent. `window_bounds` is
   pure Python; import torch only inside tensor helpers. Package `__init__.py`
   must not eagerly import `attention` / `linear` / `window` / `checkpoint`.
10. **Weights stay under the MiniMax H3 Community License.** The conversion
    script is Apache-2.0.
11. **FP8 suffixes are a no-op unless the linear received `quant_config`.**
    Pass it to `beta_proj`, `softmax_gate`, `output_gate`, and `to_out_linear`.
    Leave `FrameKDAAlpha` without it so `alpha.down` / `alpha.up` stay fp32.

→ `.agents/lessons/2026-09-03_h3-hybrid-mapping-and-module-tree.md`

## Manifest

| File | Role |
|------|------|
| `layout.py` | Packed-geometry adapter + window bounds |
| `window.py` | Decomposed window softmax |
| `linear.py` | Scan, gates, short conv |
| `attention.py` | Hybrid body used by `MiniMaxH3Attention` |
| `checkpoint.py` | Converter arithmetic (unit-tested without 70 GiB weights) |
| `__init__.py` | Layout-safe lazy exports (no eager torch) |
| `AGENTS.md` | This file |

External coordinates:

| Path | Role |
|------|------|
| `fastvideo/models/dits/minimax_h3.py` | Parent DiT; hybrid dispatch + flattened module names |
| `fastvideo/configs/models/dits/minimax_h3.py` | Arch knobs + orig→to_out mapping |
| `fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py` | `hybrid_layout` into the DiT |
| `fastvideo/mlx_runtime/minimax_h3.py` | Deferred hybrid import |
| `fastvideo/mlx_runtime/minimax_h3_hybrid.py` | MLX window + scan |
| `scripts/checkpoint_conversion/convert_vdn_h3_to_fastvideo.py` | Overlay + LoRA merge + config stamp |
| `tests/local_tests/minimax_h3/PORT_STATUS.md` | Dense-port status; hybrid listed as a layer on top |

## Cross-refs (if you change X, re-run Y)

| If you touch... | Re-run |
|-----------------|--------|
| `layout.py` / `window.py` | `test_minimax_h3_hybrid_window.py` |
| `linear.py` scan helpers | `test_minimax_h3_hybrid_linear.py` |
| mapping / LoRA merge / spec fields | `test_minimax_h3_hybrid_param_mapping.py` |
| `HybridAttention` or parent module names | `test_minimax_h3_hybrid_attention.py` (CPU, no full DiT) |
| FP8 hybrid suffixes / `quant_config` plumbing | `test_fp8_hybrid_suffixes.py` + `test_hybrid_attention_fp8_tags_gates_not_kda_alpha` |
| MLX hybrid numerics | `test_mlx_minimax_h3_hybrid_parity.py` (needs `mlx`) |
| Converter overlay | convert a real exploded checkpoint, then load with `--no-vsa` |

## Run book

```bash
pytest \
  fastvideo/tests/transformers/test_minimax_h3_hybrid_window.py \
  fastvideo/tests/transformers/test_minimax_h3_hybrid_linear.py \
  fastvideo/tests/transformers/test_minimax_h3_hybrid_param_mapping.py \
  fastvideo/tests/transformers/test_minimax_h3_hybrid_attention.py -q

pytest fastvideo/tests/mlx/test_mlx_minimax_h3_hybrid_parity.py -q

pytest \
  fastvideo/tests/ops/quantization/test_fp8_hybrid_suffixes.py \
  fastvideo/tests/transformers/test_minimax_h3_hybrid_attention.py::test_hybrid_attention_fp8_tags_gates_not_kda_alpha \
  fastvideo/tests/inference/test_basic_fasth3_profile.py::test_fp8_sets_typed_transformer_quant -q
```

`fastvideo/models/` and `fastvideo/tests/` are pre-commit-excluded. Lint
`mlx_runtime/`, `configs/`, and the denoising stage with
`pre-commit run --files`.
