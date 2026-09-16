# NVFP4 quantization for MiniMax-H3 (Blackwell / sm100)

Load-time NVFP4 weight quantization for the MiniMax-H3 joint audio-video DiT,
implemented in `fastvideo/layers/quantization/nvfp4_config.py`.

This page is written for someone with none of the session context in which the
lane was built. It covers what NVFP4 is here, why the config used to be a
**silent no-op on H3**, how to turn it on, exactly which layers are quantized,
what is required to run it, and how to write/read a compact quantized
checkpoint.

---

## 1. What NVFP4 is here

NVFP4 is NVIDIA's block-scaled FP4 format: an **e2m1** 4-bit weight code, an
**e4m3** scale shared by every 16 weights along the input dimension, and a
single **fp32/bf16 global scale** per tensor. In this repo it is backed by
[`flashinfer`](https://github.com/flashinfer-ai/flashinfer) —
`nvfp4_quantize` for the conversion and `mm_fp4` for the GEMM.

It is *not* generic FP4 / OCP-FP4 / MX-FP4 — hence the explicit name. The
scale layout used everywhere in this module is `SfLayout.layout_128x4` with
`do_shuffle=False`; rows are padded to a 128-row tile for the kernel and
narrowed back (`_nvfp4_quantize`).

For a linear weight of shape `[out, in]`:

| Tensor | dtype | shape | contents |
|---|---|---|---|
| `_nvfp4_weight` | `uint8` | `[out, ceil(in/2)]` | two e2m1 codes per byte, packed along K |
| `_nvfp4_weight_scale` | `uint8` | `[out, ceil(in/16)]` | e4m3 block-scale bit patterns (rows padded to a multiple of 128 when `out` is not; H3's dims are, so this does not apply here) |
| `_weight_global_sf` | `bfloat16` | scalar | `(448 * 6) / max\|W\|` |
| `_nvfp4_alpha` | `float32` | scalar | `1 / _weight_global_sf`, kept at fp32 precision |

All four are registered as **`persistent=False`** buffers by
`convert_model_to_nvfp4`, so they never appear in a `state_dict`. That is
deliberate (they are derived state on the standard path) and is exactly why a
compact quantized checkpoint needs the sidecar described in §7.

Weight storage drops from 16 bits/weight to **0.5625** (4 bits + 1 scale byte
per 16 weights) — about **3.6x** smaller than bf16.

## 2. Why NVFP4 used to do nothing on H3

`NVFP4Config.get_quant_method` decides what to quantize by looking the layer's
module path up in a set. That set used to be a hardcoded module-level constant
of ~577 literal `ltx2.blocks.*` strings:

```python
_LTX2_NVFP4_LINEAR_PREFIXES = frozenset(...)  # 48 blocks x 12 suffixes + adaln
```

H3's linears are named `minimax_h3.transformer_blocks.*`, so **not one path
matched**, and:

- no `NVFP4QuantizeMethod` was attached to any layer,
- `_maybe_quantize_model` in `fsdp_load.py` therefore found no NVFP4 layer and
  skipped the conversion entirely,
- the model ran dense bf16 — with **no error, no warning, and no log line**.

That is the worst possible failure shape: `transformer_quant="NVFP4"` looked
configured, and the only way to notice was to measure memory or read the code.

The fix lifts the layer list into the config (the class docstring had been
asking for this):

```python
NVFP4Config(layer_profile="refine",
            retain_original_weights=None,
            layer_prefixes=None,        # None -> the LTX-2 set (unchanged default)
            exclude_prefixes=None)      # extra never-quantize patterns
```

`layer_prefixes=None` keeps the historical LTX-2 behaviour bit-for-bit, so
existing LTX-2 deployments are unaffected. Other models opt in explicitly.

## 3. Turning it on for H3

**Option A — the H3 profile (use this one).** Build the config explicitly and
hand it to `FastVideoArgs`:

```python
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

fastvideo_args.transformer_quant = NVFP4Config.for_minimax_h3()
```

`FastVideoArgs._apply_transformer_quant` pins a pre-built instance onto
`pipeline_config.dit_config.quant_config` before the DiT is constructed, which
is required — linears attach their `quant_method` during `__init__`. Setting
`pipeline_config.dit_config.quant_config` directly works too and takes
precedence.

Equivalent, if you prefer the raw constants:

```python
from fastvideo.layers.quantization.nvfp4_config import (
    MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES,
    MINIMAX_H3_NVFP4_LINEAR_PREFIXES,
    NVFP4Config,
)

config = NVFP4Config(
    layer_prefixes=MINIMAX_H3_NVFP4_LINEAR_PREFIXES,
    exclude_prefixes=MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES,
)
```

`for_minimax_h3(...)` also forwards keyword arguments, e.g.
`for_minimax_h3(retain_original_weights=True)`.

**Option B — by name. This does not work for H3.** `transformer_quant:
"NVFP4"` (YAML) or `--transformer-quant NVFP4` resolves through
`get_quantization_config("NVFP4")()` — a no-argument constructor, so it gets
`layer_prefixes=None`, i.e. the LTX-2 set. On H3 that is the silent no-op from
§2. The registry has no way to carry a model-specific layer list; a
pre-built instance is the only way to express one.

**How to confirm it actually engaged.** Three things must be true after the
model is loaded:

1. The loader logs `Converting loaded model weights for NVFP4 linear layers`.
2. `convert_model_to_nvfp4` logs its retention receipt, e.g.
   `NVFP4 weight purge receipt: purged 0 original bf16 weight tensors ...; retained 300`.
   Expected on the standard multi-GPU path — see §6 on the FSDP retention.
3. Peak memory moves: +~7 GiB for the packed buffers, and −~25 GiB only if the
   receipt reports purges (single-process loads).

If none of those appear, the config did not cover the model's layer paths —
check `layer_prefixes` against `[name for name, _ in model.named_modules()]`.

## 4. Which H3 layers are quantized

With `MiniMaxH3ArchConfig` defaults (`prefix="minimax_h3"`, `num_layers=50`,
`hidden_size=5376`, `ffn_dim=14336`):

**Quantized — 300 linears** (`MINIMAX_H3_NVFP4_LINEAR_PREFIXES`):

| Prefix | Count |
|---|---|
| `minimax_h3.transformer_blocks.{0..49}.attn.to_q` / `to_k` / `to_v` / `to_out` | 200 |
| `minimax_h3.transformer_blocks.{0..49}.ff.fc_in` / `fc_out` | 100 |

Those 300 linears hold ≈13.5B parameters: ≈25.1 GiB as bf16, ≈7.1 GiB as
packed NVFP4 (arithmetic from the arch constants, not a measurement).

**Deliberately excluded:**

| Prefix | Why |
|---|---|
| `...attn.to_gate_compress` | **Never quantize.** See §5. |
| `minimax_h3.token_refiner.refiner_blocks.*` | Text-stream refiner; small, and its attention sits outside the packed video/audio sequence the FP4 kernels are tuned for. Quantizing it is a follow-up, not a default. |
| `minimax_h3.transformer_blocks.*.adaln_proj.linear`, `norm_out.linear` | Per-block/per-forward modulation projections; small, and they feed the modulation path whose precision the distilled deploy is sensitive to. |
| `proj_in`, `audio_proj_in`, `proj_out`, `audio_proj_out`, `time_embedder`, `rope` | H3 pins these to fp32 via `MiniMaxH3Transformer3DModel._keep_in_fp32_modules` — quantizing them would fight the model's own dtype policy. |
| `context_embedder` | Judgment call: the text conditioning stream, structurally like the fp32-pinned input projections. |
| norms, `rope` tables, `scale_shift_table`-style params | Not `LinearBase`; never candidates. |

The set is a **positive allowlist**, so extending it (say, adding the token
refiner) is a one-line change to `layer_prefixes` — nothing else in the module
needs to know.

Note that `adaln_basis` (global timestep-basis projector) and
`adaln_proj.linear` (per-block modulation) are different modules with similar
names; only the latter is even a candidate, and it is excluded.

## 5. `attn.to_gate_compress` must never be quantized

`MiniMaxH3Attention.to_gate_compress` is the VSA (Video Sparse Attention)
compression gate. Two independent reasons:

1. **Its output decides sparse routing** — which tiles the sparse attention
   attends to. That is a discrete decision, so quantization error there is not
   bounded by the weight-quantization error; it can flip a routing decision.
2. **H3's own deployment path ignores it**, and the released checkpoint
   zero-initializes the gate, so the branch is exactly disabled until it is
   finetuned. `MiniMaxH3Attention._gate_active()` tests the loaded weight once
   and skips the branch entirely while it is structurally zero. There is
   nothing to gain by quantizing it, and a nonzero quantized gate would defeat
   the skip.

The name matches no generic exclusion heuristic (no `norm`, no `scale_shift_table`,
no `proj_*`), so it would be swept up by any broad suffix rule. It is excluded
by **three independent mechanisms**, so no single mistake can quantize it:

1. **It is absent from the allowlist.** `MINIMAX_H3_NVFP4_LINEAR_PREFIXES`
   contains exactly the 300 paths of §4; the gate is not among them.
2. **`for_minimax_h3()` sets `exclude_prefixes`** to
   `MINIMAX_H3_NVFP4_EXCLUDED_LINEAR_SUFFIXES` = `("attn.to_gate_compress",)`.
3. **`_ALWAYS_EXCLUDED_LINEAR_SUFFIXES` is unconditional.** The deny check runs
   *first* in `NVFP4Config.is_nvfp4_linear_prefix`, before the allowlist, and no
   caller can override it. Passing an allowlist that explicitly contains
   `minimax_h3.transformer_blocks.0.attn.to_gate_compress` still returns
   `False`.

Exclusion entries match either a full module path or a trailing suffix, and
only at a dot boundary (`"ff.fc_in"` does not match `"cross_ff.fc_in"`).

Regression guards:
`test_gate_is_excluded_even_if_a_caller_allowlists_it` and
`test_get_quant_method_attaches_for_h3_and_skips_the_gate` in
`fastvideo/tests/ops/quantization/test_nvfp4_h3_prefixes.py`.

## 6. Requirements and caveats

**Requirements**

- `flashinfer-python` (validated set in
  [Optimizations](../inference/optimizations.md) §flashinfer). The module
  itself imports fine without it; only the quantize/GEMM ops raise, with
  `Install with 'pip install flashinfer-python'`. Exception: restoring a
  sidecar (§7) needs no flashinfer at all.
- **Blackwell / sm100.** `NVFP4Config.get_min_capability()` returns `100`. The
  kernels need FP4 tensor cores.
- A bf16 checkpoint. No pre-quantized weights are required: conversion happens
  at load, from the dense weights, every time.

**Caveat: `x_global_sf` is hardcoded `1.0`**

`NVFP4QuantizeMethod.x_global_sf` is `torch.tensor(1.0)` — a constant, never
derived from the data. It is used twice:

- as the global scale when quantizing *activations*
  (`quantize_input` / `apply`),
- as the denominator of the GEMM alpha: `alpha = layer._nvfp4_alpha / x_global_sf`
  (or `1 / (x_global_sf * weight_global_sf)`).

The weight side of the same math *is* data-derived
(`convert_model_to_nvfp4` computes `(448 * 6) / max|W|`), and the standalone
`fastvideo/layers/fp4linear.py` derives the activation side too
(`_global_sf` / `448.0 * 6.0 / maxabs`). So H3 (and LTX-2) currently quantize
activations against a fixed unit global scale instead of adapting to each
activation's dynamic range. This is the **first suspect** for any quality
regression from NVFP4, and it is a shared pre-existing property of this config,
not something the H3 enablement introduced.

It has not been measured on H3. The experiment to run: compare `x_global_sf =
1.0` against a per-tensor `448*6/max|x|` on a fixed prompt/seed and score both
with the project's quality gates. Because `x_global_sf` is *not* part of a
saved checkpoint (§7) it must be identical on both sides of a save/load — if it
ever becomes data-derived, it has to be persisted alongside the weights.

**Other caveats**

- `layer_profile` (`"base"` / `"refine"`) is an LTX-2 streaming concept. H3
  layers are never "refine-only", so every H3 quantized layer is FP4 on every
  step.
- No H3 quality evidence exists yet (no SSIM, no VLM adherence gate against the
  bf16 baseline), and no performance numbers. See §9.
- **FSDP: the dense bf16 weights are retained, not purged.** `shard_model` runs
  before `_maybe_quantize_model` in `maybe_load_fsdp_model`, so by conversion
  time every `weight` is a `DTensor`; `convert_model_to_nvfp4` quantizes the
  local shard via `to_local()` but skips the purge for DTensor weights
  (per-shard resharding bookkeeping was never implemented). On the standard
  multi-GPU path the receipt therefore reads `retained 300`, and enabling NVFP4
  *adds* the ~7 GiB of packed buffers rather than replacing the ~25 GiB of bf16
  weights. The bandwidth win at inference is real (the FP4 GEMM reads the
  packed weight); the memory win is not, until that purge lands. This is a
  pre-existing property of the purge policy, not of the H3 enablement.
- **FSDP/TP and sidecars:** conversion quantizes each rank's *local shard*, so
  the global scale is per-shard. A sidecar saved from a sharded model stores
  local shards and is only reloadable into an identically sharded model.

## 7. Compact NVFP4 checkpoints

**The problem.** The FP4 tensors are `persistent=False`, so they are not in a
`state_dict`: saving an H3 model writes ~25 GiB of dense bf16 weights for these
300 linears, and every load re-quantizes them from scratch (requiring
flashinfer and the time to run 300 quantizations).

**The format.** A *sidecar* safetensors file holding the quantized tensors,
keyed by module path:

```
<module fqn>::_nvfp4_weight
<module fqn>::_nvfp4_weight_scale
<module fqn>::_weight_global_sf
<module fqn>::_nvfp4_alpha
```

Its safetensors `metadata` carries a JSON manifest under the key
`fastvideo_nvfp4`:

```json
{
  "format": "fastvideo.nvfp4",
  "version": 1,
  "sf_layout": "layout_128x4",
  "do_shuffle": false,
  "block_size": 16,
  "num_layers": 300,
  "layers": {"minimax_h3.transformer_blocks.0.attn.to_q": [5376, 5376], "...": "..."},
  "quant_prefixes": {"minimax_h3.transformer_blocks.0.attn.to_q": "minimax_h3.transformer_blocks.0.attn.to_q"},
  "model_class": "MiniMaxH3Transformer3DModel"
}
```

`layers` records each linear's `[out, in]`, so a load can validate tensor
shapes (and report coverage) even when the bf16 weights are not present at all.
`quant_prefixes` records the prefix the layer was tagged with, which is how you
tell an H3-built sidecar from an LTX-2-built one.

**Writing one:**

```python
from fastvideo.layers.quantization.nvfp4_config import save_nvfp4_checkpoint

receipt = save_nvfp4_checkpoint(model, "transformer.nvfp4.safetensors")
# {'num_layers': 300, 'num_tensors': 1200, 'quantized_bytes': ..., 
#  'dense_bfloat16_bytes': ..., 'compression_ratio': 3.55...}
```

The model must already be converted (`convert_model_to_nvfp4` has run, which is
what the loader does at load time). The receipt, which is also logged, reports
both sizes so the win is visible without `ls -l`.

**Reading one:**

```python
from fastvideo.layers.quantization.nvfp4_config import load_nvfp4_checkpoint

restored = load_nvfp4_checkpoint(model, "transformer.nvfp4.safetensors")
```

`load_nvfp4_checkpoint`:

- registers the four buffers on every NVFP4-tagged linear, byte-for-byte as a
  fresh conversion would (`test_load_restores_buffers_without_reconverting`),
- **never calls flashinfer** — a host that only serves a pre-quantized
  checkpoint needs neither the kernels nor a GPU for this step,
- works when the dense `weight` is absent entirely (the compact case),
- validates the manifest (`format`, `version`, `sf_layout`, `do_shuffle`,
  `block_size`) and every tensor shape, raising `ValueError` — a layout
  mismatch is never downgraded, because mis-read nibbles are silent corruption,
- reports layer-set mismatches: `strict=True` (default) raises, `strict=False`
  logs and restores the intersection,
- applies the same bf16-weight retention policy as the conversion
  (`purge_dense_weights=True` by default).

Conventional path for a checkpoint file or directory:
`nvfp4_sidecar_path_for(".../transformer.safetensors")` →
`.../transformer.nvfp4.safetensors`; for a directory → `<dir>/nvfp4.safetensors`.

### What still needs a loader-side change (not done here)

`load_nvfp4_checkpoint` is complete, tested, and callable today, but the loader
does not yet *dispatch* it — `fastvideo/models/loader/fsdp_load.py` is owned by
another change, so this work did not touch it. Two separate things are needed
for a compact release, and they are independent:

1. **Skip the re-quantization.** In `_maybe_quantize_model`, where
   `convert_model_to_nvfp4(model)` is called, branch on the sidecar's presence:

   ```python
   from fastvideo.layers.quantization.nvfp4_config import (
       load_nvfp4_checkpoint, nvfp4_sidecar_path_for,
   )

   sidecar = nvfp4_sidecar_path_for(transformer_checkpoint_path)
   if os.path.exists(sidecar):
       load_nvfp4_checkpoint(model, sidecar)
   else:
       convert_model_to_nvfp4(model)
   ```

   `_maybe_quantize_model(model)` does not currently receive a path, so its
   signature (or its call site) has to grow one. Without this, the sidecar
   saves load *time* only if the caller invokes `load_nvfp4_checkpoint`
   manually after the model is built — the automatic path still re-quantizes.

2. **Drop the bf16 weights from the released file.** This is the half that
   actually shrinks the *release*. The sidecar already omits them, but the
   main checkpoint cannot: at load time the DiT is constructed with a real
   `weight` `Parameter` (`NVFP4QuantizeMethod.create_weights` allocates it), so
   `weight` is in `model.state_dict()`, and
   `load_model_from_full_model_state_dict` treats any model key the checkpoint
   does not provide as a new/unmapped parameter — it zero-initializes it and
   raises `ValueError` unless the name is on the narrow allowlist described in
   [Quantized Checkpoint Loading](loader_quant_params.md). Shipping a
   checkpoint without the 300 NVFP4 `weight` tensors therefore requires that
   check to accept `weight` on NVFP4-tagged linears *whose sidecar covers
   them* — and requires `create_weights` (or the load path) to tolerate a
   `weight` that is never filled.

   With both halves in place the transformer file for these 300 linears goes
   from ≈25 GiB to ≈7 GiB.

Until then, treat the sidecar as: a working format + restore path, a way to
serve pre-quantized weights without flashinfer, and a de-risked piece of the
compact-release work.

## 8. Tests

CPU-only, no flashinfer, no CUDA (the FP4 ops and
`NVFP4QuantizeMethod.__init__`'s cuda allocation are stubbed):

```bash
pytest fastvideo/tests/ops/quantization/test_nvfp4_h3_prefixes.py -v
pytest fastvideo/tests/ops/quantization/test_nvfp4_sidecar.py -v
```

`test_nvfp4_h3_prefixes.py` (9 tests) covers the LTX-2 default (no
regression, 577 prefixes), the 300-linear H3 set, the gate exclusion including
the hostile-allowlist case, `from_config` round-tripping, the
import-without-flashinfer contract, and a real `ReplicatedLinear` end-to-end
attachment check.

`test_nvfp4_sidecar.py` (14 tests) covers the save receipt and size win,
byte-identical restore, restore without flashinfer, restore into a model with
no dense weights, the retention policy, layer-set/layout/version/shape
mismatch handling, and the "no NVFP4 layers attached" error that names the
prefix-set cause.

## 9. What is NOT verified

- **No GPU run.** Nothing here has been executed on Blackwell, and the FP4
  kernels have not been exercised by this work. All tests stub the quantizer,
  so they verify shapes, dtypes, layout bookkeeping, keying and policy — not
  numerics.
- **No quality evidence.** No SSIM, no VLM adherence check, no comparison to
  the bf16 baseline for H3. Combined with the `x_global_sf = 1.0` caveat (§6),
  a quality regression is plausible and unmeasured.
- **No performance numbers.** Memory and throughput impact unmeasured; the
  activation-quantize cost per call is unmeasured.
- **No FSDP/TP validation** of save/load. A sidecar holds local shards
  (§6); round-tripping under `fully_shard` has not been tested.
- **No end-to-end loader dispatch** — see §7.
- **Checkpoint-wrapper caveat.** Sidecar keys are `named_modules()` FQNs. If a
  model is saved with `checkpoint_wrapper`-style prefixes and loaded without
  them (or vice versa), the keys will not match; `strict=True` will say so
  rather than restoring partially.

## 10. Files

| Path | What |
|---|---|
| `fastvideo/layers/quantization/nvfp4_config.py` | `NVFP4Config` (+ `layer_prefixes` / `exclude_prefixes` / `for_minimax_h3`), `NVFP4QuantizeMethod`, `convert_model_to_nvfp4`, `save_nvfp4_checkpoint`, `load_nvfp4_checkpoint`, `read_nvfp4_sidecar_metadata`, `nvfp4_sidecar_path_for`, `MINIMAX_H3_NVFP4_LINEAR_PREFIXES` |
| `fastvideo/tests/ops/quantization/test_nvfp4_h3_prefixes.py` | CPU-only prefix-selection tests |
| `fastvideo/tests/ops/quantization/test_nvfp4_sidecar.py` | CPU-only serialization tests |
| `fastvideo/models/loader/fsdp_load.py` | `_maybe_quantize_model` dispatch (owned elsewhere; sidecar branch not yet wired — §7) |
| `fastvideo/layers/linear.py` | `ReplicatedLinear.__init__` is where `get_quant_method` is called |
| `docs/quantization/loader_quant_params.md` | Why unmapped parameters are a hard error at load |
| `docs/quantization/h3_int8_affine.md` | The sibling INT8 lane for the same model |
