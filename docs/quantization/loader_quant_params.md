# Quantized models and the loader's zero-init allowlist

Read this before adding a quantization config that registers new parameters.
Setting `engine.quantization.transformer_quant` to such a config and then
loading a checkpoint otherwise fails with:

```
ERROR fsdp_load.py Unsupported new parameter: transformer_blocks.0.attn.to_out.scale_input.
Allowed patterns: ['gate_compress', 'proj_l']
```

## What the check actually guards

`load_model_from_full_model_state_dict` in
`fastvideo/models/loader/fsdp_load.py` builds the sharded state dict from the
checkpoint, then computes

```python
unused_keys = set(model.state_dict().keys()) - set(sharded_sd.keys())
```

i.e. *parameters the instantiated model has but the checkpoint did not
provide*. Every such key is a potential silent failure: the loader
zero-initializes it, so a name-mapping bug, a renamed layer, or a checkpoint
from a different architecture would produce a model full of zeros that trains
and generates garbage instead of failing at load.

So the loop raises `ValueError` for any unmapped key **except** a small
allowlist of names that are expected to be absent from every checkpoint. Note
that `strict=False` does *not* disable this — that flag only governs
checkpoint keys missing from the model, not model keys missing from the
checkpoint.

Genuine mismatch detection is the point of the check, so the allowlist stays
narrow: `weight`, `bias`, and every other real model parameter still raise.

## Why quantization configs register new parameters

A quantized linear method (`create_weights`) builds its own scale tensors
alongside the packed weight. The checkpoint holds only the original
`weight`/`bias`, so the scales are always "new parameters" from the loader's
point of view. Examples:

| Config | Registered by `create_weights` | In `state_dict()`? |
|---|---|---|
| `AbsMaxFP8` (`fastvideo/layers/quantization/absmax_fp8.py`) | `weight`, `scale_weight`, `scale_input` | yes |
| `INT8Affine` (`fastvideo/layers/quantization/int8_affine_config.py`) | `weight` only — codes/scales/biases are `persistent=False` buffers | no |
| `NVFP4` / `FP8` | `weight` plus `persistent=False` buffers | no |

`persistent=False` buffers never enter `state_dict()`, so they never reach the
allowlist. Only configs that call `register_parameter` for scale tensors do.

## Admitted names

`ALLOWED_NEW_PARAM_PATTERNS` (module level in
`fastvideo/models/loader/fsdp_load.py`, checked by `is_allowed_new_param`) is a
substring match, and currently admits:

| Pattern | Source |
|---|---|
| `gate_compress` | VSA gate tensor built by the attention backend |
| `proj_l` | SLA projection built by the attention backend |
| `scale_weight` | `AbsMaxFP8` per-tensor / per-merged-partition weight scale |
| `scale_input` | `AbsMaxFP8` per-tensor input (activation) scale |

The two `scale_*` entries match the parameter leaf names the quantization
layer actually registers; they are not a generic `scale` wildcard, so a real
parameter that merely contains "scale" is still rejected.

## Adding a new quantization config

1. Build the model with the config enabled and read the `Unsupported new
   parameter` error — it names the exact FQN.
2. If the offenders are scale/zero-point tensors registered with
   `register_parameter`, add their **leaf name** (e.g. `scale_weight`) to
   `ALLOWED_NEW_PARAM_PATTERNS` in `fastvideo/models/loader/fsdp_load.py`.
   Add the specific names only; do not add a bare `scale` or similar token.
3. Prefer `register_buffer(..., persistent=False)` for values recomputed at
   load time — such buffers need no allowlist entry at all.
4. Update `fastvideo/tests/ops/quantization/test_quant_param_allowlist.py`,
   which asserts both that quant scales are accepted and that an unmapped real
   weight still raises.

`fastvideo/models/loader/shard_cache.py` keeps a mirrored
`_ALLOWED_NEW_PARAM_PATTERNS` tuple used to validate cache manifests. A
mismatch there never fails a run — it only disables the shard cache — but a
new quant config should update it too so the cache stays usable.
