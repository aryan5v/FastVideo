# Quantized models and loader allowlists

Some quantization methods register scale tensors that are absent from dense
checkpoints. Without an allowlist entry, loading fails with:

```
Unsupported new parameter: ...scale_weight...
```

`ALLOWED_NEW_PARAM_PATTERNS` in `fastvideo/models/loader/fsdp_load.py` admits
expected new parameter leaf names via substring match. Prefer
`register_buffer(..., persistent=False)` for values recomputed at load time so
they never enter `state_dict()`.

When adding a quant config that calls `register_parameter` for scales, add the
leaf name to `ALLOWED_NEW_PARAM_PATTERNS` and mirror it in
`fastvideo/models/loader/shard_cache.py`.
