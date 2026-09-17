# Affine group-64 INT8 for MiniMax-H3 (CUDA)

Weight-only affine INT8 quantization for H3 DiT inference on CUDA, with the
group-64 / 8-bit affine math the Apple Silicon (MLX) deployment lane already
validates. Implemented in
`fastvideo/layers/quantization/int8_affine_config.py`.

Covers how to enable the lane, which H3 layers are quantized, and what has been
verified.

---

## 1. What this is

Affine quantization stores each weight as

```
w ≈ code * scale + bias
```

where `code` is an unsigned integer and `scale`/`bias` are shared by a **group
of 64 weights taken along the input (contraction) dimension**. For a linear
weight of shape `[out, in]` that means `in / 64` groups per output row.

The quantizer is a per-group min/max affine quantizer, not a symmetric one:
it anchors at whichever endpoint of the group's range has the larger
magnitude, so the extreme weight in each group round-trips exactly. That
detail is not incidental — it is the behaviour of MLX's `mx.quantize(...,
mode="affine")`, and it is what the project's QAT pipeline was tuned against.

### Weight-only, deliberately

Activations are **not** quantized. `apply()` dequantizes the stored codes back
to the activation dtype and runs a normal bf16/fp32 GEMM. Consequences worth
knowing:

- Accuracy is limited by the weight error alone (measured below), not by an
  activation-error term nobody has characterised for H3.
- It needs no INT8 tensor cores and no custom kernel — it runs anywhere bf16
  does, including CPU.
- The memory/bandwidth win is real only if you also drop the bf16 weight (see
  `retain_original_weight`). Compute is unaffected until a fused INT8 GEMM
  lands; that is a follow-up, not a prerequisite.

## 2. This is NOT the MLX INT8 QAT callback

Two different things in this repo are "affine group-64 INT8". Do not conflate
them:

| | MLX lane (`mlx_affine_qat.py`) | This lane (`int8_affine_config.py`) |
|---|---|---|
| Purpose | **Training-time** fake-quant callback | **Load-time** inference quantization |
| When it runs | Every forward, during QAT finetuning | Once, when the checkpoint is loaded |
| What it produces | A straight-through-estimate `w` for the optimizer to train against | Stored int8 codes + per-group scales/biases |
| Target runtime | Apple Silicon / MLX | CUDA (PyTorch) |
| Weight after | Full-precision master, unchanged | Quantized in place (bf16 copy optionally retained) |
| Gradients | Pass through to the master weight | None — inference only |

They share the *quantizer math* (this module's `int8_affine_quantize` /
`int8_affine_dequantize` are bit-identical to `mlx_affine_quantize_reference` /
`mlx_affine_dequantize_reference` — verified on fp32, fp16 and bf16 inputs)
and nothing else. Nothing here writes a QAT checkpoint, and nothing here can
be used to *run* QAT: `apply()` deliberately falls back to the dense bf16
weight whenever `torch.is_grad_enabled()`, so a training step never sees a
frozen dequantized copy.

## 3. Turning it on for H3

The config is registered under the name `INT8Affine` (see
`fastvideo/layers/quantization/__init__.py`).

**Option A — the verified H3 profile (recommended).** Build the config
explicitly, because the registry resolves a bare name through the no-argument
constructor and therefore cannot carry the H3-specific profile:

```python
from fastvideo.layers.quantization.int8_affine_config import INT8AffineConfig

fastvideo_args.transformer_quant = INT8AffineConfig.for_minimax_h3()
```

`transformer_quant` accepts a pre-built instance and pins it onto
`pipeline_config.dit_config.quant_config` in `FastVideoArgs._apply_transformer_quant`,
before the DiT is constructed — which is required, because linears attach their
`quant_method` during `__init__`. Setting
`pipeline_config.dit_config.quant_config` directly works too.

**Option B — by name.** `transformer_quant: "INT8Affine"` (YAML) or
`--transformer-quant INT8Affine` builds `INT8AffineConfig()` with the generic
defaults. That default is safe on H3 (it selects the same attention/FFN GEMMs,
minus `adaln_proj.linear`, and the same exclusions apply), but it is the
model-agnostic profile rather than the H3-verified one.

The load-time conversion is triggered from `_maybe_quantize_model` in
`fastvideo/models/loader/fsdp_load.py`. That function dispatches on an explicit
`isinstance` chain, so it needs a branch for `INT8AffineQuantizeMethod` calling
`convert_model_to_int8_affine(model)`. **If that branch is missing, inference
is still correct** — `apply()` converts lazily on first forward and logs a
warning naming this exact cause — but you will see the warning, and the
conversion happens inside the first forward instead of at load.

Any BF16 checkpoint works unchanged; there are no pre-quantized weights to
produce and none are written back (the codes/scales are non-persistent
buffers, so they are re-derived on every load).

## 4. Which H3 layers are quantized

H3's `MiniMaxH3TransformerBlock` builds
`{prefix}.transformer_blocks.{i}.{attn, ff, adaln_proj}`, and
`MiniMaxH3TokenRefiner` builds
`{prefix}.token_refiner.refiner_blocks.{i}.{attn, ff}` (no `adaln_proj`).
With the defaults from `MiniMaxH3ArchConfig` (`prefix="minimax_h3"`,
`num_layers=50`, `num_refiner_layers=2`):

**Quantized** (362 linears under `for_minimax_h3()`):

| Prefix | Count |
|---|---|
| `minimax_h3.transformer_blocks.{0..49}.attn.to_q` / `to_k` / `to_v` / `to_out` | 200 |
| `minimax_h3.transformer_blocks.{0..49}.ff.fc_in` / `fc_out` | 100 |
| `minimax_h3.transformer_blocks.{0..49}.adaln_proj.linear` | 50 |
| `minimax_h3.token_refiner.refiner_blocks.{0..1}.attn.to_q` / `to_k` / `to_v` / `to_out` | 8 |
| `minimax_h3.token_refiner.refiner_blocks.{0..1}.ff.fc_in` / `fc_out` | 4 |

**Excluded, and why:**

| Prefix | Why |
|---|---|
| `...attn.to_gate_compress` | **Never quantize this.** See §5. |
| `minimax_h3.adaln_basis` | Global timestep-basis projector feeding every block's modulation. |
| `minimax_h3.proj_in`, `audio_proj_in`, `proj_out`, `audio_proj_out` | H3 pins these to fp32 (`_keep_in_fp32_modules`) to preserve input/output precision. |
| `minimax_h3.time_embedder.fc_in` / `fc_out` | Same fp32 set. |
| `minimax_h3.context_embedder` | Judgment call — see below. |
| norms, `rope`, `scale_shift_table`-style params | Not `LinearBase`; never candidates. |

`context_embedder` is excluded **by default** although H3 does not pin it to
fp32. It is the text input projection, structurally the same kind of module as
`proj_in` / `audio_proj_in` (which H3 *does* keep in fp32), and quantizing the
text conditioning stream while leaving the video/audio input streams in fp32 is
an asymmetry nobody has validated. Pass `include_context_embedder=True` to opt
in once there is evidence either way.

Note also that `adaln_proj.linear` **is** included (it is a real per-block
GEMM) while `adaln_basis` is not. They are different modules with similar
names; the exclusion list keys on `adaln_basis`, which is not a substring of
`adaln_proj.linear`.

## 5. `attn.to_gate_compress` must never be quantized

`MiniMaxH3Attention.to_gate_compress` is the VSA sparse-attention gate. Two
reasons it is special:

1. Its output decides **sparse routing** — which tiles the sparse attention
   attends to. That is a discrete decision. Quantization error there does not
   perturb an activation by a fraction of a percent; it can flip a routing
   decision outright, and the resulting error is not bounded by the weight
   quantization error.
2. H3's own deploy path explicitly ignores it, and the released checkpoint
   zero-initializes the gate, so the branch is exactly disabled until it is
   finetuned. There is nothing to gain by quantizing it.

The name matches none of the usual exclusion heuristics (it contains no
`norm`, no `scale_shift_table`, no `proj_*`), so it would be swept into any
broad suffix rule. It is therefore excluded by an **explicit, fail-closed deny
list**:

- `INT8AffineConfig.exclude_substrings` is always the union of the hard-coded
  never-quantize names and any caller-supplied list. The constructor can only
  *add* exclusions.
- The deny check short-circuits before both the `layer_suffixes` and
  `target_layers` paths, so even explicitly listing
  `minimax_h3.transformer_blocks.7.attn.to_gate_compress` in `target_layers`
  does not select it.

`test_to_gate_compress_is_excluded_even_by_a_broad_allowlist` in
`fastvideo/tests/ops/quantization/test_int8_affine_config.py` is the
regression guard for this; it exercises both escape routes.

## 6. Requirements

- **Dependencies:** none beyond PyTorch. No `flashinfer`, no CUDA kernels, no
  INT8 tensor cores. The module imports on a CPU-only host.
- **Hardware:** `get_min_capability()` returns 75 (Turing), matching
  `AbsMaxFP8Config`. The compute path is an ordinary bf16/fp32 GEMM, so this
  is a conservative floor rather than a real requirement.
- **Shape constraint:** `group_size` (default 64) must divide the weight's
  input dimension. H3 satisfies this everywhere (`hidden_size=5376`,
  `ffn_dim=14336`, and `2 * ffn_dim`, all divisible by 64, including under
  tensor parallelism at the sizes used). A violation raises `ValueError` at
  conversion rather than silently mis-grouping.
- **State dicts:** the quantized codes/scales/biases are non-persistent
  buffers. Nothing about this config changes checkpoint contents; you cannot
  save a "quantized H3" and must re-derive at load.
- **Training:** not supported by this config (see §2). Use a separate
  `*_qat_train`-style method if a recovery run is needed.

## 7. What is tested, and what is not

**Tested** (`pytest fastvideo/tests/ops/quantization/test_int8_affine_config.py`,
16 tests, CPU-only, no GPU needed):

- The module imports without CUDA or flashinfer.
- Quantizer parity: codes/scales/biases match an independently written
  restatement of the MLX algorithm, and — separately verified outside the
  suite — are **bit-identical** to `mlx_affine_quantize_reference` /
  `mlx_affine_dequantize_reference` on fp32, fp16 and bf16 inputs. The
  in-tree parity test skips (with a reason) while `mlx_affine_qat.py` is absent
  from this worktree, and activates automatically once it lands.
- Round-trip error on `N(0,1)` weights, measured over seeds 0-5 with ~2x
  headroom: `max|Δw| / max|w| ≤ 0.0055` and `rms|Δw| / max|w| ≤ 0.0013` for an
  fp32 source; `≤ 0.0093` and `≤ 0.0021` for a bf16 source.
- Layer selection for real H3 names: the include set above is selected, and
  `to_gate_compress`, `adaln_basis`, the fp32-pinned modules,
  `context_embedder`, norms and `rope` are not — including under a hostile
  broad suffix rule and via an explicit `target_layers` set.
- The enumerated H3 prefix set and the runtime suffix rule agree exactly.
- A real `ReplicatedLinear` mounts `INT8AffineQuantizeMethod` where expected
  and `UnquantizedLinearMethod` on the gate; conversion + `apply()` runs on
  CPU, and the purge path (`retain_original_weight=False`) works.

**NOT tested — do not assume any of this works:**

- **No model-level run.** Nothing here has been executed against a real H3
  checkpoint, on GPU or otherwise. The DiT has not been instantiated with this
  config, and no forward has produced a video.
- **No quality evidence.** No SSIM, no VLM adherence check, no comparison to
  the bf16 baseline. The measured error is *weight-level* round-trip error;
  how it propagates through 50 blocks of a diffusion DiT is unmeasured.
- **No performance numbers.** Memory, throughput and load-time cost are all
  unmeasured. In particular the dequantize-per-forward path in `apply()` has
  never been timed; it may be a significant overhead at inference.
- **Load-time peak memory is unmeasured.** Conversion makes an fp32 copy of
  each weight (`.detach().float().nan_to_num()`) one layer at a time. For
  H3's largest targeted weight that is a transient ~0.6 GB on top of the
  loaded model. It should be freed per layer, but this has not been profiled
  on a real load.
- **No multi-GPU / FSDP / TP validation.** The conversion walks
  `DTensor`-wrapped weights via `to_local()`, following `convert_model_to_nvfp4`,
  but has only been exercised on single-process CPU tensors. Whether quantizing
  a *shard* independently reproduces quantizing the whole weight depends on
  which dimension the shard is taken along and on the tensor-parallel degree —
  that has not been checked. If you enable this under FSDP or TP, verify that
  the shard's input dimension is still divisible by `group_size` (the
  conversion raises if it is not) and that scales agree with a single-process
  conversion.
- **The loader-hook dispatch is not wired by this change.** See §3; the
  sibling change to `_maybe_quantize_model` is required for the load-time
  (rather than lazy) conversion path.
- **`torch.compile` interaction is untested.** The lazy fallback in `apply()`
  mutates the layer on first call, which is not compile-friendly; the load-time
  conversion path is the one to use under compile.

## 8. Do you need a QAD / QAT recovery run for 4090 deployment?

Stated plainly, because this is a stated future use:

- **To run at all on a 4090: no.** The 4090 is sm89 with bf16 support and this
  config's compute path is a plain bf16 GEMM, so it runs as-is. No recovery
  run, no kernel build, no calibration data.
- **To preserve quality on a 4090: unknown, and this is the honest answer.**
  This is post-training quantization with no recovery step. Weight-only PTQ at
  8 bits with group-64 typically holds up better than activation-quantized
  schemes, but "typically" is not evidence, and nothing in this lane has been
  evaluated end-to-end. Treat a recovery run as *contingent on an eval gate*,
  not as a known requirement: run the existing quality gates against the bf16
  baseline first, and only if they regress does QAT/QAD become the next step.
- **If it does regress, a recovery run needs new code.** This config cannot be
  used for training (§2), and no INT8 analogue of `nvfp4_qat_train_config.py` /
  `fp8_qat_train_config.py` exists. The template is one of those files: a
  `QuantizeMethodBase` that keeps a trainable master weight and
  fake-quantizes with a straight-through estimator each forward. The
  fake-quantization itself already exists here — `int8_affine_quantize` +
  `int8_affine_dequantize` compose into exactly the STE the MLX lane's
  `fake_quantize_mlx_affine` performs, minus the `simulate_dtype` cast (which
  exists only because MLX loads checkpoints as fp16).
- **Also note:** the 4090 is a different target from the MLX lane, so a
  recovery run would not be transferable work-for-work with the Apple Silicon
  QAT effort — but they share the quantizer, so a checkpoint trained with the
  MLX QAT callback is quantized by the *same* decisions this config makes.

## 9. Files

| Path | What |
|---|---|
| `fastvideo/layers/quantization/int8_affine_config.py` | `INT8AffineConfig`, `INT8AffineQuantizeMethod`, `convert_model_to_int8_affine`, `int8_affine_quantize` / `int8_affine_dequantize`, `minimax_h3_int8_affine_prefixes` |
| `fastvideo/tests/ops/quantization/test_int8_affine_config.py` | CPU-only unit tests |
| `fastvideo/layers/quantization/__init__.py` | Registry entry for the name `INT8Affine` (owned elsewhere) |
| `fastvideo/models/loader/fsdp_load.py` | `_maybe_quantize_model` dispatch (owned elsewhere) |
