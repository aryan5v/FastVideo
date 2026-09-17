# MiniMax-H3 inference examples

Basic single-request H3 examples live in `examples/inference/basic/`
(`basic_minimax_h3_t2v.py`, `basic_minimax_h3_fl2va.py`,
`basic_minimax_h3_ref2va.py`). This directory holds H3-specific benchmark
tooling.

## `h3_vsa_dmd.py` — VSA-H3 vs dense attention, few-step DMD inference

Benchmarks 3-step (DMD-style) H3 T2VA inference under two attention
backends and prints a latency/speedup table:

- `dense` — `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN` with FA4
  (`FASTVIDEO_FA4=1`). If the flash-attn package is not installed the
  FLASH_ATTN request falls back to Torch SDPA (the worker log prints
  "Using Torch SDPA backend"); the baseline is then SDPA, not FA4.
- `vsa` — `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3` at
  `--sparsity` (default 0.9), applied at generator boot through
  `FastVideoArgs.VSA_sparsity` (`pipeline.experimental`).
  `--vsa-tile-size {64,256}` (default 256) flows the same way
  (`FastVideoArgs.VSA_tile_size`). At tile 256, `--vsa-kernel triton`
  (default, no optional dependencies) uses the 256-to-64 expansion path
  and `cutedsl` opts into the FA4 CuTe 256-tile forward (requires the
  optional FA4 CuTe build, `flash_attn.cute`); at tile 64 the forward is
  always the native 64-token Triton kernel and `--vsa-kernel` is ignored.
- `microbench` — model-free per-attention-layer proxy on the exact packed
  H3 sequence geometry (dense FA4/SDPA vs the full `MiniMaxH3VSAImpl`
  tile/pool/top-k/kernel/untile path). Useful standalone, and as the
  speedup proxy when the full VSA pipeline leg is unavailable.

Each mode boots its own generator in a fresh subprocess (the backend env
var is resolved at boot), runs `--warmup` untimed request(s), then times
`--num-prompts` requests with fixed seeds shared across modes so the
per-mode videos can be eyeballed against each other. Model-load time is
reported separately from per-request latency. A crash in one mode is
contained: its signature is saved to `<output>/<mode>/crash_signature.txt`
and the remaining modes still report.

```bash
FASTVIDEO_FA4=1 python examples/inference/minimax_h3/h3_vsa_dmd.py \
    --model-path /path/to/MiniMax-H3 \
    --prompts-json /path/to/validation.json \
    --num-prompts 4 \
    --output-dir outputs/h3_vsa_dmd \
    --modes dense,vsa,microbench \
    --dmd-steps 1000,667,333 \
    --num-gpus 4
```

`--prompts-json` expects `{"data": [{"caption": ...}]}`; without it a
built-in prompt set is used. Results land in
`<output>/<mode>/results.json`, per-mode videos in `<output>/<mode>/`, and
an aggregate `summary.json` plus a final table on stdout.

### Caveat: dense-trained checkpoints under VSA

With the base (dense-trained) H3 checkpoint this benchmark measures SPEED
only. The base model was never trained under VSA top-k masks, so at 90%
sparsity output-quality parity is not expected — judge quality with a
VSA-trained (sparse-student) DMD checkpoint. The 3-step DMD ladder applied
to the base checkpoint is likewise a latency proxy for a distilled
student, not a quality reference: real few-step quality requires a DMD
student checkpoint.
