# INT8 affine (group-64) for MiniMax-H3

Weight-only load-time INT8 for the H3 DiT on CUDA.
Implementation: `fastvideo/layers/quantization/int8_affine_config.py`.

```python
from fastvideo.layers.quantization.int8_affine_config import INT8AffineConfig

fastvideo_args.transformer_quant = INT8AffineConfig.for_minimax_h3()
```

Quantizes attention and FFN linears. Excludes `attn.to_gate_compress`,
`adaln_basis`, fp32-pinned I/O projections, and norms.

Inference only; not the MLX QAT callback. Tests:
`fastvideo/tests/ops/quantization/test_int8_affine_config.py`.
