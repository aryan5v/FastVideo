# NVFP4 for MiniMax-H3

Load-time NVFP4 for the H3 DiT on Blackwell (sm100+).
Implementation: `fastvideo/layers/quantization/nvfp4_config.py` (FlashInfer).

```python
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

fastvideo_args.transformer_quant = NVFP4Config.for_minimax_h3()
```

Requires FlashInfer with NVFP4 support. Excludes `attn.to_gate_compress`.
Compact checkpoints use the NVFP4 sidecar helpers in the same module.

Tests: `fastvideo/tests/ops/quantization/test_nvfp4_h3_prefixes.py`,
`test_nvfp4_sidecar.py`.
