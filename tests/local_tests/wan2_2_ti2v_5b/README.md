# Wan2.2-TI2V-5B Port Preparation

This directory is the required preparation handoff for a future Apple-Silicon
Wan2.2-TI2V-5B port. It is not a model implementation, a training recipe, a
checkpoint converter, or a public image-to-video API.

## Preparation contract

| Field | Current value |
|---|---|
| Model family | `wan2_2_ti2v_5b` |
| Planned workloads | T2V and I2V after separate approval |
| Official reference | Not selected; see Q001 |
| Official checkout | Not cloned |
| HF weights | Not selected or downloaded |
| Source layout | Unknown |
| Conversion required | Unknown |
| Official environment | Blocked on Q001, not an installation failure |
| FastVideo implementation | Intentionally absent |

Do not download weights, clone an unofficial reference, or create a production
loader until Q001 is resolved. If the chosen weights are gated, use one of
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`; never record its value.

## Required parity activation

The placeholder test is deliberately a specific dependency skip: it activates
only after the official import path, revision, and weight layout are recorded
in `PORT_STATUS.md`. Its completed replacement must load real official weights
and a FastVideo-native component, use deterministic inputs, and compare actual
tensors. Shape-only checks, unconditional skips, and external upstream virtual
environments are not acceptable evidence.

| Gate | Planned evidence | Status |
|---|---|---|
| Transformer | Per-token timestep conditioning and text/I2V conditioning outputs | blocked on Q001/Q002 |
| VAE | 48-channel latent shape and mean/std normalization encode/decode outputs | blocked on Q001/Q003 |
| Pipeline | T2V and I2V mask/image conditioning denoised-latent parity | blocked on component parity |
| Conversion | Official to FastVideo state-dict strict load and pre-quantized MLX artifact round trip | blocked on source layout |
| Quality | Fixed prompt/image packet on a 32 GB+ Mac, valid MP4s, memory/latency, human review | blocked on implementation |

## Future-only operating gates

- A 5B implementation, model download, public I2V surface, GPU training run,
  model upload, and quality-reference upload each require explicit approval.
- The 32 GB+ Mac benchmark must report exact machine, macOS, MLX, PyTorch,
  peak memory, cold/steady latency, and generated-media validation.
- Self-forcing PRs #1307, #1042, and #814 remain independent blocked
  dependencies. They need their own causal runtime, KV-cache, training, and
  media-quality evidence before they can be considered here.
- PR #1563 is an open, blocked `FastWan FullAttn` configuration candidate. It
  has not been rebased or merged into this branch; reconsider it only with a
  fresh rebase and focused 5B configuration test.
- PRs #1557, #1496, #1488, #1494, and #1344 are explicitly out of scope.

## Commands once preparation is complete

```bash
pytest tests/local_tests/wan2_2_ti2v_5b/test_wan2_2_ti2v_5b_parity_scaffold.py -v -s
pytest tests/local_tests -k 'wan2_2_ti2v_5b and parity' -v -s
```
