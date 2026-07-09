# Wan2.2-TI2V-5B Port Preparation

This directory is the required preparation handoff for a future Apple-Silicon
Wan2.2-TI2V-5B port. It contains internal MLX prototypes and local tests
preserved from the fork's Track C/D PR stack; it is not a released model,
training recipe, checkpoint converter, or public image-to-video API.

## Preparation contract

| Field | Current value |
|---|---|
| Model family | `wan2_2_ti2v_5b` |
| Planned workloads | T2V and I2V after separate approval |
| Official reference | Candidate artifact names exist in historical PRs, but no official source/revision/checksum is pinned; see Q001 |
| Official checkout | Not cloned |
| HF weights | Not selected or downloaded |
| Source layout | Unknown |
| Conversion required | Unknown |
| Official environment | Blocked on Q001, not an installation failure |
| FastVideo implementation | Internal MLX DiT/latent helpers only; no registry, pipeline, CLI, or public I2V surface |

Do not download weights, clone an unofficial reference, or create a production
loader until Q001 is resolved. If the chosen weights are gated, use one of
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`; never record its value.

## Preserved fork PR map

| Fork PR | Preserved work on this branch | Current gate |
|---|---|---|
| #4 | Causal MLX attention/DiT/sampler, compile regression, CUDA reference and QAD smoke fixtures | Internal self-forcing track; no public streaming example; needs pinned SFWan artifact, parity, and media-quality review |
| #6 | Hardware-tier recommendation and backend-agnostic tests | 5B auto-selection remains disabled until Q001 is pinned |
| #7 | Long-rollout cache injection and bounded-KV regressions | Real-weight stream skips until reviewed SFWan artifacts are staged |
| #8 | Per-token-timestep `MLXWan22DiT` and tiny parity test | Tiny implementation parity only; not a 5B release claim |
| #9 | Local-only real-weight parity scaffold, CUDA reference, and 5B benchmark | Requires exact source revision plus transformer SHA256 before the test activates |
| #10 | Latent-only I2V preparation and strict QAD arming scaffold | Image/VAE/mask parity is still blocked; arming gate never downloads artifacts |
| #11 | 48-channel TAEHV/Wan-VAE helpers, DMD sampler, and local/reference tests | Decoder/VAE artifacts require explicit reviewed paths; no public 5B generation example retained |

Historical benchmark values and example launch commands from those PRs are not
release evidence. Reproduce them only after the official artifact contract is
recorded below and all non-skip gates pass.

## Required parity activation

The placeholder test is deliberately a specific dependency skip: it activates
only after the official import path, revision, and weight layout are recorded
in `PORT_STATUS.md`. Its completed replacement must load real official weights
and a FastVideo-native component, use deterministic inputs, and compare actual
tensors. Shape-only checks, unconditional skips, and external upstream virtual
environments are not acceptable evidence.

| Gate | Planned evidence | Status |
|---|---|---|
| Transformer | Per-token timestep conditioning and text/I2V conditioning outputs | tiny local gate only; real gate blocked on Q001/Q002 |
| VAE | 48-channel latent shape and mean/std normalization encode/decode outputs | blocked on Q001/Q003 |
| Pipeline | T2V denoised-latent parity and I2V mask/image conditioning | I2V intentionally latent-only until component parity |
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
