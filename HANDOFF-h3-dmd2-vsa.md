# Handoff: MiniMax-H3 DMD2

This is the entry point for continuing the H3 DMD2 work. Recipe rationale and
open comparisons live in the runbook; this file records repository state,
launch wiring, verification, and the next gate.

## Repository state

- Private remote: `hao-ai-lab/FastVideo-internal`, branch `h3-dmd`
- Core implementation commit: `aab5b23f`
- Development clone: `/home/vlm-wlsaidhi/FastVideo`
- Execution clone: `/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo`
- Current config: `dmd2_sp1_fsdp40_vidprom_v6.yaml`
- Current output/run namespace: `dmd2_sp1_vidprom_v6_fp32_compute`

Compute jobs read only the execution clone, its `.venv`, and absolute config
paths under Lustre. Sync the execution clone to the intended commit and verify
it is clean before submitting. Never resume the current config from an older
v6 output directory: both optimizer cadence and FSDP grouping changed.

## Context map

| Document | Purpose |
|---|---|
| [`h3_dmd.md`](examples/train/configs/distribution_matching/minimax_h3/h3_dmd.md) | Current recipe, comparison scope, accepted differences, and open issues |
| [DMD2 config README](examples/train/configs/distribution_matching/minimax_h3/README.md) | Current config inventory and launch entry points |
| [H3 SFT README](examples/train/configs/fine_tuning/minimax_h3/README.md) | SFT controls and negative memory results |
| [Training preflight lesson](.agents/lessons/2026-08-14_h3-dmd2-training-preflight-rules.md) | Durable precision, checkpoint, and execution-clone safeguards |
| [H3 validation README](tests/local_tests/minimax_h3/README.md) | Official-source and checkpoint parity commands |
| [H3 port status](tests/local_tests/minimax_h3/PORT_STATUS.md) | Component, pipeline, scheduler, and distributed-runtime evidence |
| [Generic DMD2 guide](docs/distillation/dmd.md) | User-facing DMD2 concepts and commands |
| [Training architecture](docs/design/training_architecture.md) | Modular method, role, callback, and checkpoint design |

## Current implementation contracts

- DMD2 updates are mutually exclusive. With interval 5, trainer steps 1–4
  update only the critic and step 5 updates only the student. Missing interval
  defaults to 5; non-positive values are rejected; interval 1 is student-only.
- Optimizer, scheduler, gradient clipping, EMA, and backward selection follow
  the active role. Resume initialization seeds both optimizer states.
- `StreamingLongTuningMethod` deliberately retains its combined
  generator-plus-critic iteration contract and overrides the role selectors.
- H3 keeps FP32 master weights. Transformer blocks use BF16 FSDP compute;
  `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, and
  `audio_proj_out` are separately sharded FP32 compute groups.
- The H3 training wrapper does not apply transformer-wide BF16 autocast.
  Video and audio predictions are converted back to their input latent dtypes.
- The shard cache keys and validates model-selected parameter dtypes, so a
  stale uniform cache cannot satisfy a mixed-dtype load.
- Existing SFT configs and the earlier DMD hyperparameter config explicitly
  retain uniform precision. The current v6 DMD config explicitly enables H3's
  mixed compute groups.

## Launch path

The production path is:

1. Capacity helper queues warmups and races the real Slurm submission.
2. `CONFIG` is the absolute v6 path in the execution clone.
3. [`dmd2_32xgb200.sbatch`](examples/train/slurm/dmd2_32xgb200.sbatch)
   derives world size and HSDP dimensions from the allocation.
4. [`run.sh`](examples/train/run.sh) launches one four-rank `torchrun` worker
   per tray.
5. `load_run_config()` applies dotted CLI overrides before constructing typed
   config objects and role models.

Launch-time precedence is dotted CLI override, then YAML, then dataclass
default. Submit from a Lustre-visible working directory; compute pods do not
see `/home`.

## Verification at handoff

- Focused cadence, H3, cache, EMA, streaming, and AnyFlow suite: 88 passed,
  3 hardware skips.
- Final H3/DMD/EMA/loader run: 49 passed, 2 GPU skips.
- All H3 YAML configs load through `load_run_config()` with their intended
  `uniform_parameter_dtype` value.
- Both launch scripts pass `bash -n`; `git diff --check` and the full
  pre-commit suite pass.
- No non-Markdown H3/DMD2 alignment comments refer to the comparison repo.

The real two-GPU FSDP test is implemented but was not executable on the local
non-CUDA machine:

```bash
pytest -q tests/local_tests/models/test_fsdp_load_mixed_dtype.py \
  -k declared_fp32_group_distributed_forward_backward
```

## Next gate

Before spending a full H3 allocation:

1. Run the two-GPU grouped-FSDP test above on CUDA.
2. Run one real-checkpoint H3 forward/backward and hook all five FP32 groups;
   require FP32 inputs/weights/outputs inside each group, BF16 block compute,
   latent-dtype final predictions, finite gradients, and rank consistency.
3. Run one critic iteration and one student iteration through the v6 DMD trio.
4. Only then submit the long job with the fresh v6 output namespace.

Quality experiments after the launch gate are tracked only in
[`h3_dmd.md`](examples/train/configs/distribution_matching/minimax_h3/h3_dmd.md).
