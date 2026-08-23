---
date: 2026-08-14
category: training
severity: critical
---

# Large-model training preflight

Current H3 DMD2 continuation:
[`HANDOFF-h3-dmd2-vsa.md`](../../HANDOFF-h3-dmd2-vsa.md).

Keep recipe-specific settings, comparison notes, and open H3 DMD2 work in
[`h3_dmd.md`](../../examples/train/configs/distribution_matching/minimax_h3/h3_dmd.md).
The durable operational checks are:

- Use FP32 master weights for low-learning-rate training. The trainer rejects
  lower-precision trainable parameters unless the config explicitly opts in.
- Treat FSDP storage and compute precision separately. Model-declared precision
  boundaries need their own FSDP groups and an end-to-end dtype test.
- Resume only complete DCP checkpoints containing `dcp/.metadata`; optimizer
  state must be seeded for every optimizer before loading.
- Budget checkpoint size times retention against free space before launch,
  including the transient write-then-rotate peak of one extra checkpoint
  (H3 DMD2 fp32 saves measured 741 GiB each; an ENOSPC-class short write
  killed a run mid-save on a 98%-full filesystem).
- Use absolute paths from the execution clone. Compute pods may not see the
  development clone or its working directory.
- Verify the effective GPU mesh, credentials, output directory, and execution
  commit from inside the submitted job.
- Give changed recipes a fresh output directory so `latest` cannot load an
  incompatible training state.
- Native-shape training must compile repeated dense regions with
  `torch_compile_kwargs.dynamic: true`. Static regional graphs specialize on
  packed sequence length and RoPE shape; with shared teacher/critic block code
  and FSDP grad/materialization variants, PyTorch's default eight-entry Dynamo
  cache can fail a healthy run after several new buckets. Set a bounded
  per-`torch.compile` recompile allowance in the recipe rather than mutating
  the process-global Dynamo setting, and gate more distinct calls than the
  configured limit before launch.
