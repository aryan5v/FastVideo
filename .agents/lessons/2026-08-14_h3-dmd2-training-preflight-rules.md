---
date: 2026-08-14
category: training
severity: critical
---

# Large-model training preflight

Keep recipe-specific settings, comparison notes, and open H3 DMD2 work in
[`h3_dmd.md`](../../examples/train/configs/distribution_matching/minimax_h3/h3_dmd.md).
The durable operational checks are:

- Use FP32 master weights for low-learning-rate training. The trainer rejects
  lower-precision trainable parameters unless the config explicitly opts in.
- Treat FSDP storage and compute precision separately. Model-declared precision
  boundaries need their own FSDP groups and an end-to-end dtype test.
- Resume only complete DCP checkpoints containing `dcp/.metadata`; optimizer
  state must be seeded for every optimizer before loading.
- Budget checkpoint size times retention against free space before launch.
- Use absolute paths from the execution clone. Compute pods may not see the
  development clone or its working directory.
- Verify the effective GPU mesh, credentials, output directory, and execution
  commit from inside the submitted job.
- Give changed recipes a fresh output directory so `latest` cannot load an
  incompatible training state.
