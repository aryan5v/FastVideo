# MiniMax-H3 DMD2

Agent continuation starts at
[`HANDOFF-h3-dmd2-vsa.md`](../../../../../HANDOFF-h3-dmd2-vsa.md).

This is the current runbook and parity tracker for the MiniMax-H3 DMD2
experiment. Keep operational YAML and shell comments local and short; record
cross-repository rationale and open issues here.

## Comparison scope

FastGen currently provides the generic DMD2 method, MiniMax-H3 network and SFT
configuration, and DMD2 recipes for other model families. It does not provide a
MiniMax-H3 DMD2 experiment config. The current FastVideo recipe therefore
combines:

- FastGen's generic DMD2 loss and optimizer conventions;
- FastGen's MiniMax-H3 clock, precision policy, and H3 SFT timestep draw; and
- few-step choices adapted from the Wan and LTX DMD2 recipes.

Treat this as an implementation comparison, not a claim of exact H3 recipe
parity.

## Current recipe

The recommended config is `dmd2_sp1_fsdp40_vidprom_v7_vsa90.yaml` (v6 recipe
+ per-modality critic space + VSA-H3 student at 90% sparsity; teacher/critic
stay dense). `_v6` is retained as the dense-student recipe; the config
without a suffix is the earlier alternative. v7 is a fresh lineage: v6's
audio did not recover post-hoc from the global-x0 critic bug, and the VSA
student changes the attention contract — do not resume v6 checkpoints.
The current config uses a fresh `_v6_fp32_compute` run directory; do not point
it at checkpoints created before the cadence and FSDP precision changes.

| Area | Current setting |
|---|---|
| Data | VidProM text conditioning; data-free `simulate` rollout |
| Topology | SP=1, HSDP replicate=1, shard=world |
| Student ladder | base timesteps `[1000, 667, 333]`, no additional warp |
| Score draw | uniform base timestep over `[0.001, 0.999]` |
| H3 shifts | video 12, audio 3 |
| Critic target | per-modality space: video x0, audio velocity (global x0's sigma_a^2 weight blinded the critic on audio's low-noise axis; audio stalled by step 2500 of the first v6 run) |
| Optimizers | AdamW, LR `2e-6`, betas `(0.9, 0.999)`, weight decay `0.01` |
| Cadence | four critic-only iterations, then one student-only iteration |
| Gradient clip | 10 for both trainable roles |
| Precision | FP32 masters; BF16 blocks; FP32 input, timestep, and output groups |
| Validation | deterministic three-step Euler sampling by default |

The 40-GPU values in the YAML are a reference topology. The Slurm launcher
overrides `num_gpus` and HSDP dimensions from the actual allocation. That makes
the mesh consistent; it does not prove that every smaller allocation has enough
memory.

## Launch contract

The execution clone, virtual environment, YAML, and output directory must all
be visible from the compute nodes. Before each launch:

1. Verify the development and execution clones resolve to the same commit.
2. Pass the absolute execution-clone path to the `_v6` config.
3. Use SP=1, replicate=1, and shard=`4 * nodes` on four-GPU trays.
4. Confirm the output directory is either empty or contains checkpoints from
   this exact recipe. `resume_from_checkpoint: latest` is intentionally active.
5. Check checkpoint capacity and W&B credentials from the job's effective
   `HOME`.

Example:

```bash
NODES=10
REPO=/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo
CONFIG=$REPO/examples/train/configs/distribution_matching/minimax_h3/dmd2_sp1_fsdp40_vidprom_v6.yaml

REPO=$REPO CONFIG=$CONFIG LUSTRE_HOME=/mnt/lustre/vlm-wlsaidhi \
SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=$((4 * NODES)) \
  sbatch --nodes=$NODES -p hpc-rack-2 --requeue \
  "$REPO/examples/train/slurm/dmd2_32xgb200.sbatch"
```

Dotted CLI values appended by the launcher override YAML values; YAML values
override dataclass defaults. An explicit `CONFIG` also overrides the sbatch
default. A helper that explicitly names the non-v6 file will continue to run
that recipe even after the launcher default changes.

## Confirmed equivalences

- One base timestep maps to the H3 video/audio noise amounts with shifts 12/3.
- H3's raw inference output is `clean - noise`; the training adapter negates it
  at the boundary and exposes DMD's `noise - clean` convention.
- Video and audio losses and VSD normalizers are computed separately and then
  summed, so packed video element count does not mute audio.
- Scale-1 teacher guidance uses one conditional forward, matching the
  guidance-distilled H3 checkpoint.
- The three validation points are derived from the same unwarped base ladder as
  training.
- Student and critic optimization is mutually exclusive: interval 5 gives four
  critic steps followed by one student step.
- H3 input, timestep, and output projections have separate FP32 FSDP compute
  groups; predictions are converted back to latent dtype at the model boundary.

## Open differences and issues

| Priority | Status | Issue |
|---|---|---|
| P1 | Deliberate experiment difference | `simulate` builds stochastic student trajectories from noise. FastGen's multistep DMD path forward-noises real data at a sampled ladder point. A paired-latent A/B is required to isolate this difference. |
| P1 | Open if stochastic validation is used | `dmd_stochastic_renoise` is not a typed pipeline field and its hop noise uses the global RNG rather than the request generator. Default validation remains deterministic. |
| P2 | Open | The x0 critic objective estimates effective per-modality sigma-squared from already-rounded noised tensors. It is exact algebraically but biased at the lowest BF16 timesteps; direct `critic.predict_x0()` MSE would match FastGen more closely. |
| P1 | Resolved (2026-08-18, env overlay) | VSA-H3 training gradient explosion root-caused: the fastvideo_kernel 0.3.2 PyPI wheel ships the pre-95f4f547 Triton backward (bf16 K pre-scaling; error exp2-amplified by logit magnitude, so unit-scale tests pass while real activations explode — 1e7-1e9 in DMD2, ~30x in SFT). Fix: the repo's corrected block_sparse_attn_triton.py overlaid into the venv (see site-packages OVERLAY_NOTE.md); scale-sensitivity sweep on H3 packed geometry passed (grad ratios 1.000+-0.0002 across scales 1-32 at keep 0.5/0.2/0.1). Upstream 95f4f547 to public main and cut a fixed kernel release to retire the overlay. PR #1639 (FA4 CuTe backward, unmerged) is a later 3x-speed upgrade, opt-in via FASTVIDEO_VSA_CUTEDSL, and needs the out-of-place addcmul port at video_sparse_attn_h3.py:358. |
| P1 | Closed (2026-08-19): infra, not the VSA path — confirmed by positive test | The "inference-side VSA fault" (jobs 2307/2321) was an NCCL p2p transport connect failure — `transport/p2p.cc:288 Cuda failure 400 'invalid resource handle'` — at the boot's FIRST FSDP weight all-gather, which step-0 validation reaches before training does. With `hsdp_shard_dim=world` and straggler ranks still loading their validation pipelines, the all-gather could not have completed, so no transformer (hence no VSA) kernel had executed anywhere when the fault fired: malformed VSA metadata cannot have caused it. Both failures were on rack-2 tray sets (2307: 2-4/2-6/2-7/2-9/2-10/2-11/2-14; 2321: 2-2/2-9/...); every later run — VSA training 2384/2385/2392 and dense training + passing validation 2389 — ran on rack-3 and cleared its first all-gathers, and rack-2 trays have prior NCCL-fault history (memory 2026-08-13). The VSA-H3 inference metadata path itself passed a full repro at the exact validation shapes (768x1344x124, S≈37.8k, 56 heads: training/inference metadata parity, in-bounds incl. route-A 64-expansion, real-kernel 3-step ladder at 0.9 with tile-buffer reuse, gate 4-stack, dense parity max 2.4e-4, plus a 4-GPU FSDP2 staggered-validation pattern — `vsa_gate/valfix_repro.py`, job 2399, run on rack-2 tray 2-0). Guard added: `_validate_h3_tile_geometry` now fails synchronously on out-of-bounds tile geometry, so a real metadata bug can never surface as an unattributable async NCCL error again; regression test `fastvideo/tests/attention/test_vsa_h3_inference_metadata_parity.py`. CONFIRMED 2026-08-19: job 2392 resumed from checkpoint-500 on rack-3 with validation re-enabled and completed a full 64-prompt VSA-student validation round (validation now runs every 100 steps, sparse student evaluated sparse — the 8228b394 dense-eval mismatch is retired). |
| P2 | Deliberate omission | FastVideo has no DMD2 GAN/discriminator branch. FastGen's generic default is `0.001`; several Wan/LTX recipes use `0.03`. There is no H3 value to copy directly. |
| P3 | Accepted | FastVideo uses the pinned official H3 scheduler endpoint while FastGen's RF schedule is capped at `0.999`. The resulting ladder differences are small and should not be changed without output evidence. |
| P3 | Accepted | FastVideo clips both student and critic at 10; FastGen's default callback targets only the student. |

Resolved robustness items remain covered by unit tests: the VSD denominator is
computed in FP32 with a `1e-6` floor, and the uniform integer score sampler draws
inside its configured bounds instead of clamping out-of-range samples onto the
endpoints.

## Regional torch.compile (ported 2026-08-19, opt-in, v7 stays eager)

Port of the (still-open) upstream PR hao-ai-lab/FastVideo#1718:
`training.model.enable_torch_compile: true` regionally compiles the
`_compile_conditions` blocks (H3: main + refiner blocks) with
`fullgraph=True` + `emulate_precision_casts` after FSDP setup. Upstream
measured -27% steady-state step latency on a 4x GB200 Wan DMD2 run with
FA4. Nothing changes while the flag is off (the default); v7 production
configs do not set it.

H3-specific state and follow-ups before enabling it on a real run:

- A VSA-backed role (`VIDEO_SPARSE_ATTN[_H3]`) is auto-skipped with a
  warning (Triton kernels + SP all-to-alls + the sync metadata guard are
  not fullgraph-traceable); dense FLASH_ATTN/FA4 roles — the DMD2
  teacher/critic — are the compile candidates.
- H3 keeps its post-FSDP activation-checkpoint ordering (upstream moved
  Wan's AC pre-FSDP via `pre_fsdp_transform`). Compile then wraps the
  FSDP block forward inside the later AC wrapper — functional, but not
  the upstream-tested ordering. Before switching H3 to pre-FSDP AC:
  the loader's name-keyed paths are already AC-prefix-canonicalized, but
  the shard-cache load path and the loader's first-parameter dtype assert
  are not verified under AC-prefixed `named_parameters()`.
- Wan's compiled-modulation gradient corruption (fixed upstream with
  opaque modulation ops) is a warning shot: H3's factorized AdaLN has an
  analogous chunked-modulation pattern. Before trusting a compiled H3
  run, A/B critic/student grad norms against eager for one step, like the
  upstream PR did.
- Upstream PR #1718 is unmerged (CI/review pending); re-diff against the
  merged version when it lands.

## Verification

CPU contracts cover cadence, optimizer/resume selection, FP32 group policy,
autocast exclusion, output dtype, and cache compatibility. Before launching the
full checkpoint, run the gated two-GPU FSDP forward/backward test:

```bash
pytest -q tests/local_tests/models/test_fsdp_load_mixed_dtype.py \
  -k declared_fp32_group_distributed_forward_backward
```

Quality comparisons still require:

- direct-x0 versus sigma-squared critic loss and gradient parity in BF16; and
- matched-seed `simulate` versus paired-latent runs, with deterministic
  validation samples from the same checkpoint.
