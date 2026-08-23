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

The live recipe is `dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml`:
all-real video/audio latents from the five frozen native-shape sources at
global batch 64 (64 DP x accum 1), with no data-free carry. It starts a fresh
base-model and optimizer lineage; never resume v8/v9 into its output. The
full v10 contract and launch gates are below. The 32-GPU v10 config is retained
as the failed job-2960 receipt; it must not share the 64-GPU output namespace.
The previous
`dmd2_sp1_fsdp40_nuva_v9_dataforce_vsa64.yaml` remains the mixed data-forcing
plus carry recipe; `_v8_bwdsim_vsa64` is carry-only (data-free), `_v7_vsa90`
is the pre-parity 256-tile recipe, and `_v6` is the dense-student recipe.

### Gold-standard refresh (2026-08-21)

The reference tarball was refreshed (`fastgen.tar`, Aug 21; branch
`jberner/h3_new` @ `2c85efc8`, four commits past the v8 reference
`3429c171`). The H3 recipe knobs are unchanged — grid, draws, losses, lrs,
cadence, GAN-off, data-free — so v8's mapping below still holds verbatim.
What changed upstream, and our disposition:

- **Data-driven regime (ported as v9 `rollout_data_forcing`)**: FastGen's
  DMD2 has always had two regimes. `backward_simulation: false` (its
  default) forces the student onto real data: `t_student` is drawn
  uniformly over the sampling grid's rungs (`sample_from_t_list`, never
  t=0) and the real latents are forward-noised to that rung per modality
  shift; the critic still fits the student's generation, never real data.
  Their H3 config stays `backward_simulation: true` + text-only loaders
  (data-free); their Wan `config_dmd2_bwd_sim_vidprom` runs the carry
  *with* real data, but there real data feeds only the GAN branch. There
  is no per-batch mixing and no ratio knob upstream — the regime is
  config-global and the "ratio" is the dataset composition. v9's per-batch
  routing (latents present → forced, text-only → carried walk; walk pauses
  on forced batches) is our composition of the two regimes for mixed
  `data_path` loading.
- **Carry refactor (not ported)**: atomic carry overlay
  (`CARRY_OVERLAY_KEYS`), a boundary marker instead of an emptied slot,
  fail-loud carry-state validation, and an iteration-aware stagger
  (`(rank + completed_iterations) % steps`) so a resume recovers the phase
  an uninterrupted rank would occupy. Our v8 carry is behaviorally
  equivalent mid-run; the resume-phase refinement is a (small) behavior
  change with no knob, so it was left out to keep v8 byte-identical.
  Revisit if resume-phase clustering ever shows up in the rung histogram.
- **New opt-in knobs (not adopted — their H3 config does not set them)**:
  `fake_score_sample_t_cfg` (a separate critic-iteration noising-time
  density) and the `shifted_logitnormal` time-dist type; only the Wan
  AnyFlow on-policy recipe uses them.
- **AnyFlow** (flow-map student, in-iteration rollout, co-trained Stage-1
  loss on real batches): new Wan-only method, not H3, not ported.
- **GAN**: still off in their H3 config (`gan_loss_weight_gen 0.0`); ours
  has no GAN branch (see the deliberate omission below). v9 does NOT turn
  it on with the new real data.

### v9 data forcing

`method.rollout_data_forcing: true` (requires `rollout_carry`) routes each
batch by latent presence: a t2va parquet row trains the student at
`add_noise(real, eps, t)` with `t` drawn uniformly from
`dmd_denoising_steps` (exact `sample_from_t_list` semantics — the grid's
non-zero rungs — under the 12/3 modality shifts, i.e. the uncarried
`rollout_mode: data_latent` math); a text-only row advances the carried
walk. The slot's walk pauses untouched on forced batches. The one-time
stagger pre-walk still runs on each slot's first-ever call even when that
call is data-forced, keeping the FSDP collective count uniform across
ranks. Mixed loading is declared `preprocessed_data_type: t2va` (the
superset schema): text-only rows surface empty latent columns and route to
the walk; a half-present latent pair fails loudly. The realized mix is
observable as the mean of the `data_forced` metric; rebalance with
per-root `"path:N"` repeat counts in `data_path`.

### v8 ↔ FastGen h3_new mapping

| FastGen (gold standard) | Ours (v8) |
|---|---|
| `backward_simulation: true` + `CarryCallback` (per-accum-slot carry) | `method.rollout_carry: true`, `rollout_carry_slots = grad-accum steps`; carry lives on the method instance, transient across resumes |
| One generation forward per iteration; both phases advance the walk; staggered rank starts (uniform no-grad pre-walk) | Same semantics, offset `(rank·slots + slot) % 4` |
| `student_sample_steps: 4`, `t_list = f_12(linspace(0.999, 0, 5))`, `student_sample_type: ode` | `dmd_denoising_steps: [999, 749, 500, 250]` in base-t (identical noise levels; the adapter applies the 12/3 shifts), `rollout_sample_type: ode` |
| Score draw `time_dist_type: shifted`, shift 5.0 on video's clock | `score_timestep_shift: 2.4` (our knob draws `tau = f_{1/s}(U)`; `f_{1/2.4} = f_{5/12}` reproduces `t = f_5(U)` on the shift-12 clock exactly) |
| `fake_score_pred_type: x0` (both modalities; the shifted draw keeps sigma_audio in range) | `fake_score_loss_space: x0` (global — supersedes v7's `{video: x0, audio: velocity}` patch) |
| lr `1e-6` both roles; AdamW `(0.9, 0.999)`, wd `0.01`; student clip 1, critic unclipped | Same lrs/AdamW; clip `1.0` applies to whichever role steps (critic norms ~0.1, the clip is slack — accepted deviation) |
| `precision_fsdp: float32` | FP32 masters + FP32 module groups (already ours) |
| Prod shape 768×1344 @ 345 frames; validation every 50 | Kept ours: 768×1344 @ 124 frames; validation every 100 on held-out synth64, 4-step sampling (accepted deviations) |
| Dense student | VSA-H3 student, sparsity 0.9 at **64-token (4,4,4) tiles**, native Triton fwd+bwd (fixed-kernel overlay), no 256 remap; teacher/critic dense (our addition) |

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
| P1 | Addressed in v9 (2026-08-21) | `simulate` builds student trajectories from noise while FastGen's data-driven multistep path forward-noises real data at a sampled ladder point. v9's `rollout_data_forcing` runs both per batch over mixed data. The paired-latent A/B (forced-only vs walk-only at matched seeds) remains open for isolating the quality effect. |
| P1 | Open until launch | v9 pre-launch gates: (1) the NuVA t2va parquet under `/mnt/lustre/vlm-wlsaidhi/fastvideo/data/nuva_t2va/` must be complete and shaped 768x1344 @ 124 frames (num_latent_t 37; wrong shapes fail loudly at prepare/unpack) — and the root must contain ONLY production parquet: the loader sweeps `*.parquet` recursively, so scratch/harness subdirectories would be swept into training (and the `map_style_cache` pickle must be regenerated after any file change); (2) the regional-compile A/B verdict (`vsa_gate/compile_ab/VERDICT.md`) was not yet written when the v9 YAML was authored — follow its flip lines, and on NO-GO set `enable_torch_compile: false`; (3) confirm the realized latent-row fraction from the `data_forced` metric matches the intended mix. |
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

## v10 data-only native-shape launch (2026-08-22/23)

`dmd2_sp1_fsdp64_v10_dataonly_mixed_vsa64.yaml` starts a fresh lineage over
the five finalized shared T2VA sources. It has no simulated/carry rollout:
every microbatch forward-noises a real video/audio latent at a uniformly
sampled non-zero rung of the four-step grid. Sixteen four-GPU trays at local
batch 1 and accumulation 1 give global batch 64; student and critic learning
rates are both `2e-6`. Native-shape bucketing is mandatory. Validation uses
only `validation/heldout64.json`, honors each record's native spatial shape,
and logs the raw held-out video beside its generated counterpart. Raw
15-second references contain 362 frames; validation generation caps those
requests at 345 frames, the largest released `17*n+5` geometry within H3's
15-second inference ceiling. Shorter record lengths are unchanged, and the
full 362-frame reference remains intact for side-by-side logging.

The 64-rank bucket sampler pads each of the 90 exact-shape buckets to a multiple
of 64: 3,225 repeated rows over 56,832 scheduled rows (about 5.7% duplication),
up from 1,497 repeats at 32 ranks. With accumulation 1, every optimizer step is
one exact-shape global microbatch; it no longer combines two successive shape
buckets as the 32-GPU/accumulation-2 recipe did.

### First 32-GPU launch receipt and capacity failure

Production job `2960` started at 2026-08-22 17:08:12 UTC on
`hpc-rack-3-[2-9]`: eight Slinky trays, 32 GB200 GPUs, SP=1, and HSDP=(1,32).
It runs exact source SHA `9142b0249799d23c5959c85783346b2733f361b3` from the
dedicated `FastVideo-v10` execution clone. W&B run `48kti012` is online under
project `wlsaidhi/h3-dmd2-vsa`.

The launch repeated the Triton-backward, sm_100a, real odd-tile-route, and
grouped-FSDP gates before loading weights. Step-zero validation generated all
64 held-out records across the five sources; every output was readable with
the requested spatial shape, 24 fps, and audio. Four finite critic updates
were followed by the first finite student update at step 5
(`generator_loss=0.0399398`, `grad_norm/student=0.125461`). This is a fresh
lineage: the output namespace contained no checkpoint to resume.

Job 2960 later failed at completed step 153 while backpropagating the critic on
the second accumulated `1760x768-362f` microbatch. Full FSDP2 was active over
all 32 ranks (HSDP `(1,32)`); the failure was activation capacity, not missing
sharding or allocator fragmentation. Rank 0 had 173.60 GiB allocated, 2.01 GiB
free, and failed a 3.83 GiB allocation. The launch-time config still delayed
checkpointing until step 500, so the failed namespace has validation artifacts
but no checkpoint.

The 64-GPU recovery keeps SP=1 and expands FULL_SHARD to HSDP `(1,64)`. The
three resident FP32 roles contain 35.05B + 33.12B + 33.12B parameters; student
and critic (68.17B total) also own FP32 gradients and two Adam moments. Those
sharded persistent tensors account for about 35.6 GiB/rank at shard-32 and
17.8 GiB/rank at shard-64. Holding the failed activation/full-layer working set
constant therefore projects about 16 GiB free after satisfying the failed
3.83 GiB allocation. This is a useful capacity margin, but not an empirical
proof: activations and the per-block full-parameter all-gather are unchanged.
Before the production submit, run the exact `1760x768-362f` critic/student gate
with both Adam states pre-seeded on the final 64-rank commit; a fresh probe with
`resume_from_checkpoint: null` is insufficient because it omits the other
optimizer's mature state at the relevant backward peak.

Checkpointing now separates the two products that the modular trainer had
previously coupled. At every scheduled validation (step zero and then every
100 steps) it retains a pipeline-loadable bf16 student export under
`inference/checkpoint-<step>`; these form the unlimited, immutable inference
lineage. The step-100 start gate applies only to full FSDP
optimizer/scheduler/dataloader/RNG state under `checkpoint-<step>`, where only
the newest three are retained. At 4000 steps, 41 H3 student exports use about
2.6 TiB and three measured ~741 GiB training states use about 2.2 TiB.
Write-before-rotate peak is about 5.6 TiB, so preflight requires at least 6 TiB
free.

The non-submitting preflight is:

```bash
bash examples/train/slurm/prepare_h3_dmd2_v10_slinky.sh
```

It targets the dedicated execution clone `FastVideo-v10`, runs the native
data finalizer in `--verify-only` mode (including READY/manifests, parquet
schema/hash/buckets, and exact map-style cache), checks all 64 held-out raw
videos, requires the fresh fsdp64 output namespace, and prints but does not
execute the sixteen-tray `sbatch` command. It also binds the kernel receipt to
the final execution commit and requires at least 6 TiB free for the immutable
bf16 inference lineage plus keep-three resumable states and transient rotation
write. Rack-2 is the selected production lane. A cold Slinky topology can reject a direct
sixteen-node request even when the backing Kubernetes pool has capacity. Follow
`/home/vlm-wlsaidhi/ddnet-rl/SLURM_LAUNCH.md`: start enough one-node primer
jobs (the guide uses 26 demands for a 16-node target), wait until at least 16
are running, cancel only those exact primer IDs, and immediately race the real
sixteen-node submit.
Job 2960 was accepted on the second production submit after primers 2938 and
2944 warmed rack-3. `PARTITION=hpc-rack-3` remains an explicit operator
fallback.
The sbatch keeps `HOME` untouched on Slinky workers: `LUSTRE_HOME` seeds
dedicated HF/W&B/NETRC paths, while compiler caches use a job-scoped node-local
root under `/tmp`.

### v10 kernel and FA4 environment

Build the kernel only from the final execution commit:

```bash
REPO=/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo-v10 \
  bash scripts/train/rebuild_h3_v10_kernel.sh
```

The procedure requires merged PR #1719 (sm_100a forward) and #1730 (correct
Triton backward), uses the pinned CUDA-13/aarch64 toolchain, records the source
commit/kernel tree/retained-wheel hash/stable installed-prefix hash, and
atomically publishes
`/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix` while retaining the old
prefix as a backup. Production import order is exact:

```text
/mnt/lustre/vlm-wlsaidhi/fastvideo/v10_kernel/prefix:
/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay:
/mnt/lustre/vlm-wlsaidhi/fastvideo/fa4_overlay/nvidia_cutlass_dsl/python_packages
```

The new prefix must win: `fa4_overlay` contains a stale `fastvideo_kernel`
copy and exists only to supply `flash_attn.cute` plus its pinned CUTLASS DSL.
The older `vsa_gate/sm100a_main/prefix` must not be used because its Triton
backward predates #1730. The v10 submit helper exports this exact path and the
generic sbatch runs `gate_h3_v10_kernel.sh` on the head compute tray only when
`H3_V10_KERNEL_GATE=1`. That gate requires the receipt's source commit to equal
the execution HEAD, verifies the retained wheel and installed-prefix content
hashes, checks module provenance and kernel-tree identity, compares real
Triton-64 forward and dQ/dK/dV to FP32 dense attention across activation scales,
checks sm_100a against its reference, and proves the production H3 no-grad call
used sm_100a by making fallback to Triton fatal.

Audit of `integration/h3-vsa-fullgraph-all-20260822`: packed-varlen FA4 is
already carried by the v10 lineage as `9901ec342` (teacher no-grad forwards;
critic grad calls retain the established route). Kernel custom-op commit
`658c56d00` and regional graph commits `9432a87ee` / `37927079f` are for
compiled sparse inference and do not establish VSA training-backward parity,
so that inference stack is not imported into v10.

V10 enables the supported #1718 regional-training port (`863e87342` plus its
AC/cache/attention-policy fixes). The loader applies `fullgraph=True` and
`emulate_precision_casts=True` to all 52 repeated blocks of each dense
teacher/critic; its explicit safety policy logs that the VSA-H3 student stays
eager. This is not an all-three-role compile claim. The earlier fixed-shape,
8-GPU A/B measured roughly 4.7%/6.6% critic/student step improvement but used
SDPA rather than the production FA4 route and observed a `-24.7%` first-step
critic grad-norm difference at `+0.064%` loss. The rack-2 recovery must verify
FA4 selection, 52 compiled regions per dense role, the VSA eager fallback,
finite first critic/student updates, and max-shape memory before it is treated
as healthy; mixed-shape recompiles remain an observed launch metric rather
than an assumed MFU gain.

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

## MFU accounting (2026-08-19)

Calculator: `scripts/train/mfu_calc_minimax_h3.py` (a copy lives at
`/mnt/lustre/vlm-wlsaidhi/fastvideo/vsa_gate/mfu_calc.py`). Pure Python, no
GPU or fastvideo import; formulas and the DMD2 forward/backward census are
documented in its docstring.

Key assumptions (all verified against the code, not the launch notes):

- H3 DiT: 33.12 B params analytically (matches the 62 GiB bf16 transformer
  dir and the 741 GiB student+critic fp32 DCP save at 12 bytes/param/model);
  only the 19.27 B per-token GEMM params (50 x 385.35 M block QKV/out+SwiGLU)
  count toward linear FLOPs — the 13 B of per-(timestep,modality) AdaLN
  tables run on ~2 rows, not per token. Attention width is 7168 (56x128),
  not the 5376 hidden width.
- Packed S = 38,010 (text ~300 + audio 414 + video 37,296 at 768x1344x124).
- Full activation checkpointing: a grad-mode pass costs fwd + recompute +
  bwd = 4x forward FLOPs; a no-grad forward costs 1x.
- DMD2 cadence per micro-round (x accum 2): critic step = 3 no-grad VSA
  student fwd + dense critic grad unit = `3F_s + 4F_d`; student step =
  `6F_s + 2F_d` (rollout 2 no-grad + 1 grad fwd, one critic + one teacher
  no-grad dense fwd at guidance 1). Teacher/critic are always dense.
- VSA-H3 exempt mode at 0.9: video queries keep 18/180 video tiles + the 4
  prefix tiles; prefix queries stay dense; no `vsa_dense_layers` in training
  (that knob is inference-only). Effective token-pair keep 13.4% (tile-level
  0.139 incl. padding). The student's extra `to_gate_compress` GEMM (+1.93 B
  params) is counted in "actual" only.
- Peak: 2.25 PFLOP/s dense (non-sparse) BF16 per B200/GB200 GPU (NVIDIA
  spec). MFU scales as 1/peak — restate it if you assume a different number.
- Two conventions: **dense-equiv** counts the student's attention as if
  dense (speedup bragging); **actual** counts the realized sparse FLOPs
  (honest utilization).

Measured (W&B `step_time_sec` medians, split by `update_student`; step time
excludes validation/checkpointing but includes data loading and optimizer):

| Config | Runs | s/step critic / student | MFU dense-equiv (c / s / 4:1 blend) | MFU actual |
|---|---|---|---|---|
| v6 dense, 32 GPU, accum 1 | byg5ajyy, 4fek9itk, 9w6pdvnt | 51.2 / 57.8 | 21.5 / 21.8 / 21.6 % | same (dense) |
| v7 vsa90 Triton, 32 GPU, accum 2 | bmwvz3er (also 124lb3wh, 6b71zndg) | 83.7 / 71.3 | 26.3 / 35.3 / 27.9 % | 21.1 / 23.0 / 21.4 % |
| v7 vsa90 CuTe (projected) | leg-B gauntlet + SFT A/B | ~79.1 / ~62.1 | 27.8 / 40.5 / 29.9 % | 22.3 / 26.3 / 22.9 % |
| SFT vsa90 probe, 4 GPU SP4, Triton | gl5mrw23 | 4.59 | 34.2 % | 18.3 % |
| SFT vsa90 probe, 4 GPU SP4, CuTe | 3470j8b6 | 3.82 | 41.1 % | 22.0 % |

Reading: v6 and v7 sit at the same ~21.5% actual utilization — VSA-90
converts the saved attention FLOPs into 1.29x more samples/s (1.641 ->
1.268 s per sample-step on the blend) rather than higher hardware
utilization, as expected. Critic steps are bound by the dense critic grad
unit (4F_d of 7 units), so the CuTe flip mostly helps student steps. The
CuTe projection uses the 2026-08-19 gauntlet (leg B full-layer fwd+bwd at
the training shape: 45.9 ms CuTe vs 91.7 ms Triton => ~0.76 s saved per
student forward-equivalent at SP=1, cross-checked by the SFT A/B: 4.59 ->
3.82 s is 0.77 s saved per forward-equivalent).

After the CuTe flip, the one-command "after" measurement (pull the medians,
then compute):

```bash
HOME=/mnt/lustre/vlm-wlsaidhi \
  /mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo/.venv/bin/python - <<'EOF'
import statistics as st, wandb
run = [r for r in wandb.Api().runs("h3-dmd2-vsa")
       if r.name == "dmd2_sp1_vidprom_v7_vsa90"][-1]  # latest incarnation
rows = list(run.history(keys=["step_time_sec", "update_student"],
                        samples=10000, pandas=False))
crit = st.median(r["step_time_sec"] for r in rows if r["update_student"] < 0.5)
stud = st.median(r["step_time_sec"] for r in rows if r["update_student"] >= 0.5)
print(f"critic {crit:.2f}s student {stud:.2f}s")
EOF

python scripts/train/mfu_calc_minimax_h3.py --step-type dmd-blend \
  --step-time-critic <crit> --step-time-student <stud> \
  --gpus 32 --accum 2 --student-backend vsa --sparsity 0.9
```
