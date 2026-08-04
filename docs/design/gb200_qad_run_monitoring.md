# Monitoring Runbook — 14B + 5B INT8 QAD runs (GB200 SLURM)

## GPU / cluster details

- Access: `ssh -i ~/.ssh/id_ed25519_nvidia vlm-aryan@nv-vllm-slinky-login-node`
- Cluster: 12 nodes × 4 NVIDIA GB200 (189 GB HBM each), SLURM partition `all`.
- Shared storage: `/mnt/nfs/vlm-aryan`; task dir `/mnt/nfs/vlm-aryan/wan14b-qad-int8-20260803` (`$T`).
- Constraints: no GPU work on the login node; never touch other users' jobs/files;
  store everything under `$T`; never print the SSH key or tokens.

## What is running

| Job | ID | Config | Resources | wandb run (project `distillation_wan`) |
|---|---|---|---|---|
| 14B full QAD | 1103 | `dmd2_t2v_14b_mlx_int8.yaml` — 4000 steps, affine int8 QAT, validation every 100 | 4 nodes / 16 GPUs | `wan2.1_14b_dmd2_3steps_mlx_int8` |
| 5B full QAD | 1105 (+ backup 1106) | `dmd2_t2v_5b_mlx_int8.yaml` — 4000 steps, affine int8 QAT, validation every 100 | 2 nodes / 8 GPUs | `wan2.2_5b_dmd2_3steps_mlx_int8` |

Expected wall-clock: 14B ~35–40 h; 5B ~40–45 h. Both passed their 20-step
smokes (jobs 1099, 1102) before the full runs launched.

## Agent prompt — monitor and maintain until completion

You are monitoring two QAD training runs on the GB200 SLURM cluster until they
finish. Work autonomously on recoverable failures; escalate only the cases
listed at the end.

Every ~30–60 min:

1. `ssh -i ~/.ssh/id_ed25519_nvidia vlm-aryan@nv-vllm-slinky-login-node 'squeue -u vlm-aryan -o "%.10i %.20j %.10T %.10M %R"'`
2. For finished/gone jobs: `sacct -j <id> --format=JobID,State,ExitCode,Elapsed -n`.
3. Tail logs: `$T/logs/full_1103.out` (14B) and `$T/logs/full5b_1105.out` (5B).
4. wandb (entity `aryan5v-san-jose-state-university`, project `distillation_wan`):
   confirm steps advancing, `total_loss`/`fake_score_loss` sane and not NaN,
   `grad_norm/student` not exploding (>50 sustained = problem), and validation
   videos appearing every 100 steps.

Known failure modes and the established recoveries (all scripts under `$T`):

- **CUDA OOM in backward reduce-scatter**: the resident validation pipeline is
  the usual straw. The full configs keep validation (it fits at 8/16 GPUs);
  if a run OOMs anyway, resubmit with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (already set) and, if it recurs, raise validation `every_steps` beyond the
  run or report — do NOT silently disable validation in the full runs.
- **EMA shape mismatch (`size of tensor a ... must match ... tensor b`)**:
  fixed in the branch (`training_utils.py` update re-inits on drift). If it
  appears, the job is running a stale checkout — the inner scripts re-checkout
  `wan14b-qad-int8-gb200` at start; resubmit.
- **NCCL init failure (`unhandled cuda error`, `scontrol: command not found`)**:
  transient/multi-node. Resubmit the same sbatch; the rendezvous uses an
  NFS master file, not scontrol. If the same node fails twice, resubmit with
  `--exclude=<node>`.
- **Job PENDING for hours**: normal — another tenant holds most nodes. Do not
  cancel our jobs to force them in; just wait.
- **Checkpoint/resume**: runs checkpoint every 100 steps to
  `$T/repo/outputs/<run>/checkpoint-N`. If a run dies mid-way, resubmit with
  `--training.checkpoint.resume_from_checkpoint <path>` appended to the
  torchrun args in the relevant `*_inner.sh`, then submit.

Completion criteria (both must hold): `full ... PASSED` in the log AND
`checkpoint-4000` present in `$T/repo/outputs/<run>/` AND the wandb run shows
`finished` with validation videos through step 4000.

Escalate to the user (stop and report, do not improvise): loss NaN/divergence,
sustained grad_norm explosion, repeated OOM after the recoveries above,
validation videos that look broken past step ~500, or any need to change the
recipe (LR, steps, mesh, QAT grid).

Report cadence: note status (steps done, loss, ETA) after each check; on
completion, record final metrics and checkpoint paths.

## Current known state (2026-08-04)

- 1103 (14B): loading models at last check; watch for first training steps.
- 1105 (5B): had a transient NCCL init error on hpc-rack-2-4; torchrun elastic
  retried. 1106 is the queued backup — if 1105 dies, let 1106 run; do not
  submit a third copy.
