# Metal 4 / macOS 27 / M5 optimization — tracking index

Single source of truth for the Apple-Silicon denoise-optimization experiment so
work doesn't get mixed up. Update this when a branch, module, or finding changes.

## Goal
Make the MLX Wan/QAD **denoise** path faster on Apple Silicon. Denoise is ~97% of
wall-clock. Build on M4 now; ship an M5 payoff as a one-command remote check.

## Branch map (keep these separate)
| Branch | Role | Touch? |
|---|---|---|
| `aryan/release/fastwan-qad-int8-1.3b-mlx` (PR #12) | **LAUNCH lane** — 1.3B QAD release | ❌ never |
| `aryan/mac-mlx-vae-decode` | MLX VAE/TAEHV decode + 5B sampler (base we branch from) | read-only |
| `aryan/mac-m5-metal4` | **integration branch** — all experiment work lands here | ✅ ours |
| `aryan/mac-m5-attn` | grok worktree for windowed attention (merged into integration) | grok |
| `aryan/mac-m5-*` (future) | one isolated grok worktree per task | grok |

Rule: grok always runs in a **dedicated `git worktree` + `--cwd`** (its `--worktree`
flag does NOT isolate in headless mode — it leaked into the launch branch once and
had to be reset). I merge grok branches into the integration branch.

## Modules & experiments (all on `aryan/mac-m5-metal4`)
| File | What | Status / key result |
|---|---|---|
| `fastvideo/mlx_runtime/quant_backends.py` | affine-int8 / mxfp8 / mxfp4 / nvfp4 quantized matmul | ✅ all 4 work on mlx 0.31.2; mxfp4 = 0.53 B/wt @ ~12% err |
| `fastvideo/mlx_runtime/accel_probe.py` | GEMM + attention microbench; M4-vs-M5 accel gate | ✅ M4 baseline captured |
| `fastvideo/mlx_runtime/windowed_attention.py` | chunked local attention (grok) | ✅ 7–21× at S=32760, quality TBD |
| `fastvideo/mlx_runtime/attn_scaling_probe.py` | full vs windowed sweep (grok) | ✅ O(S²) confirmed |
| `scripts/m5_validation.sh` | one-command runbook: probe + distilled QAD A/B | ✅ send to M5 friend |
| `docs/experiments/m5-*.md` | per-experiment write-ups | ✅ |

## Findings so far (M4 Max, mlx 0.31.2, macOS 27)
1. **Denoise is ~100% attention-bound**: one full attention call at S=32,760 = ~1.04s × 30 layers ≈ the whole 30s/step. FFN/linear GEMMs are noise.
2. **Do NOT hand-write a flash-attention kernel**: MLX fused SDPA already runs at 6.32 TFLOP/s ≥ the 5.97 TFLOP/s fp16 GEMM peak — already saturated.
3. **compile + fast_norm**: free ~6% (int8 32.6→30.5 s/step), no quality change.
4. **Weight quant = memory win, not speed** here (only touches the negligible FFN). Keep for fitting 5B.
5. **Windowed attention = 7–21× on the dominant cost** — but cosine error on *random* data is low (worst case). Real quality only knowable end-to-end → task #11.
6. **M5 is the hardware bet**: FLUX 4-bit is 3.8× faster M5-vs-M4; our denoise is the same compute-bound class. Open question the probe answers: does the M5 accelerate SDPA/attention specifically?

## How to run
- Local A/B (distilled model): `scripts/m5_validation.sh m4_max`
- Accel probe only: `python -m fastvideo.mlx_runtime.accel_probe --json bench/accel/<tag>.json`
- Attention sweep: `python -m fastvideo.mlx_runtime.attn_scaling_probe --json bench/accel/attn.json`
- M5 (friend): same `scripts/m5_validation.sh friend_m5_24gb`, then diff `accel.json` vs `bench/accel/m4_max.json`.

## Open tasks
- **#10** run `m5_validation.sh` on a borrowed 24GB M5 (whenever available).
- **#11** wire windowed attention into the DiT (self-attention only) + end-to-end ms_ssim/quality on the distilled QAD model → the make-or-break for windowing.
