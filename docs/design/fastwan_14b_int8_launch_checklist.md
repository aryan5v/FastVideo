# FastWan-14B-INT8 (Apple Silicon) — Launch Checklist

Mirrors the FastWan-QAD RTX 5090 launch shape. Status: 2026-08-03.

## 1. Checkpoint production (in flight)

- [x] Training branch + configs: `wan14b-qad-int8-gb200`
      (`dmd2_t2v_14b_mlx_int8{,_smoke}.yaml`), affine int8 QAT, v2 recipe mechanics.
- [x] Env gate: GB200 job 1090 PASSED (fake-quant roundtrip 0.00546 ==
      M5 survey value; both configs parse; DMD2/WanModel/wandb OK).
- [x] Dataset: `FastVideo/Wan-Syn_77x448x832_600k` (~1.6 TB) downloading to
      `/mnt/nfs/vlm-aryan/wan14b-qad-int8-20260803/data/`.
- [ ] Smoke: job 1091 (20 steps, 4×GB200, validation at 10/20, checkpoint at 20).
- [ ] Full: job 1092 (4000 steps, 16×GB200) — starts only on smoke success.
- [ ] EMA checkpoint exported to Diffusers layout (`dcp_to_diffusers`).
- [ ] MLX pre-quantized checkpoint produced (`--save-mlx-checkpoint`).

## 2. Quality gates (before any announcement)

- [ ] SSIM/LPIPS vs bf16 50-step teacher on the standard prompt set.
- [ ] SSIM vs the failed 14B **mxfp4** checkpoint (the head-to-head that
      justifies the format decision).
- [ ] VBench on the 14B INT8 student.
- [ ] Visual review of wandb validation videos across the run (100-step cadence).
- [ ] Seed SSIM references for `fastvideo/tests/ssim/` (Mac tier).

## 3. Runtime readiness

- [x] MLX runtime with int8 checkpoint load, on-device DMD sampling,
      `mx.compile`, TAEHV decode, RIFE fast mode, memory-tier presets.
- [x] Automatic prompt-embedding cache (`mlx-release-wins` branch; fold into
      the runtime PR or land with the 14B runtime update).
- [ ] 14B shape validated in the MLX runtime on M5 (24 GB) — the runtime is
      config-driven; verify DiT load + 3-step generate + decode.
- [ ] Sequenced residency (encode → free → denoise → free → decode) so the
      24 GB tier holds (`mlx_runtime_next_wins.md` #5 — launch-blocking).
- [ ] Bench table: M5 24/32/64 GB × {plain, RIFE×2} × {taehv, wan-vae}.

## 4. Release artifacts

- [ ] HF repo `FastVideo/FastWan-QAD-14B-INT8`: weights (diffusers + MLX
      checkpoint), model card with tier table + fast-mode usage.
- [ ] Blog post (draft: `docs/blog/draft_fastwan_14b_int8_apple_silicon.md`) —
      fill TODO(measure) numbers from §2/§3.
- [ ] `docs/inference/support_matrix.md` row: FastWan-14B-INT8 × Apple Silicon.
- [ ] Install/usage docs under `docs/getting_started/installation/mps.md`.
- [ ] Example: `examples/inference/basic/mlx_wan_prompt_to_video.py` flags
      verified against the 14B checkpoint.

## 5. Positioning notes (for the blog/social)

- Headline: 14B video generation fully local on a laptop-class Mac — the
  largest model shipped on Apple Silicon in the FastVideo family.
- The format story is the differentiator: measured int8 > mxfp8 (12.4×),
  int4 > nvfp4 > mxfp4; weight-only buys no speed on Metal, so int8 is a
  *memory* decision and we spend the bits on accuracy.
- QAD recipe continuity with the 5090 post: same DMD2 + QAT structure,
  different grid (affine int8 for Metal vs NVFP4 for Blackwell) — one recipe,
  two hardware-native artifacts.
- Apple Silicon tailwinds: M5 Neural Accelerators, MFA-class attention,
  macOS 26 / Metal 4, MLX 0.32.
