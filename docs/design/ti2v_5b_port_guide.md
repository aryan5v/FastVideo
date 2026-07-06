# Guide: Wan2.2-TI2V-5B on the MLX Fast Lane (Track D)

For the Mac agent session (port) and the DGX agent (QAD run 6). Objective:
bring the second model family to the Mac fast lane — Wan2.2-TI2V-5B, which
buys two things at once: the quality tier for 32 GB+ Macs and **image-to-video**
(the same checkpoint does T2V and I2V). This is the roadmap's M6.

Researcher framing: the 1.3B proved the recipe; the 5B is where Mac hardware
differentiates. At INT8 the 5B DiT is ~5 GB of weights — trivial for 32 GB+
unified memory, impossible for 8–12 GB consumer NVIDIA cards. "The best
model you can run at home runs on a Mac" is the story this track earns.

## Architecture deltas vs the ported Wan2.1-T2V (read these files first)

From `fastvideo/configs/pipelines/wan.py::Wan2_2_TI2V_5B_Config` and the
`expand_timesteps` path in `fastvideo/models/dits/wanvideo.py`:

1. **Scale:** 24 heads × 128 head-dim (hidden 3072), 30 layers, ffn 14336
   (vs 12 heads/1536/30/8960). Pure config for the MLX runtime — the block
   math is identical. Watch INT8 group-size divisibility (all dims are
   multiples of 64 — verify at load, the capability probe will catch it).
2. **Per-token timestep conditioning (`expand_timesteps=True`).** Timestep
   is `[batch, seq_len]`, not `[batch]`. In the torch forward:
   `timestep_proj` unflattens to `(batch, seq, 6, inner)` and the block's
   modulation `e` is per-token; the output head takes
   `temb.dim() == 3` (see the `ts_seq_len` branches at
   `wanvideo.py::forward` and the block's `temb.dim() == 4` branch). The MLX
   `condition()`/block modulation/`output()` must grow the same branches.
   This is the main porting work; everything else is config.
3. **The VAE is different and I2V rides on it.** `z_dim=48` latents (vs 16),
   higher spatial compression, and TI2V conditions by *replacing the first
   latent frame with the encoded input image* (inspect
   `fastvideo/pipelines/basic/wan/` TI2V preset/stages for the exact
   conditioning — write down the mechanism in this doc during rung 1).
   Consequences: the MLX runtime's `in_channels` handling is already
   config-driven; decode uses the Wan2.2 VAE on torch-MPS (TAEHV's
   `taew2_1.pth` is for the 2.1 VAE — check whether a 2.2-compatible TAE
   exists upstream; if not, full VAE decode on MPS is the launch decode path
   and chunked/tiled decode matters for memory).
4. **I2V input path:** image → VAE encode (torch-MPS) → first-latent
   replacement → same DiT loop. No image cross-attention in TI2V-5B (no
   CLIP image embedder like Wan2.1-I2V-14B) — verify during rung 1; it is
   why this model is the right I2V entry point for the port.
5. **Sampling:** `FastWan2.2-TI2V-5B-FullAttn-Diffusers` is the already
   3-step-distilled FullAttn variant — the correct base for both the Mac
   runtime and the QAD student init (dense attention, matching our deploy
   path). flow_shift 5.0 (vs 8.0) — thread through the presets.

## Port ladder

1. **Rung 1 — architecture reconnaissance.** Diff the TI2V forward paths in
   `wanvideo.py` (all `ts_seq_len`/`expand_timesteps` branches), the TI2V
   pipeline stages, and the Wan2.2 VAE config. Append the findings (exact
   conditioning mechanism, any ops the MLX runtime lacks) as a table here.
2. **Rung 2 — MLX runtime generalization.** Extend `fastwan.py`'s
   `condition()`, block modulation, and `output()` for per-token timesteps
   (keep the 2.1 path intact; branch on config like torch does). Tiny-config
   parity fixtures: add a `TINY_ARCH_TI2V` variant to `tiny_wan.py` with
   `expand_timesteps=True` and extend the full-DiT parity test to both
   arches — this is the gate, runnable on CPU in CI like everything else.
3. **Rung 3 — real weights, T2V first.** Load
   `FastWan2.2-TI2V-5B-FullAttn`, full parity vs torch, then e2e T2V clips
   on the Mac (fp16 + int8 benchmark cells; expect ~5 GB INT8 weights,
   denoise cost ~4× the 1.3B per step at equal resolution — record actuals).
4. **Rung 4 — I2V.** Image encode on torch-MPS + first-latent conditioning
   in the MLX loop; benchmark I2V cells (motion7 prompts + a small image
   set; add an `i2v7` prompt/image set to the harness).
5. **Rung 5 — presets.** `mac-32gb`/`mac-64gb` presets point at the 5B;
   16 GB keeps the 1.3B. Decode strategy per tier decided by benchmark
   (see delta 3).

## QAD run 6 (DGX)

Recipe: clone the run-2 pattern onto the 5B — student + critic init from
`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`, teacher = base
`Wan2.2-TI2V-5B`, `mlx_qat` callback unchanged (it is architecture-agnostic;
the module-name excludes already cover Wan blocks), 3-step schedule from the
`FAST_WAN_2_2_TI2V_5B` preset, grad-accum 4. Open questions to resolve
before launch (rung-1 findings): dataset — check `examples/distill/`'s
Wan2.2-TI2V-5B recipes (`Data-free`, `crush_smol`) for what upstream used;
the data-free path may make this run *cheaper* to stage than run 2 was.

Prereq gate: rungs 2–3 green (per-token-timestep parity + real-weight
parity), plus the standard checks (EMA full-shape keys in first checkpoint,
QAT armed with the expected weight count — ~30 blocks × 10 + head ≈ 300+).
Cost estimate: ~3.5–4.5 days on 4×B200 (5B params, z_dim-48 latents mean
fewer tokens per frame at equal pixels — measure the smoke's s/step before
extrapolating; 8 GPUs halves it if the box is idle).

## Gates and deliverables

- [ ] Rung-1 findings table appended here (conditioning mechanism, VAE, ops gaps)
- [ ] Per-token-timestep support in `fastvideo/mlx_runtime/fastwan.py` + tiny-arch parity test (CI)
- [ ] Real-weight full parity + first T2V clips on Mac (fp16/int8 cells)
- [ ] I2V path + benchmark cells
- [ ] Preset updates + baseline-doc rows (32 GB/64 GB tiers)
- [ ] Run-6 handoff: recipe YAML + dataset decision + smoke checklist
