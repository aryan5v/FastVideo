#!/usr/bin/env python3
"""Spectral analysis of MiniMax-H3 AdaLN timestep conditioning."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

SPRINT = "/mnt/nfs/vlm-aryan/fasth3-14b-2step-qad-20260829"
M = "/mnt/nfs/vlm-aryan/fasth3-h3-serve-cookbook-eval-20260831/repo-main-3d8ac9d1"
HARNESS = f"{M}/examples/inference/basic/basic_fasth3.py"
OUT_DIR = Path(SPRINT) / "adaln_rank_analysis"

N_GRID = 4096
ENERGY_THRESHOLDS = (0.90, 0.95, 0.99, 0.999)
COVERAGE_K = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)
MAX_SIGMA_STORED = 1024
DEVICE = "cuda:0"
FP32_EPS = float(torch.finfo(torch.float32).eps)
N_MODALITY = 3   # MINIMAX_H3_MODALITY_NUM


def log(msg: str) -> None:
    print(f"[adaln-rank] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_fastvideo_args(model_path: str):
    spec = importlib.util.spec_from_file_location("fasth3_harness", HARNESS)
    harness = importlib.util.module_from_spec(spec)
    sys.modules["fasth3_harness"] = harness
    spec.loader.exec_module(harness)

    argv = [
        "--model-path", model_path,
        "--prompt", "adaln-rank-analysis",
        "--num-gpus", "1",
        "--no-fa4",
        "--no-inference-torch-compile",
        "--steps", "5",
    ]
    args = harness.parse_args(argv)
    args.fa4 = False
    harness.configure_environment(args)
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    config = harness.build_generator_config(args)
    config.pipeline.experimental["attention_backend"] = "TORCH_SDPA"
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    return generator_config_to_fastvideo_args(config)


def load_dit(fastvideo_args, transformer_path: str):
    from fastvideo.models.loader.component_loader import PipelineComponentLoader
    log(f"loading transformer from {transformer_path}")
    model = PipelineComponentLoader.load_module(
        module_name="transformer",
        component_model_path=transformer_path,
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    )
    log(f"loaded class={type(model).__name__}")
    return model


@contextlib.contextmanager
def on_device_fp32(module):
    """Temporarily put exactly this small module on GPU in fp32, then restore."""
    saved = [(name, p.dtype, p.device) for name, p in module.named_parameters()]
    module.to(device=DEVICE, dtype=torch.float32)
    try:
        yield module
    finally:
        params = dict(module.named_parameters())
        for name, dtype, device in saved:
            params[name].data = params[name].data.to(device=device, dtype=dtype)
        torch.cuda.empty_cache()


def scheduler_timesteps(shift: float, grid_points: int) -> torch.Tensor:
    """Verbatim replica of MiniMaxH3Scheduler.set_timesteps (scheduler file read)."""
    base = torch.linspace(1.0, 0.0, int(grid_points), dtype=torch.float32)
    sigma = shift * base / (1 + (shift - 1) * base)
    sigma = torch.unique_consecutive(sigma)
    return 1.0 - sigma[:-1]


def build_grids(video_shift: float, audio_shift: float):
    uniform = torch.linspace(0.0, 1.0, N_GRID, dtype=torch.float32)
    pieces = [torch.tensor([0.0, 0.999, 1.0], dtype=torch.float32)]
    for shift in (video_shift, audio_shift):
        for n in (4, 5, 49, 50):
            pieces.append(scheduler_timesteps(shift, n))
    ops = torch.unique(torch.cat(pieces)).sort().values
    control = torch.linspace(0.0, 1000.0, N_GRID, dtype=torch.float32)
    return uniform, ops, control


def spectral_stats(sigma: torch.Tensor, shape: tuple[int, int]) -> dict:
    s = sigma.detach().double().cpu()
    energy = s * s
    total = float(energy.sum())
    out: dict = {
        "shape": [int(shape[0]), int(shape[1])],
        "frobenius_norm": float(total ** 0.5),
        "s_max": float(s[0]),
        "s_min": float(s[-1]),
        "numerical_rank_fp32tol": int((s > s[0] * max(shape) * FP32_EPS).sum()),
        "stable_rank_trace_over_smax2": float(total / (float(s[0]) ** 2)),
    }
    for rel in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        out[f"rank_sigma_above_{rel:g}_of_smax"] = int((s > s[0] * rel).sum())
    p = energy / total
    nz = p > 0
    out["entropy_effective_rank"] = float(torch.exp(-(p[nz] * p[nz].log()).sum()))
    cum = torch.cumsum(energy, 0) / total
    for th in ENERGY_THRESHOLDS:
        k = int(torch.searchsorted(cum, th).item()) + 1
        out[f"rank_at_{th:.3f}_energy"] = k
        out[f"relerr_at_{th:.3f}_energy"] = float(max(0.0, 1.0 - float(cum[k - 1])) ** 0.5)
    return out


def gram_eigvals(G: torch.Tensor) -> torch.Tensor:
    """Descending singular values of the trajectory from its (T x T) Gram matrix."""
    lam = torch.linalg.eigvalsh(G.double())
    return torch.clamp(lam.flip(0), min=0.0).sqrt()


def analyze_blocks(model, mods, Z, H, store_sigma: bool, do_coverage: bool) -> tuple[dict, dict]:
    T = int(Z.shape[0])
    D = 6 * H * N_MODALITY
    per_block: dict = {}
    G_union = torch.zeros(T, T, dtype=torch.float32, device=DEVICE)
    G_union_norm = torch.zeros_like(G_union)
    G_mod_union = [torch.zeros_like(G_union) for _ in range(N_MODALITY)]
    G_list: list[torch.Tensor] = []

    for name, block in mods:
        t0 = time.time()
        proj = block.adaln_proj
        with on_device_fp32(proj) as mod:
            six = torch.stack([s.float() for s in mod(Z)], dim=0)     # (6, 3T, H)
        M = six.permute(1, 0, 2).reshape(T, N_MODALITY, 6, H).reshape(T, -1)
        del six
        M = M - M.mean(dim=0, keepdim=True)
        G = M @ M.t()
        G = 0.5 * (G + G.t())
        sv = gram_eigvals(G)
        stats = spectral_stats(sv, (T, D))
        stats["apply_silu"] = bool(getattr(proj, "apply_silu", None))
        stats["linear_weight_shape"] = list(proj.linear.weight.shape)
        stats["modulation_dim"] = D
        entry = {"stats": stats,
                 "frobenius_per_modality": [float(M[:, m * 6 * H:(m + 1) * 6 * H].double().norm())
                                            for m in range(N_MODALITY)]}
        if store_sigma:
            entry["sigma_top"] = [float(v) for v in sv[:MAX_SIGMA_STORED].cpu()]
            entry["sigma_stored"] = int(min(MAX_SIGMA_STORED, sv.numel()))
            entry["sigma_nonzero"] = int(sv.numel())
        per_block[name] = entry

        tr = float(torch.diagonal(G).sum())
        G_union += G
        if tr > 0:
            G_union_norm += G / tr
        for m in range(N_MODALITY):
            sl = M[:, m * 6 * H:(m + 1) * 6 * H]
            G_mod_union[m] += sl @ sl.t()
        if do_coverage:
            G_list.append(G)
        del M
        log(f"    {name}: rank90={stats['rank_at_0.900_energy']} rank99={stats['rank_at_0.990_energy']} "
            f"rank999={stats['rank_at_0.999_energy']} stable={stats['stable_rank_trace_over_smax2']:.2f} "
            f"s_max={stats['s_max']:.4g} ({time.time()-t0:.1f}s)")

    def quartiles(key):
        vals = sorted(pb["stats"][key] for pb in per_block.values())
        return [vals[0], vals[len(vals) // 2], vals[-1]]

    level_b = {
        "modulation_dim": D,
        "blocks": per_block,
        "summary": {
            "n_blocks": len(per_block),
            "rank90_min_median_max": quartiles("rank_at_0.900_energy"),
            "rank99_min_median_max": quartiles("rank_at_0.990_energy"),
            "rank999_min_median_max": quartiles("rank_at_0.999_energy"),
            "stable_rank_min_median_max": quartiles("stable_rank_trace_over_smax2"),
        },
    }

    cols = D * len(per_block)

    def union_stats(Gmat: torch.Tensor, label: str) -> dict:
        sv = gram_eigvals(Gmat)
        return {"label": label, "stats": spectral_stats(sv, (T, cols)),
                "sigma_top": [float(v) for v in sv[:MAX_SIGMA_STORED].cpu()]}

    level_c = {
        "energy_weighted": union_stats(G_union, "sum of centered block Gram matrices"),
        "block_normalized": union_stats(G_union_norm,
                                        "sum of trace-normalized centered block Gram matrices"),
        "per_modality_uniform_weighted": [
            {"modality": m,
             "stats": spectral_stats(gram_eigvals(G_mod_union[m]), (T, 6 * H * len(per_block)))}
            for m in range(N_MODALITY)
        ],
    }
    if do_coverage:
        Gmat = G_union
        lam, V = torch.linalg.eigh(Gmat.double())
        Vf = V.flip(1)[:, :max(COVERAGE_K)].float()
        rows = []
        for G in G_list:
            GV = G @ Vf
            d = (Vf * GV).sum(dim=0).double()
            rows.append((torch.cumsum(d, 0) / float(torch.diagonal(G).sum())).cpu())
        C = torch.stack(rows)
        level_c["coverage_k"] = list(COVERAGE_K)
        level_c["coverage_min_median_max_fraction_of_block_energy"] = {
            str(k): [float(C[:, k - 1].min()), float(C[:, k - 1].median()), float(C[:, k - 1].max())]
            for k in COVERAGE_K}
        level_c["coverage_per_block"] = [[float(x) for x in row] for row in C]
    return level_b, level_c


def measure(model, tag: str, ckpt: str, n_blocks_limit: int | None):
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3AdaLayerNormModulation

    log(f"adaln_rank={model.adaln_rank} hidden={model.hidden_size} "
        f"blocks={len(model.transformer_blocks)}")

    ckpt_dir = Path(ckpt)
    video_shift = float(json.loads((ckpt_dir / "scheduler" / "scheduler_config.json").read_text())["shift"])
    audio_shift = float(json.loads((ckpt_dir / "audio_scheduler" / "scheduler_config.json").read_text())["shift"])
    uniform, ops, control = build_grids(video_shift, audio_shift)
    grids = {"uniform_grid": uniform, "operating_points": ops, "control_0_1000": control}
    log(f"grids: uniform={tuple(uniform.shape)} ops={tuple(ops.shape)} "
        f"ops_range=({float(ops[0]):.4f},{float(ops[-1]):.4f}) "
        f"control={tuple(control.shape)} video_shift={video_shift} audio_shift={audio_shift}")

    H = int(model.hidden_size)

    result: dict = {
        "model_tag": tag,
        "checkpoint": str(ckpt),
        "adaln_rank": int(model.adaln_rank),
        "hidden_size": H,
        "num_layers": len(model.transformer_blocks),
        "num_refiner_layers": len(model.token_refiner.refiner_blocks),
        "scheduler_shifts": {"video": video_shift, "audio": audio_shift},
        "grids": {
            "uniform_grid": {"n": int(uniform.numel()), "range": [0.0, 1.0], "role": "primary"},
            "operating_points": {"n": int(ops.numel()), "values": [round(float(v), 6) for v in ops],
                                 "role": "the literal timesteps inference feeds"},
            "control_0_1000": {"n": int(control.numel()), "range": [0.0, 1000.0],
                               "role": "CONTROL ONLY -- not a convention this codebase uses"},
        },
        "precision_note": ("level A/B/C spectra are computed with the three AdaLN "
                           "modules cast to float32 in memory (checkpoint weights are "
                           "bf16); level_a_shared_coordinate.uniform_grid_bf16_native "
                           "repeats level A on the unmodified bf16 modules to expose "
                           "the storage-format noise floor."),
        "device": DEVICE,
    }

    def shared_coordinate(grid: torch.Tensor, native_bf16: bool = False) -> torch.Tensor:
        t = grid.to(DEVICE, dtype=torch.float32)
        if native_bf16:
            temb = model.time_proj(t)
            temb = model.time_embedder(temb.to(model.time_embedder.fc_in.weight.dtype))
            z, _ = model.adaln_basis(torch.nn.functional.silu(temb).to(model.adaln_basis.weight.dtype))
            return z.detach().float()
        with on_device_fp32(model.time_embedder) as te:
            temb = te(model.time_proj(t).to(te.fc_in.weight.dtype))
        with on_device_fp32(model.adaln_basis) as basis:
            z, _ = basis(torch.nn.functional.silu(temb).to(basis.weight.dtype))
        return z.detach().float()

    def level_a(Zmat: torch.Tensor) -> dict:
        Zc = Zmat - Zmat.mean(dim=0, keepdim=True)
        sv = torch.linalg.svdvals(Zc.double()).float()
        stats = spectral_stats(sv, tuple(Zc.shape))
        stats["centered_frobenius"] = float(Zc.double().norm())
        stats["raw_frobenius"] = float(Zmat.double().norm())
        stats["mean_row_norm"] = float(Zmat.mean(dim=0).double().norm())
        return {"sigma": [float(v) for v in sv.cpu()], "stats": stats}

    t0 = time.time()
    Zcache = {name: shared_coordinate(g) for name, g in grids.items()}
    log(f"shared coordinates done in {time.time()-t0:.1f}s  "
        f"{ {k: tuple(v.shape) for k, v in Zcache.items()} }")
    level_a_out = {name: level_a(Z) for name, Z in Zcache.items()}
    level_a_out["uniform_grid_bf16_native"] = level_a(shared_coordinate(uniform, native_bf16=True))
    result["level_a_shared_coordinate"] = level_a_out
    for name in level_a_out:
        log(f"LEVEL A {name}: {json.dumps(level_a_out[name]['stats'])}")

    mods: list[tuple[str, torch.nn.Module]] = [
        (f"transformer_blocks.{i}", b) for i, b in enumerate(model.transformer_blocks)]
    refiner_mods = [n for n, m in model.token_refiner.named_modules()
                    if isinstance(m, MiniMaxH3AdaLayerNormModulation)]
    result["refiner_adaln_modules"] = refiner_mods
    log(f"refiner AdaLN modules found: {refiner_mods or 'NONE'}")
    if n_blocks_limit is not None:
        mods = mods[:n_blocks_limit]
    log(f"per-block modulation dim = {6 * H * N_MODALITY} (6 x {H} x {N_MODALITY} modalities), "
        f"{len(mods)} blocks")

    result["level_b_per_block"] = {}
    result["level_c_shared_basis"] = {}
    for name, Z in Zcache.items():
        primary = name == "uniform_grid"
        log(f"  --- block analysis on {name} (T={Z.shape[0]}) ---")
        level_b, level_c = analyze_blocks(model, mods, Z, H, store_sigma=True, do_coverage=primary)
        result["level_b_per_block"][name] = level_b
        result["level_c_shared_basis"][name] = level_c
        log(f"LEVEL B {name} summary: {json.dumps(level_b['summary'])}")
        log(f"LEVEL C {name} weighted: {json.dumps(level_c['energy_weighted']['stats'])}")
        log(f"LEVEL C {name} normalized: {json.dumps(level_c['block_normalized']['stats'])}")
        if primary:
            log(f"LEVEL C {name} coverage: "
                f"{json.dumps(level_c['coverage_min_median_max_fraction_of_block_energy'])}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-blocks", type=int, default=None, help="debug: only the first N blocks")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fva = build_fastvideo_args(args.model_path)
    model = load_dit(fva, str(Path(args.model_path) / "transformer"))
    model.eval()

    result = measure(model, args.tag, args.model_path, args.max_blocks)
    out_path = out_dir / f"adaln_rank_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=1))
    log(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
