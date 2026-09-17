from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

SPRINT = SPRINT_ROOT
ANALYSIS = Path(SPRINT) / "adaln_rank_analysis"
SWEEP = ANALYSIS / "sweep"
FOLDS = SWEEP / "folds"

DMD2_CKPT = (f"{SPRINT}/runs/release20b-dmd2-v12-corrected-c4-parent750-32gpu-4000-v3"
             f"/job-paired8972-8975-4000-v3/inference/checkpoint-1400")

RANK_LIST = (768, 64, 16, 8)


def log(msg: str) -> None:
    print(f"[emit-folds] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_analysis_module():
    """Import adaln_lowrank.py by path (it is not on sys.path)."""
    path = ANALYSIS / "adaln_lowrank.py"
    spec = importlib.util.spec_from_file_location("adaln_lowrank", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adaln_lowrank"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DMD2_CKPT)
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANK_LIST))
    ap.add_argument("--out", default=str(FOLDS))
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    al = load_analysis_module()
    al.DEVICE = "cuda:0"

    fva = al.build_fastvideo_args(args.model_path)
    model = al.load_dit(fva, str(Path(args.model_path) / "transformer"))
    model.eval()
    model.to(al.DEVICE)

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, 1)

    aff = al.assert_affine_configuration(model)
    hidden = int(model.hidden_size)
    n_blocks = len(model.transformer_blocks)
    log(f"hidden={hidden} blocks={n_blocks} adaln_rank={model.adaln_rank}")
    log(f"apply_silu flags = {aff['apply_silu']}")
    for k, v in aff["shapes"].items():
        if k == "norm_out" or k.endswith(".0.adaln_proj"):
            log(f"  {k}: {v}")

    sites = al.adaln_sites(model)
    orig_basis_w = model.adaln_basis.weight.detach().float().clone()
    orig_basis_b = (model.adaln_basis.bias.detach().float().clone()
                    if model.adaln_basis.bias is not None else None)

    grid = torch.linspace(0.0, 1.0, al.N_GRID)
    with al.adaln_cast(model, torch.float32):
        U = al.compute_u(model, grid)
    log(f"U computed: shape={list(U.shape)} on {U.device}")
    coord_dim = int(U.shape[1])
    log(f"coord_dim={coord_dim}")

    # Deterministic fit: move to CPU and do the SVD in float64 there.  A cuda
    # SVD could in principle differ between ranks; the CPU path cannot.
    U_cpu = U.detach().cpu()
    U_centered_norm = float((U_cpu - U_cpu.mean(dim=0)).norm().item())

    summary = {
        "model_path": args.model_path,
        "n_grid": al.N_GRID,
        "grid": "linspace(0,1,4096)",
        "hidden_size": hidden,
        "coord_dim": coord_dim,
        "num_transformer_blocks": n_blocks,
        "adaln_params_total": int(sum(lin.weight.numel() + (lin.bias.numel() if lin.bias is not None else 0)
                                      for _n, _o, lin in sites)
                                  + model.adaln_basis.weight.numel()),
        "folds": {},
    }

    for rank in ranks:
        if rank >= coord_dim:
            mu = torch.zeros(coord_dim, dtype=torch.float64)
            V = torch.eye(coord_dim, dtype=torch.float64)
            fit = "identity (V_r = I, mu = 0): bit-identical folded weights"
            sigma = None
        else:
            mu64, V64, sigma64 = None, None, None
            mu_c, V_c, sigma_c = al.fit_basis(U_cpu, rank)   # CPU float64 SVD
            mu = mu_c.double()
            V = V_c.double()
            sigma = sigma_c
            fit = f"centered SVD of U - mu over linspace(0,1,{al.N_GRID}), cpu/float64"

        # fold_weights wants device tensors; it reads the live module weights.
        fold = al.fold_weights(model, V.float().to(al.DEVICE), mu.float().to(al.DEVICE))
        basis_w, basis_b, block_ws, block_bs, norm_w, norm_b = fold

        # Reconstruction receipt for this rank: compare every AdaLN projection
        # folded vs original at the grid times.  Correctness evidence only --
        # the rank decision is behavioral, not reconstruction-based.
        Z = (U_cpu - mu.float()) @ V.float()                       # (4096, r)
        z_ref_in = U_cpu
        rec = {"max_abs": 0.0, "site": None, "ref_max_abs": 0.0}
        rec_norm = {"max_abs": 0.0, "site": None, "ref_max_abs": 0.0}
        for idx, (name, _owner, lin) in enumerate(sites):
            W = lin.weight.detach().float().cpu()
            b = lin.bias.detach().float().cpu()
            if name == "norm_out":
                P, bp = norm_w, norm_b
            else:
                P, bp = block_ws[idx], block_bs[idx]
            ref = torch.nn.functional.linear(z_ref_in, W, b)
            comp = torch.nn.functional.linear(Z, P.detach().float().cpu(), bp.detach().float().cpu())
            d = (comp - ref).abs().max().item()
            rmax = ref.abs().max().item()
            if name == "norm_out":
                rec_norm = {"max_abs": d, "site": name, "ref_max_abs": rmax}
            elif d > rec["max_abs"]:
                rec = {"max_abs": d, "site": name, "ref_max_abs": rmax}

        payload = {
            "rank": rank,
            "hidden_size": hidden,
            "coord_dim": coord_dim,
            "num_transformer_blocks": n_blocks,
            "fit": fit,
            "storage_dtype": "float32",
            "model_adaln_dtype": str(model.adaln_basis.weight.dtype),
            "basis_weight": basis_w.detach().to(torch.float32).cpu().contiguous(),
            "basis_bias": basis_b.detach().to(torch.float32).cpu().contiguous(),
            "block_weights": torch.stack([w.detach().to(torch.float32).cpu().contiguous() for w in block_ws]),
            "block_biases": torch.stack([b.detach().to(torch.float32).cpu().contiguous() for b in block_bs]),
            "norm_out_weight": norm_w.detach().to(torch.float32).cpu().contiguous(),
            "norm_out_bias": norm_b.detach().to(torch.float32).cpu().contiguous(),
            "mu": mu.float().cpu(),
            "V_r": V.float().cpu(),
        }
        # Identity-check the fold against the live weights.  Only meaningful at
        # r >= hidden, where V_r = I makes the folded weights a copy; at lower
        # ranks the folded tensors are legitimately different objects.
        ident_err = None
        if rank >= coord_dim:
            ident_err = float((payload["basis_weight"] - orig_basis_w.cpu()).abs().max())
            blk_err = 0.0
            for idx, (_n, _o, lin) in enumerate(sites):
                if _n == "norm_out":
                    blk_err = max(blk_err, float((payload["norm_out_weight"]
                                                  - lin.weight.detach().float().cpu()).abs().max()))
                else:
                    blk_err = max(blk_err, float((payload["block_weights"][idx]
                                                  - lin.weight.detach().float().cpu()).abs().max()))
            ident_err = max(ident_err, blk_err)

        path = out_dir / f"fold_r{rank}.pt"
        torch.save(payload, path)
        n_params = (payload["basis_weight"].numel() + payload["block_weights"].numel()
                    + payload["norm_out_weight"].numel())
        size_gb = path.stat().st_size / 1e9
        summary["folds"][str(rank)] = {
            "path": str(path),
            "file_gb": size_gb,
            "adaln_params": int(n_params),
            "reconstruction_block_max_abs": rec,
            "reconstruction_norm_out_max_abs": rec_norm,
            "identity_fold_max_abs_vs_live_weights": ident_err,
            "storage_dtype": payload["storage_dtype"],
            "model_adaln_dtype": payload["model_adaln_dtype"],
            "fit": fit,
            "mu_norm": float(mu.norm().item()),
            "sigma_top5": ([float(x) for x in sigma[:5]] if sigma is not None else None),
            "sigma_tail5": ([float(x) for x in sigma[-5:]] if sigma is not None else None),
        }
        log(f"r={rank:4d} -> {path.name} ({size_gb:.2f} GB, {n_params/1e6:.1f}M adaln params, "
            f"model adaln dtype {payload['model_adaln_dtype']}) "
            f"max|comp-orig| block={rec['max_abs']:.4e} norm_out={rec_norm['max_abs']:.4e} "
            f"identity_vs_live={ident_err}")

    summary["U_centered_norm"] = U_centered_norm
    summary["rank_list"] = ranks
    (out_dir / "fold_meta.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_dir/'fold_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
