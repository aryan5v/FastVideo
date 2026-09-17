from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ENV_RANK = "FASTVIDEO_ADALN_RANK"
ENV_FOLD_DIR = "FASTVIDEO_ADALN_FOLD_DIR"

DEFAULT_FOLD_DIR = "" + SPRINT_ROOT + "/adaln_rank_analysis/sweep/folds"

_MARKER = "_fastvideo_adaln_rank_patch"


def log(msg: str) -> None:
    print(f"[adaln-rank-patch] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class FoldedLinear(torch.nn.Module):
    """Drop-in for ReplicatedLinear's unquantized forward: returns (out, None).

    Linear consumers in minimax_h3.py unpack as ``x, _ = self.linear(...)`` and
    take the input dtype from ``self.linear.weight.dtype``, so this must expose
    the same ``weight``/``bias`` Parameter names and the same 2-tuple return.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.to(dtype).contiguous())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(bias.to(dtype).contiguous())

    def forward(self, x: torch.Tensor):
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias), None


def load_fold(rank: int, fold_dir: str | None = None) -> dict:
    d = Path(fold_dir or os.environ.get(ENV_FOLD_DIR) or DEFAULT_FOLD_DIR)
    path = d / f"fold_r{rank}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing fold for rank {rank}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["rank"]) != int(rank):
        raise ValueError(f"fold file {path} declares rank {payload['rank']}, expected {rank}")
    return payload


def _find_transformer(pipeline):
    """Locate the DiT module in the pipeline, honouring LazyModule deferral."""
    mods = getattr(pipeline, "modules", None)
    if isinstance(mods, dict):
        for name in ("transformer", "transformer_refine", "transformer_2", "model", "dit"):
            if name in mods:
                return mods[name]
    # Fallback: the same tree search the quant export hook uses.
    import torch.nn as _nn

    def walk(obj, seen=None, depth=0):
        if depth > 6:
            return None
        seen = seen or set()
        if id(obj) in seen:
            return None
        seen.add(id(obj))
        if isinstance(obj, _nn.Module):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r = walk(v, seen, depth + 1)
                if r is not None:
                    return r
            return None
        for a in ("transformer", "model", "dit", "stages", "modules", "_stages", "_modules"):
            v = getattr(obj, a, None)
            if v is None:
                continue
            r = walk(v, seen, depth + 1)
            if r is not None:
                return r
        for v in getattr(obj, "__dict__", {}).values():
            r = walk(v, seen, depth + 1)
            if r is not None:
                return r
        return None

    return walk(pipeline)


def _is_lazy(obj) -> bool:
    return type(obj).__name__ == "LazyModule" and hasattr(obj, "set_materialize_transform")


def _apply_to_model(model, payload: dict, device: torch.device) -> dict:
    """Swap the AdaLN modules. Idempotent via _MARKER."""
    if getattr(model, _MARKER, None) is not None:
        return getattr(model, _MARKER)

    if getattr(model, "adaln_basis", None) is None:
        raise RuntimeError("model.adaln_basis is None: this checkpoint is not the rank-768 "
                           "rank-reduced configuration the fold was fit for")

    rank = int(payload["rank"])
    sites = []
    for index, block in enumerate(model.transformer_blocks):
        sites.append((f"transformer_blocks.{index}.adaln_proj", block, "linear"))
    sites.append(("norm_out", model, "linear"))

    # --- receipt: capture the ORIGINAL weights of one block + norm_out and the
    # basis, so the fold can be checked against the live modules in situ.
    probe_block = model.transformer_blocks[0]
    orig = {
        "basis_w": model.adaln_basis.weight.detach().float().clone(),
        "basis_b": (model.adaln_basis.bias.detach().float().clone()
                    if model.adaln_basis.bias is not None else None),
        "blk_w": probe_block.adaln_proj.linear.weight.detach().float().clone(),
        "blk_b": (probe_block.adaln_proj.linear.bias.detach().float().clone()
                  if probe_block.adaln_proj.linear.bias is not None else None),
        "out_w": model.norm_out.linear.weight.detach().float().clone(),
        "out_b": (model.norm_out.linear.bias.detach().float().clone()
                  if model.norm_out.linear.bias is not None else None),
    }

    target_dtype = model.adaln_basis.weight.dtype
    dev = model.adaln_basis.weight.device

    new_basis = FoldedLinear(payload["basis_weight"], payload["basis_bias"], target_dtype).to(dev)
    block_ws = payload["block_weights"]
    block_bs = payload["block_biases"]
    if block_ws.shape[0] != len(model.transformer_blocks):
        raise ValueError(f"fold has {block_ws.shape[0]} block weights, "
                         f"model has {len(model.transformer_blocks)} blocks")
    new_blocks = [
        FoldedLinear(block_ws[i], block_bs[i], target_dtype).to(dev)
        for i in range(len(model.transformer_blocks))
    ]
    new_out = FoldedLinear(payload["norm_out_weight"], payload["norm_out_bias"], target_dtype).to(dev)

    # --- the swap ---
    model.adaln_basis = new_basis
    for i, block in enumerate(model.transformer_blocks):
        block.adaln_proj.linear = new_blocks[i]
    model.norm_out.linear = new_out

    with torch.no_grad():
        v = probe_block.adaln_proj.linear.weight.dtype
        t_deployed = [0.0, 0.027027, 0.076923, 0.2, 0.000334, 0.1005, 0.25, 0.5]
        t_probe = torch.tensor(t_deployed, device=dev, dtype=torch.float32)
        temb = model.time_embedder(model.time_proj(t_probe).to(model.time_embedder.fc_in.weight.dtype))
        silu = F.silu(temb).to(torch.float32)
        u_orig = F.linear(silu, orig["basis_w"], orig["basis_b"])
        # folded basis: z = V_r^T (u - mu), applied to silu(temb) directly
        z_new, _ = new_basis(silu.to(new_basis.weight.dtype))

        ref_blk = F.linear(u_orig, orig["blk_w"], orig["blk_b"])
        # AFTER the swap these linears are FoldedLinear, so they consume the
        # rank-r coordinate z(t), NOT the rank-768 u(t).  Feeding u_orig here
        # is the bug the second smoke run caught.
        comp_blk, _ = probe_block.adaln_proj.linear(z_new.to(v))
        ref_out = F.linear(u_orig, orig["out_w"], orig["out_b"])
        comp_out, _ = model.norm_out.linear(z_new.to(v))

        blk_err = float((comp_blk.float() - ref_blk).abs().max())
        out_err = float((comp_out.float() - ref_out).abs().max())
        ref_scale = max(float(ref_blk.abs().max()), float(ref_out.abs().max()))
        z_err = float((z_new.float()
                       - (u_orig - payload["mu"].to(dev)) @ payload["V_r"].to(dev)).abs().max())
        blk_rel = blk_err / ref_scale if ref_scale else None

    receipt = {
        "rank": rank,
        "applied": True,
        "device": str(dev),
        "module_dtype": str(target_dtype),
        "num_blocks": len(model.transformer_blocks),
        "basis_weight_shape": list(new_basis.weight.shape),
        "block0_weight_shape": list(probe_block.adaln_proj.linear.weight.shape),
        "norm_out_weight_shape": list(model.norm_out.linear.weight.shape),
        "probe_max_abs_block0": blk_err,
        "probe_max_abs_norm_out": out_err,
        "probe_rel_block0": blk_rel,
        "probe_ref_scale": ref_scale,
        "probe_z_vs_VrTu_minus_mu": z_err,
        "probe_t": [float(x) for x in t_probe],
        "probe_note": "probe times are the deployed 4-call ladder (video shift 12, audio shift 3)",
    }
    setattr(model, _MARKER, receipt)
    return receipt


def install(pipeline, rank: int, fold_dir: str | None = None) -> dict:
    """Apply the rank-patch to the pipeline's transformer, on THIS rank.

    Returns a receipt dict.  Also registers a materialize transform when the
    component is deferred, so a release/reload cycle re-applies the patch.
    """
    payload = load_fold(rank, fold_dir)
    transformer = _find_transformer(pipeline)
    if transformer is None:
        raise RuntimeError("could not locate an nn.Module transformer in the pipeline")

    if _is_lazy(transformer):
        # NOTE: no local `import torch` here -- it would shadow the module-level
        # import for the whole function and make the else-branch reference an
        # unbound local (that exact bug killed the first smoke run).
        dev = torch.device("cuda", torch.cuda.current_device())

        def _transform(model):
            rec = _apply_to_model(model, payload, dev)
            log(f"lazy materialize transform applied: rank={rank} shapes="
                f"{rec['basis_weight_shape']}/{rec['block0_weight_shape']}")
            return model

        transformer.set_materialize_transform(_transform)
        model = transformer.materialize()
        log("materialized deferred transformer to apply the rank patch at startup")
    else:
        model = transformer
        dev = torch.device("cuda", torch.cuda.current_device())
        _apply_to_model(model, payload, dev)

    receipt = getattr(model, _MARKER, None) or {}
    return receipt


def maybe_install_from_env(pipeline) -> dict | None:
    raw = os.environ.get(ENV_RANK, "").strip()
    if not raw:
        return None
    rank = int(raw)
    rec = install(pipeline, rank, os.environ.get(ENV_FOLD_DIR))
    return rec
