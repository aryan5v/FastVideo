# SPDX-License-Identifier: Apache-2.0
"""Cross-device fidelity gate: the GB200 FP4 attention emulator vs the real sm_120/121 sparse FP4 kernel.

Two steps, on two machines:

``capture`` (GB200, one GPU): runs the bf16 FastH3 OmniRef teacher on a few
OmniRef rows with the deployed sm_100a VSA-128 kernel and, at the chosen PDD
forwards and layers, saves the attention inputs exactly as the fine kernel
sees them (tile-ordered post-RoPE Q/K/V, the VSA tile mask and tile sizes),
plus the emulator's outputs for every numeric mode and the bf16 output::

    python fidelity_check.py capture --model-path <composed OmniRef dir> \\
        --eval-manifest <omniref-eval/manifest.json> --out captures/ --rows 2 --rungs 0 4 7 --layers 0 20 41 --heads 8

``compare`` (DGX Spark or RTX 5090/PRO 6000, needs ``attn_qat_infer``)::

    python fidelity_check.py compare --captures captures/ --report fidelity.json

checks that ``tile128_mask_to_fp4_blocks`` reproduces ``vsa_tile_mask_to_fp4_blocks``
exactly, runs ``sageattn_blackwell_sparse_bshd`` (single- and two-level P),
and reports rel-L2(emulator, kernel) per capture next to rel-L2(kernel, bf16).
Gate: rel-L2(emulator, kernel) <= 0.01 for the mode that will ship. The
``first_block_max_floor`` variants decide how the kernel's uninitialized
first-block ``AbsMaxP`` behaves in practice.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

GATE = 0.01
MODES = {
    "single": dict(two_level_p=False),
    "two_level": dict(two_level_p=True),
    "single_floor0": dict(two_level_p=False, first_block_max_floor=0.0),
    "two_level_floor0": dict(two_level_p=True, first_block_max_floor=0.0),
    "single_smoothk": dict(two_level_p=False, smooth_k=True),
    "two_level_smoothk": dict(two_level_p=True, smooth_k=True),
}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


# --------------------------------------------------------------------------- capture (GB200)
class _Recorder:
    """``fine_attention_override`` that records chosen calls and returns the deployed sm_100a output."""

    def __init__(self, layers: set[int], heads: int, out_dir: Path) -> None:
        self.layers, self.heads, self.out_dir = layers, heads, out_dir
        self.tag = ""
        self.row = ""

    def bind(self, layer: int):
        return lambda query, key, value, mask, metadata: self(query, key, value, mask, metadata, layer)

    def __call__(self, query, key, value, mask, metadata, layer: int):
        from fastvideo.attention.backends import video_sparse_attn_h3 as vsa
        q, k, v = (t.transpose(1, 2).contiguous() for t in (query, key, value))
        vbs = metadata.variable_block_sizes
        pad = vbs.numel() % 2
        if pad:  # the sm_100a kernel pairs tiles; add the zero partner as VSA-H3 does
            q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, 128)) for t in (q, k, v))
        kernel_mask = torch.nn.functional.pad(mask, (0, pad, 0, pad), value=False)
        kernel_vbs = torch.nn.functional.pad(vbs, (0, pad), value=0).to(torch.int32)
        q2k_idx, q2k_num = vsa.map_to_index(kernel_mask)
        out, _ = vsa._sm100a.block_sparse_attn_sm100a(q, k, v, q2k_idx, q2k_num, kernel_vbs, need_lse=False)
        out = out[:, :, :query.shape[1]]
        if self.tag and layer in self.layers:
            h = slice(0, self.heads)
            torch.save({"q": q[:, h, :query.shape[1]].cpu(), "k": k[:, h, :query.shape[1]].cpu(),
                        "v": v[:, h, :query.shape[1]].cpu(), "mask": mask[:, h].cpu(), "vbs": vbs.cpu(),
                        "bf16_out": out[:, h].cpu()}, self.out_dir / f"{self.tag}-layer{layer:02d}.pt")
        return out.transpose(1, 2)


def capture(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax_h3_nvfp4_decoder"))
    import generate_omniref_latents as gen

    from fastvideo.attention.backends.fp4_vsa_qat import load_qat_vsa_modules, vsa_impls

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [m for m in json.loads(Path(args.eval_manifest).read_text())
            if m["resolution"] == "480p" and m["case"] in args.cases][:args.rows]
    driver = gen.OmniRefLatentGenerator(argparse.Namespace(model_path=args.model_path, master_port=args.port))
    recorder = _Recorder(set(args.layers), args.heads, out_dir)
    transformer = driver.pipeline.get_module("transformer")
    for impl in vsa_impls(transformer):
        impl.fine_attention_override = recorder.bind(int(impl.layer_idx))
    stage = driver.denoise
    original_forward = transformer.forward

    def forward(*f_args: Any, **kwargs: Any) -> Any:
        from fastvideo.forward_context import get_forward_context
        rung = int(get_forward_context().current_timestep)
        recorder.tag = f"{recorder.row}-rung{rung}" if rung in args.rungs else ""
        return original_forward(*f_args, **kwargs)

    transformer.forward = forward
    for item in rows:
        recorder.row = item["id"][-24:]
        stage.forward(driver._batch(gen.read_row(item["parquet"]), item["seed"]), driver.fastvideo_args)
    transformer.forward = original_forward
    driver.shutdown()
    reference, train = load_qat_vsa_modules()
    for path in sorted(out_dir.glob("*-layer*.pt")):
        data = torch.load(path)
        idx, num, valid = reference.tile128_mask_to_fp4_blocks(data["mask"].cuda(), data["vbs"].cuda())
        q, k, v = (data[n].cuda() for n in "qkv")
        data["emulator"] = {mode: train.fp4_vsa_attn_qat(q, k, v, idx, num, valid, **kw).cpu()
                            for mode, kw in MODES.items()}
        torch.save(data, path)
        print(json.dumps({"capture": path.name, "fp4_vs_bf16": {m: round(_rel(o, data["bf16_out"]), 4)
                                                                for m, o in data["emulator"].items()}}), flush=True)


# --------------------------------------------------------------------------- compare (sm_12x)
def compare(args: argparse.Namespace) -> None:
    import attn_qat_infer.api as api

    from fastvideo.attention.backends.fp4_vsa_qat import load_qat_vsa_modules

    reference, train = load_qat_vsa_modules()
    results = []
    for path in sorted(Path(args.captures).glob("*-layer*.pt")):
        data = torch.load(path)
        mask, vbs = data["mask"].cuda(), data["vbs"].cuda()
        mine = reference.tile128_mask_to_fp4_blocks(mask, vbs)
        theirs = api.vsa_tile_mask_to_fp4_blocks(mask, 128, vbs.to(torch.int32), validate=True)
        lists_equal = all(torch.equal(a.int(), b.int()) for a, b in zip(mine, theirs[:3], strict=True))
        q, k, v = (data[n].cuda().transpose(1, 2).contiguous() for n in "qkv")  # BSHD for the kernel
        row = {"capture": path.name, "lists_equal": lists_equal}
        kernels: dict[tuple[bool, bool], torch.Tensor] = {}
        for mode, kwargs in MODES.items():
            two_level, smooth = kwargs.get("two_level_p", False), kwargs.get("smooth_k", False)
            if (two_level, smooth) not in kernels:
                # The bshd entry never smooths K itself; smooth exactly as the emulator does (input dtype, all rows).
                k_in = k - k.mean(dim=1, keepdim=True) if smooth else k
                kernels[(two_level, smooth)] = api.sageattn_blackwell_sparse_bshd(
                    q, k_in, v, *theirs[:3], theirs[3], single_level_p_quant=not two_level, validate=True)
            kernel = kernels[(two_level, smooth)]
            row[f"{mode}/kernel_vs_bf16"] = _rel(kernel.cpu(), data["bf16_out"])
            row[f"{mode}/gb200_emulator_vs_kernel"] = _rel(data["emulator"][mode], kernel.cpu())
            local = train.fp4_vsa_attn_qat(*(data[n].cuda() for n in "qkv"), *mine, **kwargs)
            row[f"{mode}/local_emulator_vs_kernel"] = _rel(local, kernel)
        results.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    summary = {key: max(r[key] for r in results) for key in results[0] if key.endswith("_vs_kernel")} if results else {}
    summary.update({f"mean/{key}": sum(r[key] for r in results) / len(results) for key in results[0]
                    if key.endswith("_vs_bf16")} if results else {})
    report = {"gate": GATE, "worst": summary, "passes": {k: v <= GATE for k, v in summary.items()},
              "lists_equal": all(r["lists_equal"] for r in results), "captures": results}
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps({"worst": summary, "lists_equal": report["lists_equal"]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--model-path", required=True)
    cap.add_argument("--eval-manifest", required=True)
    cap.add_argument("--out", required=True)
    cap.add_argument("--rows", type=int, default=2)
    cap.add_argument("--rungs", type=int, nargs="+", default=[0, 4, 7])
    cap.add_argument("--layers", type=int, nargs="+", default=[0, 20, 41])
    cap.add_argument("--heads", type=int, default=8)
    cap.add_argument("--cases", nargs="+", default=["first_frame", "first_last_frame", "storyboard"])
    cap.add_argument("--port", type=int, default=29700)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--captures", required=True)
    cmp_.add_argument("--report", required=True)
    args = parser.parse_args()
    os.environ.setdefault("FASTVIDEO_H3_VSA_FP4", "0")
    capture(args) if args.cmd == "capture" else compare(args)


if __name__ == "__main__":
    main()
