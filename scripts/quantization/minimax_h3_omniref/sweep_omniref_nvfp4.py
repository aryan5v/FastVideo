# SPDX-License-Identifier: Apache-2.0
"""Pick the OmniRef NVFP4 PTQ variant: activation clip rule per FFN layer and bf16 blocks for audio.

``local`` (calibration rows, one process per GPU):
  - for every FFN linear, the layer-output squared error of the NVFP4 path against bf16 for each
    candidate activation amax (calibrated max, per-row-amax p99 / p90, 0.7 / 0.5 / 0.35 / 0.25 x max,
    per-element p99.99 / p99.999 where the late fc_out histograms exist, and a per-call dynamic scale),
    split by token group, on balanced rows plus the rows that set the outlier maxima;
  - on ``--attr-rows`` rows per shard, teacher-forced "only block b quantized" arms: each block's share
    of the audio and video x0 error.
``choose``: per-layer MSE-optimal candidate, block ranking by audio error, and the variant arms.
``arms``: teacher-forced x0 error of every arm on other calibration rows.
``select``: per-rung ratios against the V2 baseline; picks the variant (fewest bf16 blocks among
those passing video and audio <= 1.2x V2, else the lowest worst ratio) and writes ``variant.json``.

Activation tokens of all modalities share each linear, so a separate audio scale is not possible;
audio can only be helped by the shared clip choice or by keeping linears in higher precision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "distill" / "minimax_h3_nvfp4_decoder"))

from calibrate_omniref_nvfp4 import add_plan_args, calibration_plan, current_layout, generate_rows, real_transformer  # noqa: E402
from eval_omniref_nvfp4 import TeacherForcing, _per_rung  # noqa: E402
from h3_amax_collector import GROUPS, amax_key, hist_percentile, is_ffn_key, token_groups  # noqa: E402
from h3_nvfp4_swap import INPUT_SF, FP4_AMAX_SCALE, NVFP4Swap, Variant  # noqa: E402

MULTIPLIERS = (0.7, 0.5, 0.35, 0.25)
GATE = 1.2
BF16_BLOCK_COUNTS = (2, 4, 8)


def module_name(key: str) -> str:
    block, sub = key.split(".", 1)
    return f"transformer_blocks.{block[1:]}.{sub}"


def candidate_table(calib_dir: Path) -> dict[str, dict[str, float | None]]:
    """Per FFN layer: candidate name -> amax (None = dynamic per call)."""
    merged = torch.load(calib_dir / "amax_merged.pt", weights_only=False)
    rows = merged["rows"]
    keys = sorted({k for k in rows[0]["amax"] if is_ffn_key(k)})
    table: dict[str, dict[str, float | None]] = {}
    for key in keys:
        per_row = np.array([max(row["amax"][key]) for row in rows])
        full = float(per_row.max())
        cands: dict[str, float | None] = {"max": full, "row_p99": float(np.percentile(per_row, 99)),
                                          "row_p90": float(np.percentile(per_row, 90)), "dynamic": None}
        cands.update({f"x{m:g}": full * m for m in MULTIPLIERS})
        if key in merged["hist"]:
            cands["elem_p99_99"] = hist_percentile(merged["hist"][key], 0.9999)
            cands["elem_p99_999"] = hist_percentile(merged["hist"][key], 0.99999)
        table[module_name(key)] = cands
    return table


def sweep_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(local-MSE rows, arm-comparison rows): disjoint calibration rows; local also gets the outlier rows."""
    plan = calibration_plan(args)
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for clip in plan:
        by_group.setdefault((clip["case"], clip["resolution"]), []).append(clip)
    local, arms = [], []
    for (case, res), clips in sorted(by_group.items()):
        n_local, n_arm = (args.local_480p, args.arm_480p) if res == "480p" else (args.local_768p, 0)
        local += clips[:n_local]
        arms += clips[n_local:n_local + n_arm]
    report = json.loads((Path(args.calib_dir) / "amax_report.json").read_text())["layers"]
    outlier_ids = {top["id"] for entry in report.values() if "rows" in entry and entry["rows"]["max_over_p99"] > 1.3
                   for top in entry["rows"]["top_rows"][:1]}
    taken = {clip["id"] for clip in local + arms}
    local += [clip for clip in plan if clip["id"] in outlier_ids and clip["id"] not in taken]
    return local, arms


class LocalError:
    """Forward hooks on the FFN linears: per-candidate output squared error by token group."""

    def __init__(self, swap: NVFP4Swap, table: dict[str, dict[str, float | None]]) -> None:
        self.table = table
        self.layers = {name: (module, method) for name, module, _, method in swap.layers if name in table}
        self.sse: dict[str, dict[str, torch.Tensor]] = {n: {} for n in self.layers}
        self.ref: dict[str, torch.Tensor] = {}
        self.active = False
        self._groups: tuple[int, torch.Tensor] | None = None
        self.handles = [module.register_forward_hook(self._hook(name)) for name, (module, _) in self.layers.items()]

    def groups(self, rows: int, device: torch.device) -> torch.Tensor | None:
        layout = current_layout()
        if layout is None or layout.sequence_length != rows:
            return None
        if self._groups is None or self._groups[0] != id(layout):
            ids = token_groups(layout.sequence_length, layout.text_indices.to(device), layout.video_indices.to(device),
                               layout.audio_indices.to(device), int(layout.num_condition_video_rows),
                               int(layout.num_condition_audio_rows))
            self._groups = (id(layout), torch.where(ids < 0, len(GROUPS), ids))
        return self._groups[1]

    def _by_group(self, per_token: torch.Tensor, groups: torch.Tensor | None) -> torch.Tensor:
        out = torch.zeros(len(GROUPS) + 1, dtype=torch.float64, device=per_token.device)
        if groups is None:
            out[-1] = per_token.sum()
            return out
        return out.scatter_add_(0, groups, per_token.double())

    def _hook(self, name: str):

        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.active:
                return
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            ref = (output[0] if isinstance(output, tuple) else output).reshape(x.shape[0], -1).float()
            groups = self.groups(x.shape[0], x.device)
            energy = self._by_group((ref**2).sum(-1), groups)
            self.ref[name] = energy if name not in self.ref else self.ref[name] + energy
            _, method = self.layers[name]
            saved = module._buffers.get(INPUT_SF)
            try:
                for cand, amax in self.table[name].items():
                    if amax is None:
                        module._buffers.pop(INPUT_SF, None)
                        method._dynamic_act_cached = True
                    else:
                        module._buffers[INPUT_SF] = torch.tensor(FP4_AMAX_SCALE / max(amax, 1e-12),
                                                                 dtype=torch.float32, device=x.device)
                        method._dynamic_act_cached = False
                    quant = method.apply(module, x).float()
                    err = self._by_group(((quant - ref)**2).sum(-1), groups)
                    store = self.sse[name]
                    store[cand] = err if cand not in store else store[cand] + err
            finally:
                method._dynamic_act_cached = False
                module._buffers[INPUT_SF] = saved

        return hook

    def state(self) -> dict[str, Any]:
        return {"sse": {n: {c: v.cpu() for c, v in d.items()} for n, d in self.sse.items()},
                "ref": {n: v.cpu() for n, v in self.ref.items()}, "groups": list(GROUPS) + ["ungrouped"]}


def build_driver(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, NVFP4Swap]:
    from generate_omniref_latents import OmniRefLatentGenerator

    driver = OmniRefLatentGenerator(args)
    transformer = real_transformer(driver)
    return driver, transformer, NVFP4Swap(transformer, args.export, device=torch.device("cuda"))


def block_arms(swap: NVFP4Swap) -> dict[str, tuple[set[str], None]]:
    arms: dict[str, set[str]] = {}
    for name, *_ in swap.layers:
        block = int(name.split("transformer_blocks.", 1)[1].split(".", 1)[0])
        arms.setdefault(f"block_{block:02d}", set()).add(name)
    return {arm: (names, None) for arm, names in arms.items()}


def run_local(args: argparse.Namespace) -> None:
    from generate_omniref_latents import read_row

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    local, _ = sweep_rows(args)
    mine = local[args.shard::args.num_shards]
    attr = sorted(mine, key=lambda c: (not c["case"].startswith("continue"), c["resolution"] != "480p"))[:args.attr_rows]
    attr = [c for c in attr if c["resolution"] == "480p"]
    driver, transformer, swap = build_driver(args)
    table = candidate_table(Path(args.calib_dir))
    local_error = LocalError(swap, table)
    forcing = TeacherForcing(transformer, swap, driver.denoise)
    forcing.arms = {"nvfp4": (None, None), **block_arms(swap)}
    attr_path = out / f"attr-shard{args.shard:02d}.jsonl"
    for clip in attr:
        start = time.perf_counter()
        forcing.records = []
        try:
            generate_rows(driver, read_row(clip["parquet"]), clip["seed"])
        finally:
            records, forcing.records = forcing.records, None
        with open(attr_path, "a") as handle:
            handle.write(json.dumps({**{k: clip[k] for k in ("id", "case", "resolution")}, "forwards": records}) + "\n")
        print(json.dumps({"attr": clip["id"], "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    for clip in mine:
        start = time.perf_counter()
        local_error.active = True
        try:
            generate_rows(driver, read_row(clip["parquet"]), clip["seed"])
        finally:
            local_error.active = False
        torch.save({**local_error.state(), "rows": [c["id"] for c in mine]}, out / f"local-shard{args.shard:02d}.pt")
        print(json.dumps({"local": clip["id"], "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    forcing.remove()
    driver.shutdown()


def choose(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    table = candidate_table(Path(args.calib_dir))
    states = [torch.load(p, weights_only=False) for p in sorted(out.glob("local-shard*.pt"))]
    sse: dict[str, dict[str, torch.Tensor]] = {}
    ref: dict[str, torch.Tensor] = {}
    for state in states:
        for name, cands in state["sse"].items():
            for cand, value in cands.items():
                sse.setdefault(name, {})[cand] = sse.get(name, {}).get(cand, 0) + value
            ref[name] = ref.get(name, 0) + state["ref"][name]
    audio = GROUPS.index("audio")
    layers: dict[str, Any] = {}
    mse_variant: dict[str, float] = {}
    mse_dynamic: set[str] = set()
    for name, cands in sse.items():
        rel = {c: float(v.sum() / ref[name].sum()) for c, v in cands.items()}
        rel_audio = {c: float(v[audio] / max(float(ref[name][audio]), 1e-30)) for c, v in cands.items()}
        static = {c: r for c, r in rel.items() if c != "dynamic"}
        best = min(static, key=static.get)
        layers[name] = {"rel_mse": rel, "rel_mse_audio": rel_audio, "best_static": best, "amax": table[name]}
        mse_variant[name] = table[name][best]
        if rel["dynamic"] < 0.9 * static[best]:
            mse_dynamic.add(name)
    attr = [json.loads(line) for p in sorted(out.glob("attr-shard*.jsonl")) for line in p.read_text().splitlines()]
    blocks = sorted({arm for r in attr for arm in r["forwards"][0] if arm.startswith("block_")})
    audio_by_block = {b: float(np.mean(_per_rung(attr, b, "audio", "x0_rel"))) for b in blocks}
    video_by_block = {b: float(np.mean(_per_rung(attr, b, "video", "x0_rel"))) for b in blocks}
    ranked = sorted(blocks, key=lambda b: -audio_by_block[b])

    def bf16_blocks(count: int) -> frozenset[str]:
        chosen = {int(b.split("_")[1]) for b in ranked[:count]}
        return frozenset(n for n in table if int(n.split(".")[1]) in chosen) | frozenset(
            f"transformer_blocks.{b}.attn.{p}" for b in chosen for p in ("to_q", "to_k", "to_v", "to_out",
                                                                       "to_gate_compress"))

    elem = {n: c["elem_p99_999"] for n, c in table.items() if "elem_p99_999" in c}
    row_p99 = {n: c["row_p99"] for n, c in table.items()}
    arms = {
        "max": Variant(),
        "row_p99": Variant(amax=row_p99),
        "elem_p99_999_late_fc_out": Variant(amax=elem),
        "mse": Variant(amax=mse_variant),
        "dynamic_ffn": Variant(dynamic=frozenset(table)),
        **{f"mse_bf16_top{k}": Variant(amax=mse_variant, bf16=bf16_blocks(k)) for k in BF16_BLOCK_COUNTS},
        "max_bf16_top4": Variant(bf16=bf16_blocks(4)),
    }
    (out / "arms.json").write_text(json.dumps({k: v.to_json() for k, v in arms.items()}, indent=1) + "\n")
    picks: dict[str, int] = {}
    for entry in layers.values():
        picks[entry["best_static"]] = picks.get(entry["best_static"], 0) + 1
    report = {"local_rows": sum(len(s["rows"]) for s in states[:1]), "best_static_counts": picks,
              "dynamic_much_better_layers": sorted(mse_dynamic), "audio_by_block": audio_by_block,
              "video_by_block": video_by_block, "blocks_by_audio": ranked[:12], "layers": layers}
    (out / "choose_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("best_static_counts", "blocks_by_audio")}), flush=True)


def run_arms(args: argparse.Namespace) -> None:
    from generate_omniref_latents import read_row

    out = Path(args.output_dir)
    _, rows = sweep_rows(args)
    mine = rows[args.shard::args.num_shards]
    path = out / f"arms-shard{args.shard:02d}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    arms = {name: (None, Variant.from_json(raw)) for name, raw in json.loads((out / "arms.json").read_text()).items()}
    driver, transformer, swap = build_driver(args)
    forcing = TeacherForcing(transformer, swap, driver.denoise)
    forcing.arms = arms
    for clip in [c for c in mine if c["id"] not in done]:
        start = time.perf_counter()
        forcing.records = []
        try:
            generate_rows(driver, read_row(clip["parquet"]), clip["seed"])
        finally:
            records, forcing.records = forcing.records, None
        with open(path, "a") as handle:
            handle.write(json.dumps({**{k: clip[k] for k in ("id", "case", "resolution")}, "forwards": records}) + "\n")
        print(json.dumps({"arms": clip["id"], "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    forcing.remove()
    driver.shutdown()


def select(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    rows = [json.loads(line) for p in sorted(out.glob("arms-shard*.jsonl")) for line in p.read_text().splitlines()]
    v2 = [json.loads(line) for p in sorted(Path(args.v2_dir).glob("t1-v2-slot*.jsonl")) for line in p.read_text().splitlines()]
    base_v, base_a = _per_rung(v2, "nvfp4", "video", "x0_rel"), _per_rung(v2, "nvfp4", "audio", "x0_rel")
    arms = json.loads((out / "arms.json").read_text())
    summary = {}
    for arm in arms:
        video, audio = _per_rung(rows, arm, "video", "x0_rel"), _per_rung(rows, arm, "audio", "x0_rel")
        rv = [a / b for a, b in zip(video, base_v, strict=True)]
        ra = [a / b for a, b in zip(audio, base_a, strict=True)]
        by_case = {case: [round(a / b, 3) for a, b in zip(_per_rung([r for r in rows if r["case"] == case], arm, "audio",
                                                                     "x0_rel"), base_a, strict=True)]
                   for case in sorted({r["case"] for r in rows})}
        summary[arm] = {"video_x0_rel": video, "audio_x0_rel": audio, "video_ratio_max": max(rv),
                        "audio_ratio_max": max(ra), "audio_ratio_by_case": by_case,
                        "bf16_linears": len(arms[arm]["bf16"]), "dynamic_linears": len(arms[arm]["dynamic"])}
    passing = [a for a, s in summary.items() if s["video_ratio_max"] <= GATE and s["audio_ratio_max"] <= GATE]
    if passing:
        best = min(passing, key=lambda a: (summary[a]["bf16_linears"], summary[a]["audio_ratio_max"]))
    else:
        best = min(summary, key=lambda a: max(summary[a]["video_ratio_max"], summary[a]["audio_ratio_max"]))
    (out / "variant.json").write_text(json.dumps(arms[best], indent=1) + "\n")
    (out / "sweep_report.json").write_text(json.dumps({"rows": len(rows), "selected": best, "passing": passing,
                                                       "arms": summary}, indent=1) + "\n")
    print(json.dumps({"selected": best, "passing": passing, **{a: [round(s["video_ratio_max"], 3),
                                                                     round(s["audio_ratio_max"], 3)]
                                                                 for a, s in summary.items()}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("local", "choose", "arms", "select"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--export")
    parser.add_argument("--v2-dir")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--local-480p", type=int, default=3, help="local-MSE rows per case at 480p")
    parser.add_argument("--local-768p", type=int, default=1)
    parser.add_argument("--arm-480p", type=int, default=4, help="arm-comparison rows per case at 480p")
    parser.add_argument("--attr-rows", type=int, default=1, help="per-block attribution rows per shard")
    add_plan_args(parser)
    args = parser.parse_args()
    {"local": run_local, "choose": choose, "arms": run_arms, "select": select}[args.mode](args)


if __name__ == "__main__":
    main()
