# SPDX-License-Identifier: Apache-2.0
"""Stage A gate for a FastH3 OmniRef NVFP4 PTQ export: teacher-forced error and free-running samples.

``omniref``: for each held-out row (``calibrate_omniref_nvfp4.heldout_plan``), run the bf16 pipeline
at seed s; inside every DiT call the same inputs and forward context (PDD head fusion, VSA metadata)
are replayed with the NVFP4 linears (``NVFP4Swap``), which is T1: per-forward velocity and x0
error on target video and target audio rows, free of trajectory divergence. The bf16 trajectory is
untouched, so its final latent is compared with the calibration job's bf16 seed-s latent
(run-to-run check). Then the NVFP4 model samples seed s freely (T3/T4 input). The first
``--attribution-rows`` rows per shard also replay layer-group subsets (which linears dominate).

``v2``: the same T1 on FastH3 V2 bf16 vs the V2 NVFP4-Consumer export (T2V prompts): the baseline.

``report``: per-rung T1 means, the ratio to V2, attribution, and Gate A (T1 part; T4 from
``omniref_t4_metrics.py report``).
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

from calibrate_omniref_nvfp4 import (add_plan_args, current_layout, generate_rows, heldout_name,  # noqa: E402
                                     heldout_plan, real_transformer, save_atomic)
from h3_nvfp4_swap import NVFP4Swap  # noqa: E402

GATE_T1_RATIO = 1.2


# --------------------------------------------------------------------------- arms
def attribution_arms(swap: NVFP4Swap) -> dict[str, set[str]]:
    """Layer-group subsets of the exported linears (each arm quantizes only its group)."""
    names = [name for name, *_ in swap.layers]

    def block(name: str) -> int:
        return int(name.split("transformer_blocks.", 1)[1].split(".", 1)[0])

    arms = {
        "ffn_in": {n for n in names if n.endswith("ff.fc_in")},
        "ffn_out": {n for n in names if n.endswith("ff.fc_out")},
        "ffn_out_late": {n for n in names if n.endswith("ff.fc_out") and block(n) >= 25},
        "attn_qkv": {n for n in names if n.rsplit(".", 1)[1] in ("to_q", "to_k", "to_v")},
        "attn_out": {n for n in names if n.endswith("attn.to_out")},
        "gate": {n for n in names if n.endswith("to_gate_compress")},
    }
    for start in range(0, 50, 10):
        arms[f"blocks_{start:02d}_{start + 9:02d}"] = {n for n in names if start <= block(n) < start + 10}
    return {k: v for k, v in arms.items() if v}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def compare(x_t: torch.Tensor, ref_v: torch.Tensor, q_v: torch.Tensor, sigma: float) -> dict[str, float]:
    """Velocity and flow-matching x0 (= x_t - sigma * v) error of ``q_v`` against ``ref_v``."""
    x0_ref, x0_q = x_t - sigma * ref_v, x_t - sigma * q_v
    return {"v_rel": _rel(q_v, ref_v), "x0_rel": _rel(x0_q, x0_ref), "x0_cos": _cos(x0_q, x0_ref)}


class TeacherForcing:
    """Wrap the DiT forward: return bf16, and replay each call under every NVFP4 arm."""

    def __init__(self, transformer: torch.nn.Module, swap: NVFP4Swap, stage: Any) -> None:
        self.transformer, self.swap, self.stage = transformer, swap, stage
        self.original = transformer.forward
        self.arms: dict[str, set[str] | None] = {"nvfp4": None}
        self.records: list[dict[str, Any]] | None = None
        transformer.forward = self._forward

    def remove(self) -> None:
        self.transformer.forward = self.original

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        from fastvideo.forward_context import get_forward_context

        reference = self.original(*args, **kwargs)
        if self.records is None:
            return reference
        index = int(get_forward_context().current_timestep)
        layout = current_layout()
        ncv = int(layout.num_condition_video_rows) if layout is not None else 0
        nca = int(layout.num_condition_audio_rows) if layout is not None else 0
        sigma_v = float(self.stage.scheduler.sigmas[index])
        sigma_a = float(self.stage.audio_scheduler.sigmas[index])
        x_v = kwargs["hidden_states"][0, ncv:].float()
        x_a = kwargs["audio_hidden_states"][0, nca:].float()
        ref_v, ref_a = reference[0][0, ncv:].float(), reference[1][0, nca:].float()
        record: dict[str, Any] = {"forward": index, "sigma_video": sigma_v, "sigma_audio": sigma_a}
        for arm, only in self.arms.items():
            with self.swap.enabled(only):
                out = self.original(*args, **kwargs)
            record[arm] = {"video": compare(x_v, ref_v, out[0][0, ncv:].float(), sigma_v),
                           "audio": compare(x_a, ref_a, out[1][0, nca:].float(), sigma_a)}
            del out
        self.records.append(record)
        return reference


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- omniref
def run_omniref(args: argparse.Namespace) -> None:
    from generate_omniref_latents import OmniRefLatentGenerator, read_row

    out = Path(args.output_dir)
    (out / "nvfp4").mkdir(parents=True, exist_ok=True)
    t1_path = out / f"t1-omniref-shard{args.shard:02d}.jsonl"
    done = {json.loads(line)["id"] for line in t1_path.read_text().splitlines()} if t1_path.exists() else set()
    plan = heldout_plan(args)[args.shard::args.num_shards]
    todo = [clip for clip in plan if clip["id"] not in done]
    print(json.dumps({"shard": args.shard, "rows": len(plan), "todo": len(todo)}), flush=True)
    if not todo:
        return
    driver = OmniRefLatentGenerator(args)
    transformer = real_transformer(driver)
    swap = NVFP4Swap(transformer, args.export, device=torch.device("cuda"))
    print(json.dumps({"nvfp4_linears": len(swap), "static_scales": swap.static_scales}), flush=True)
    forcing = TeacherForcing(transformer, swap, driver.denoise)
    arms = attribution_arms(swap)
    bf16_dir = Path(args.bf16_dir)
    for position, clip in enumerate(todo):
        start = time.perf_counter()
        row = read_row(clip["parquet"])
        attribution = position < args.attribution_rows and clip["resolution"] == "480p"
        forcing.arms = {"nvfp4": None, **(arms if attribution else {})}
        forcing.records = []
        try:
            bf16 = generate_rows(driver, row, clip["seed"])
        finally:
            records, forcing.records = forcing.records, None
        saved_path = bf16_dir / heldout_name(clip, "bf16_s0")
        rerun = None
        if saved_path.exists():
            saved = torch.load(saved_path, weights_only=False)
            rerun = {"video_rel": _rel(bf16["video"], saved["video"]), "audio_rel": _rel(bf16["audio"], saved["audio"])}
        with swap.enabled():
            nvfp4 = generate_rows(driver, row, clip["seed"], decode_audio=True)
        meta = {k: clip[k] for k in ("id", "source", "case", "resolution", "parquet", "plan_index")}
        save_atomic({**nvfp4, **meta, "seed": clip["seed"], "tag": "nvfp4_s0"},
                    out / "nvfp4" / heldout_name(clip, "nvfp4_s0"))
        append_jsonl(t1_path, {**meta, "seed": clip["seed"], "attribution": attribution, "bf16_rerun": rerun,
                               "final_video_rel_vs_bf16": _rel(nvfp4["video"], bf16["video"]),
                               "forwards": records})
        print(json.dumps({"id": clip["id"], "seconds": round(time.perf_counter() - start, 1), "x0_rel_video":
                          [round(r["nvfp4"]["video"]["x0_rel"], 4) for r in records], "bf16_rerun": rerun}),
              flush=True)
    forcing.remove()
    driver.shutdown()


# --------------------------------------------------------------------------- V2 baseline
def run_v2(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "minimax_h3_realtime"))
    from probe_speed_levers import build
    from generate_v2_latents import PROMPTS

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1_path = out / f"t1-v2-slot{args.shard:02d}.jsonl"
    done = {json.loads(line)["prompt"] for line in t1_path.read_text().splitlines()} if t1_path.exists() else set()
    prompts = [p for p in list(PROMPTS)[args.shard::args.num_shards] if p not in done]
    if not prompts:
        return
    build_args = argparse.Namespace(config=args.v2_config, model_path=args.model_path, slot=args.shard)
    generator, pipeline, _, request = build(build_args)
    sampling = dict(request["sampling"])

    def run(prompt: str, frames: int) -> None:
        generator.generate(request={"prompt": PROMPTS[prompt], "negative_prompt": "", "sampling": {
            **sampling, "seed": args.v2_seed, "width": 832, "height": 480, "num_frames": frames},
                                    "output": {"save_video": False, "return_frames": True}})

    run(prompts[0], args.v2_frames)  # post_init applies the DMD rungs; H3 accepts 124-362 frames
    stage = pipeline._stage_name_mapping["denoising_stage"]
    transformer = pipeline.get_module("transformer")
    transformer = transformer.materialize() if hasattr(transformer, "materialize") else transformer
    swap = NVFP4Swap(transformer, args.export, device=torch.device("cuda"))
    forcing = TeacherForcing(transformer, swap, stage)
    for prompt in prompts:
        forcing.records = []
        try:
            run(prompt, args.v2_frames)
        finally:
            records, forcing.records = forcing.records, None
        append_jsonl(t1_path, {"prompt": prompt, "seed": args.v2_seed, "num_frames": args.v2_frames,
                               "nvfp4_linears": len(swap), "static_scales": swap.static_scales, "forwards": records})
        print(json.dumps({"prompt": prompt, "x0_rel_video": [round(r["nvfp4"]["video"]["x0_rel"], 4)
                                                              for r in records]}), flush=True)
    forcing.remove()
    generator.shutdown()


# --------------------------------------------------------------------------- report
def _per_rung(rows: list[dict[str, Any]], arm: str, modality: str, metric: str) -> list[float]:
    by_rung: dict[int, list[float]] = {}
    for row in rows:
        for record in row["forwards"]:
            if arm in record:
                by_rung.setdefault(record["forward"], []).append(record[arm][modality][metric])
    return [float(np.mean(by_rung[k])) for k in sorted(by_rung)]


def _load(pattern: str, directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for path in sorted(directory.glob(pattern)) for line in path.read_text().splitlines()]


def report(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    omniref = _load("t1-omniref-shard*.jsonl", out)
    v2 = _load("t1-v2-slot*.jsonl", out)
    result: dict[str, Any] = {"omniref_rows": len(omniref), "v2_prompts": len(v2)}
    subsets = {"all": omniref, **{res: [r for r in omniref if r["resolution"] == res] for res in ("480p", "768p")}}
    for name, rows in subsets.items():
        if rows:
            result[f"omniref_{name}"] = {f"{mod}_{metric}": _per_rung(rows, "nvfp4", mod, metric)
                                         for mod in ("video", "audio") for metric in ("x0_rel", "v_rel", "x0_cos")}
    result["omniref_by_case_480p_x0_rel_video"] = {
        case: _per_rung([r for r in subsets["480p"] if r["case"] == case], "nvfp4", "video", "x0_rel")
        for case in sorted({r["case"] for r in subsets["480p"]})}
    if v2:
        result["v2"] = {f"{mod}_{metric}": _per_rung(v2, "nvfp4", mod, metric) for mod in ("video", "audio")
                        for metric in ("x0_rel", "v_rel", "x0_cos")}
        gate_rows = subsets["480p"] or omniref
        ours, base = _per_rung(gate_rows, "nvfp4", "video", "x0_rel"), result["v2"]["video_x0_rel"]
        ratios = [a / b for a, b in zip(ours, base, strict=True)]
        audio_ratios = [a / b for a, b in zip(_per_rung(gate_rows, "nvfp4", "audio", "x0_rel"),
                                              result["v2"]["audio_x0_rel"], strict=True)]
        result["gate_t1"] = {"threshold": GATE_T1_RATIO, "video_ratio_per_rung": [round(r, 3) for r in ratios],
                             "audio_ratio_per_rung": [round(r, 3) for r in audio_ratios],
                             "passed": all(r <= GATE_T1_RATIO for r in ratios)}
    attributed = [r for r in omniref if r.get("attribution")]
    if attributed:
        arms = sorted({arm for r in attributed for rec in r["forwards"] for arm in rec
                       if isinstance(rec[arm], dict) and "video" in rec[arm]})
        result["attribution_x0_rel_video"] = {
            arm: [round(v, 5) for v in _per_rung(attributed, arm, "video", "x0_rel")] for arm in arms}
        result["attribution_rows"] = len(attributed)
    reruns = [r["bf16_rerun"]["video_rel"] for r in omniref if r.get("bf16_rerun")]
    if reruns:
        result["bf16_rerun_final_video_rel"] = {"mean": float(np.mean(reruns)), "max": float(np.max(reruns))}
    result["final_video_rel_nvfp4_vs_bf16"] = float(np.mean([r["final_video_rel_vs_bf16"] for r in omniref]))
    (out / "t1_report.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result.get("gate_t1", {}), indent=None), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("omniref", "v2", "report"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="OmniRef composed model dir (omniref) or V2 bf16 dir (v2)")
    parser.add_argument("--export", help="packed nvfp4_weights.safetensors to swap in")
    parser.add_argument("--bf16-dir", help="calibration job's heldout_bf16 directory")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--attribution-rows", type=int, default=1, help="rows per shard with layer-group arms")
    parser.add_argument("--v2-config", default="examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml")
    parser.add_argument("--v2-seed", type=int, default=1234)
    parser.add_argument("--v2-frames", type=int, default=124)
    add_plan_args(parser)
    args = parser.parse_args()
    {"omniref": run_omniref, "v2": run_v2, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
