"""Summarize ``FASTVIDEO_H3_PROFILE_DIR`` Chrome traces: GPU time per kernel category, busy union and idle.

  python profile_summary.py <trace dir> [--top 25]

Per rank trace (``forward-rank<N>.json``): the forward's GPU window (first to last kernel), the union of kernel
intervals (busy), idle = window - busy, and kernel time (sum of durations, streams may overlap) per category.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = (
    ("collective", re.compile(r"nccl|allgather|all_gather|alltoall|all_to_all|reduce_scatter", re.I)),
    ("attention", re.compile(r"flash|fmha|attn|attention|sm100a|vsa|softmax_?fwd|block_sparse|cute", re.I)),
    ("quantize", re.compile(r"quant|nvfp4|fp4_|fp8_|cvt_|swizzle|scale_?factor", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|nvjet|cublas|xmma|sm100_|sm90_|tcgen|matmul|splitk|mma", re.I)),
    ("norm_rope_elementwise", re.compile(r".*")),
)


def categorize(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(path: Path, top: int) -> dict:
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    if not kernels:
        raise SystemExit(f"{path}: no kernel events")
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
    window = max(b for _, b in intervals) - min(a for a, _ in intervals)
    busy = union_length(intervals)
    by_category: dict[str, float] = defaultdict(float)
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for event in kernels:
        by_category[categorize(event["name"])] += event["dur"]
        slot = by_name[event["name"]]
        slot[0] += event["dur"]
        slot[1] += 1
    biggest = sorted(by_name.items(), key=lambda item: -item[1][0])[:top]
    return {
        "file": path.name,
        "window_ms": round(window / 1e3, 1),
        "busy_ms": round(busy / 1e3, 1),
        "idle_ms": round((window - busy) / 1e3, 1),
        "kernel_ms": {k: round(v / 1e3, 1) for k, v in sorted(by_category.items(), key=lambda kv: -kv[1])},
        "top": [(name[:110], round(v[0] / 1e3, 1), v[1], categorize(name)) for name, v in biggest],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    for path in sorted(Path(args.directory).glob("forward-rank*.json")):
        report = summarize(path, args.top)
        print("RANK", json.dumps({k: v for k, v in report.items() if k != "top"}))
        if path.name.endswith("rank0.json"):
            for row in report["top"]:
                print("  TOP", row)


if __name__ == "__main__":
    main()
