# SPDX-License-Identifier: Apache-2.0
"""Static NVFP4 activation-scale calibration for the MiniMax-H3 DiT: an input amax collector.

Forward pre-hooks on each main block's FFN linears (``ff.fc_in`` = diffusers ``ff.net.0.proj``,
``ff.fc_out`` = ``ff.net.2``) record the running max of ``|x|`` over every forward of a row,
split by the packed token group (text incl. Qwen3-VL vision tokens, reference video, target
video, reference audio, target audio). The attention inputs (``attn.to_q``, shared by
``to_k``/``to_v``/``to_gate_compress``, and ``attn.to_out``) are recorded for reporting only.

``runtime_amax_table`` returns the ``FASTVIDEO_NVFP4_ACT_AMAX`` / converter ``--act-amax`` JSON,
``{"b<block>.ff.fc_in": amax, ...}`` (FFN keys only: the Consumer layout keeps attention on the
unit scale). For late ``ff.fc_out`` layers the collector also keeps the global top-k magnitudes and
a log2 histogram, so a percentile or MSE-optimal clip can be chosen later (plan step C3).
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import torch

FFN_SUBS = ("ff.fc_in", "ff.fc_out")
REPORT_SUBS = ("attn.to_q", "attn.to_out")
GROUPS = ("text", "ref_video", "video", "ref_audio", "audio")
TAIL_SUB = "ff.fc_out"
TAIL_MIN_BLOCK = 25
TOPK = 256
HIST_MIN_LOG2, HIST_MAX_LOG2, HIST_BINS = -24.0, 24.0, 384  # 1/8-octave bins up to 1.7e7
HIST_CHUNK_ROWS = 8192
_BLOCK_MODULE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.(.+)$")


def amax_key(module_name: str) -> str | None:
    """``transformer_blocks.3.ff.fc_in`` -> ``b3.ff.fc_in`` (the runtime/converter key); None elsewhere."""
    match = _BLOCK_MODULE.search(module_name)
    return f"b{match.group(1)}.{match.group(2)}" if match else None


def key_block(key: str) -> int:
    return int(key.split(".", 1)[0][1:])


def is_ffn_key(key: str) -> bool:
    return key.split(".", 1)[1] in FFN_SUBS


def token_groups(sequence_length: int, text_indices: torch.Tensor, video_indices: torch.Tensor,
                 audio_indices: torch.Tensor, num_condition_video_rows: int,
                 num_condition_audio_rows: int) -> torch.Tensor:
    """Group id (index into ``GROUPS``) of every packed position; -1 marks an uncovered position."""
    groups = torch.full((sequence_length, ), -1, dtype=torch.long, device=video_indices.device)
    groups[text_indices.to(groups.device)] = GROUPS.index("text")
    groups[video_indices[:num_condition_video_rows]] = GROUPS.index("ref_video")
    groups[video_indices[num_condition_video_rows:]] = GROUPS.index("video")
    audio_indices = audio_indices.to(groups.device)
    groups[audio_indices[:num_condition_audio_rows]] = GROUPS.index("ref_audio")
    groups[audio_indices[num_condition_audio_rows:]] = GROUPS.index("audio")
    return groups


def _log2_hist(values: torch.Tensor) -> torch.Tensor:
    logs = values.float().clamp_min(2.0**HIST_MIN_LOG2).log2()
    return torch.histc(logs, bins=HIST_BINS, min=HIST_MIN_LOG2, max=HIST_MAX_LOG2).to(torch.float64)


def hist_edges() -> torch.Tensor:
    return torch.linspace(HIST_MIN_LOG2, HIST_MAX_LOG2, HIST_BINS + 1, dtype=torch.float64).exp2()


def hist_percentile(counts: torch.Tensor, fraction: float) -> float:
    """Upper bin edge below which ``fraction`` of the histogrammed magnitudes fall."""
    cdf = counts.double().cumsum(0)
    index = int(torch.searchsorted(cdf, torch.tensor(fraction * float(cdf[-1]), dtype=torch.float64)))
    return float(hist_edges()[min(index + 1, HIST_BINS)])


class H3AmaxCollector:
    """Hook-based input amax recorder for the H3 DiT block linears.

    Call ``set_token_groups`` before each transformer forward (``wrap_transformer`` does it from
    the forward's packed indices), ``begin_row`` / ``end_row`` around each calibration row.
    """

    def __init__(self, model: torch.nn.Module, subs: Sequence[str] = FFN_SUBS + REPORT_SUBS,
                 tail_min_block: int = TAIL_MIN_BLOCK, topk: int = TOPK) -> None:
        self.modules: dict[str, torch.nn.Module] = {}
        for name, module in model.named_modules():
            key = amax_key(name)
            if key is not None and key.split(".", 1)[1] in subs:
                if key in self.modules:
                    raise ValueError(f"two modules map to amax key {key!r}")
                self.modules[key] = module
        if not self.modules:
            raise ValueError("no transformer_blocks.<n>.<sub> linears matched")
        self.tail_keys = {k for k in self.modules if k.endswith(TAIL_SUB) and key_block(k) >= tail_min_block}
        self.topk_size = topk
        self.topk: dict[str, torch.Tensor] = {}
        self.hist: dict[str, torch.Tensor] = {}
        self.rows: list[dict[str, Any]] = []
        self._row: dict[str, torch.Tensor] | None = None
        self._groups: torch.Tensor | None = None
        self._handles: list[Any] = []

    # ------------------------------------------------------------------ hooks
    def attach(self) -> H3AmaxCollector:
        for key, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(key)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def set_token_groups(self, groups: torch.Tensor | None) -> None:
        self._groups = groups

    def wrap_transformer(self, transformer: torch.nn.Module, layout_lookup) -> Any:
        """Set token groups from each forward's packed indices; ``layout_lookup()`` returns the layout."""
        original = transformer.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            layout = layout_lookup()
            if layout is not None and "video_indices" in kwargs:
                self.set_token_groups(
                    token_groups(int(kwargs["position_ids"].shape[0]), kwargs["text_indices"], kwargs["video_indices"],
                                 kwargs["audio_indices"], int(layout.num_condition_video_rows),
                                 int(layout.num_condition_audio_rows)))
            else:
                self.set_token_groups(None)
            return original(*args, **kwargs)

        transformer.forward = forward
        return lambda: setattr(transformer, "forward", original)

    def _hook(self, key: str):

        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if self._row is not None:
                self.observe(key, args[0])

        return hook

    # ------------------------------------------------------------------ recording
    def begin_row(self) -> None:
        self._row = {}

    def observe(self, key: str, x: torch.Tensor) -> None:
        assert self._row is not None
        flat = x.detach().reshape(-1, x.shape[-1])
        token_amax = flat.abs().amax(dim=-1).float()
        per_group = torch.zeros(len(GROUPS) + 1, device=flat.device)  # last slot: positions without a group
        groups = self._groups
        if groups is not None and groups.numel() * (flat.shape[0] // max(groups.numel(), 1)) == flat.shape[0]:
            index = groups.to(flat.device).repeat(flat.shape[0] // groups.numel())
            index = torch.where(index < 0, len(GROUPS), index)
            per_group.scatter_reduce_(0, index, token_amax, reduce="amax")
        else:
            per_group[len(GROUPS)] = token_amax.max()
        previous = self._row.get(key)
        self._row[key] = per_group if previous is None else torch.maximum(previous, per_group)
        if key in self.tail_keys:
            self._observe_tail(key, flat)

    def _observe_tail(self, key: str, flat: torch.Tensor) -> None:
        magnitudes = flat.abs()
        top = magnitudes.flatten().float().topk(min(self.topk_size, magnitudes.numel())).values
        previous = self.topk.get(key)
        merged = top if previous is None else torch.cat((previous.to(top.device), top))
        self.topk[key] = merged.topk(min(self.topk_size, merged.numel())).values
        counts = sum(_log2_hist(magnitudes[i:i + HIST_CHUNK_ROWS]) for i in range(0, magnitudes.shape[0],
                                                                                    HIST_CHUNK_ROWS))
        self.hist[key] = counts if key not in self.hist else self.hist[key] + counts.to(self.hist[key].device)

    def end_row(self, **meta: Any) -> dict[str, Any]:
        """Close the row; returns ``{**meta, "amax": {key: [per-group amax..., ungrouped]}}``."""
        assert self._row is not None, "end_row without begin_row"
        missing = sorted(set(self.modules) - set(self._row))
        if missing:
            raise RuntimeError(f"no activations reached {missing[:4]} (of {len(missing)}) in this row")
        record = {**meta, "amax": {k: v.cpu().tolist() for k, v in sorted(self._row.items())}}
        self.rows.append(record)
        self._row = None
        return record

    def state(self) -> dict[str, Any]:
        return {"groups": list(GROUPS) + ["ungrouped"], "rows": self.rows,
                "topk": {k: v.cpu() for k, v in self.topk.items()}, "hist": {k: v.cpu() for k, v in self.hist.items()},
                "hist_log2_range": [HIST_MIN_LOG2, HIST_MAX_LOG2, HIST_BINS]}


# ---------------------------------------------------------------------- merge
def merge_states(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine shard states: rows concatenated (sorted by ``plan_index`` when present), tails merged."""
    rows = sorted((row for state in states for row in state["rows"]), key=lambda row: row.get("plan_index", 0))
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate calibration rows across shards")
    topk: dict[str, torch.Tensor] = {}
    hist: dict[str, torch.Tensor] = {}
    for state in states:
        for key, values in state["topk"].items():
            merged = values if key not in topk else torch.cat((topk[key], values))
            topk[key] = merged.topk(min(TOPK, merged.numel())).values
        for key, counts in state["hist"].items():
            hist[key] = counts if key not in hist else hist[key] + counts
    return {"groups": states[0]["groups"], "rows": rows, "topk": topk, "hist": hist,
            "hist_log2_range": states[0]["hist_log2_range"]}


def _row_max(rows: Sequence[dict[str, Any]], key: str) -> float:
    return max(max(row["amax"][key]) for row in rows)


def runtime_amax_table(rows: Sequence[dict[str, Any]], margin: float = 1.0) -> dict[str, float]:
    """``{"b<block>.ff.fc_in"|"b<block>.ff.fc_out": amax}``, the runtime / converter JSON (FFN keys only)."""
    if not rows:
        raise ValueError("no calibration rows")
    keys = sorted({k for row in rows for k in row["amax"] if is_ffn_key(k)}, key=lambda k: (key_block(k), k))
    table = {key: _row_max(rows, key) * margin for key in keys}
    bad = [k for k, v in table.items() if not math.isfinite(v) or v <= 0]
    if bad:
        raise ValueError(f"non-positive or non-finite amax for {bad[:4]}")
    return table


def convergence(rows: Sequence[dict[str, Any]], tolerance: float = 0.05) -> dict[str, Any]:
    """Amax of the first half of the rows (in plan order) against all rows, per FFN layer."""
    half = rows[:len(rows) // 2]
    full, first = runtime_amax_table(rows), runtime_amax_table(half)
    gaps = {key: 1.0 - first[key] / full[key] for key in full}
    worst = sorted(gaps.items(), key=lambda item: -item[1])[:10]
    return {"rows": len(rows), "first_half_rows": len(half), "tolerance": tolerance,
            "passed": all(gap <= tolerance for gap in gaps.values()),
            "layers_over_tolerance": sum(gap > tolerance for gap in gaps.values()),
            "max_gap": worst[0][1], "worst": [{"key": k, "gap": round(g, 4), "first_half": first[k], "full": full[k]}
                                              for k, g in worst]}


def row_statistics(rows: Sequence[dict[str, Any]], key: str, groups: Sequence[str]) -> dict[str, Any]:
    """Distribution of the per-row amax of one layer: are a few rows driving the max, or does it drift up?

    ``prefix_max`` is the running max after 1/8, 1/4, 1/2 and all rows (plan order); ``max_over_p99`` well
    above 1 means the max is set by rare outlier rows (an extreme-value statistic that keeps growing with
    the sample), which argues for a percentile / MSE clip over plain max for that layer.
    """
    values = [max(row["amax"][key]) for row in rows]
    ordered = sorted(values)

    def quantile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    top = sorted(range(len(rows)), key=lambda i: -values[i])[:3]
    return {
        "p50": quantile(0.5), "p90": quantile(0.9), "p99": quantile(0.99), "max": ordered[-1],
        "max_over_p99": ordered[-1] / max(quantile(0.99), 1e-12),
        "rows_within_5pct_of_max": sum(v >= 0.95 * ordered[-1] for v in values),
        "prefix_max": [max(values[:max(1, len(values) * n // 8)]) for n in (1, 2, 4, 8)],
        "top_rows": [{"id": rows[i].get("id"), "case": rows[i].get("case"), "value": values[i],
                      "group": groups[max(range(len(groups)), key=rows[i]["amax"][key].__getitem__)]} for i in top],
    }


def layer_report(merged: dict[str, Any]) -> dict[str, Any]:
    """Per layer: overall amax, per-group amax, per-case amax, and for tail layers top-k / percentiles."""
    rows = merged["rows"]
    groups = merged["groups"]
    keys = sorted({k for row in rows for k in row["amax"]}, key=lambda k: (key_block(k), k))
    cases = sorted({row.get("case", "?") for row in rows})
    report: dict[str, Any] = {}
    for key in keys:
        per_group = [max(row["amax"][key][i] for row in rows) for i in range(len(groups))]
        entry: dict[str, Any] = {
            "all": max(per_group),
            "groups": {g: v for g, v in zip(groups, per_group, strict=True) if v > 0},
            "argmax_group": groups[max(range(len(groups)), key=per_group.__getitem__)],
            "cases": {case: max(max(r["amax"][key]) for r in rows if r.get("case", "?") == case) for case in cases},
        }
        if is_ffn_key(key):
            entry["rows"] = row_statistics(rows, key, groups)
        if key in merged["hist"]:
            counts = merged["hist"][key]
            top = merged["topk"][key]
            entry["tail"] = {
                "top": [float(v) for v in top[:16]],
                "top_k_min": float(top[-1]),
                "p99_99": hist_percentile(counts, 0.9999),
                "p99_999": hist_percentile(counts, 0.99999),
                "p99_9999": hist_percentile(counts, 0.999999),
                "count": float(counts.sum()),
            }
        report[key] = entry
    return report
