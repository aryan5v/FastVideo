# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the OmniRef NVFP4 amax collector and its runtime/converter key format."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts/quantization/minimax_h3_omniref"
CONVERTER = ROOT / "scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


col = _load("h3_amax_collector", SCRIPTS / "h3_amax_collector.py")


class _Block(nn.Module):

    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn, self.ff = nn.Module(), nn.Module()
        self.attn.to_q = nn.Linear(width, width, bias=False)
        self.attn.to_out = nn.Linear(width, width, bias=False)
        self.ff.fc_in = nn.Linear(width, 2 * width, bias=False)
        self.ff.fc_out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.to_out(self.attn.to_q(x))
        hidden, gate = self.ff.fc_in(x).chunk(2, dim=-1)
        return x + self.ff.fc_out(hidden * torch.nn.functional.silu(gate))


class _Model(nn.Module):

    def __init__(self, blocks: int = 3, width: int = 16) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block(width) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _input_amax(model: _Model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference per-token input |x| max of every recorded linear, by direct capture."""
    seen: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        key = col.amax_key(name)
        if key is not None and isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(
                lambda m, a, key=key: seen.__setitem__(key, a[0].detach().abs().amax(-1).reshape(-1))))
    model(x)
    for handle in handles:
        handle.remove()
    return seen


def test_amax_key_matches_runtime_and_converter_format() -> None:
    assert col.amax_key("transformer_blocks.3.ff.fc_in") == "b3.ff.fc_in"
    assert col.amax_key("minimax_h3.transformer_blocks.49.ff.fc_out") == "b49.ff.fc_out"
    assert col.amax_key("token_refiner.blocks.0.ff.fc_in") is None
    # The runtime's lookup (nvfp4_config._static_activation_global_sf) derives the same key.
    prefix = "transformer_blocks.12.ff.fc_out"
    match = re.search(r"transformer_blocks\.(\d+)\.(.+)$", prefix)
    assert f"b{match.group(1)}.{match.group(2)}" == col.amax_key(prefix)


def test_token_groups_cover_packed_layout() -> None:
    text, video, audio = torch.tensor([0, 1]), torch.tensor([2, 3, 4, 5]), torch.tensor([6, 7, 8])
    groups = col.token_groups(10, text, video, audio, num_condition_video_rows=1, num_condition_audio_rows=2)
    names = [col.GROUPS[g] if g >= 0 else None for g in groups.tolist()]
    assert names == ["text", "text", "ref_video", "video", "video", "video", "ref_audio", "ref_audio", "audio", None]


def test_collector_records_running_max_per_group_and_row() -> None:
    torch.manual_seed(0)
    model = _Model()
    collector = col.H3AmaxCollector(model, tail_min_block=2, topk=8).attach()
    groups = col.token_groups(6, torch.tensor([0]), torch.tensor([1, 2, 3]), torch.tensor([4, 5]), 1, 1)
    collector.set_token_groups(groups)
    rows = []
    for row in range(2):
        collector.begin_row()
        references = []
        for _ in range(3):  # several forwards per row (the PDD rungs)
            x = torch.randn(1, 6, 16) * (row + 1)
            recording, collector._row = collector._row, None  # the reference capture must not be recorded
            references.append(_input_amax(model, x))
            collector._row = recording
            collector.set_token_groups(groups)
            with torch.no_grad():
                model(x)
        rows.append(collector.end_row(id=f"r{row}", case="first_frame", plan_index=row))
        expected = {key: torch.stack([r[key] for r in references]).amax(0) for key in references[0]}
        for key, per_token in expected.items():
            recorded = torch.tensor(rows[-1]["amax"][key])
            for group_index in range(len(col.GROUPS)):
                mask = groups == group_index
                assert recorded[group_index].item() == pytest.approx(per_token[mask].max().item(), rel=1e-6)
    collector.detach()
    assert set(rows[0]["amax"]) == {f"b{b}.{s}" for b in range(3) for s in col.FFN_SUBS + col.REPORT_SUBS}
    assert set(collector.topk) == set(collector.hist) == {"b2.ff.fc_out"}
    assert collector.topk["b2.ff.fc_out"].numel() == 8
    assert collector.topk["b2.ff.fc_out"][0].item() == pytest.approx(max(max(r["amax"]["b2.ff.fc_out"]) for r in rows),
                                                                      rel=1e-6)
    # Histogram counts every fc_out input element of every forward.
    assert collector.hist["b2.ff.fc_out"].sum().item() == 2 * 3 * 6 * 16

    table = col.runtime_amax_table(collector.rows)
    assert set(table) == {f"b{b}.{s}" for b in range(3) for s in col.FFN_SUBS}  # FFN only: attention stays unit
    for key, value in table.items():
        assert value == pytest.approx(max(max(r["amax"][key]) for r in rows))


def test_end_row_rejects_layers_without_activations() -> None:
    model = _Model()
    collector = col.H3AmaxCollector(model)
    collector.begin_row()
    with pytest.raises(RuntimeError, match="no activations"):
        collector.end_row(id="x")


def test_ungrouped_when_token_count_differs() -> None:
    model = _Model(blocks=1)
    collector = col.H3AmaxCollector(model).attach()
    collector.set_token_groups(torch.zeros(5, dtype=torch.long))
    collector.begin_row()
    model(torch.randn(1, 7, 16))
    record = collector.end_row(id="x")
    values = record["amax"]["b0.ff.fc_in"]
    assert values[:len(col.GROUPS)] == [0.0] * len(col.GROUPS) and values[-1] > 0


def _rows(values: list[float]) -> list[dict]:
    return [{"id": str(i), "plan_index": i, "case": "c", "amax": {"b0.ff.fc_in": [v], "b0.ff.fc_out": [1.0]}}
            for i, v in enumerate(values)]


def test_convergence_first_half_vs_full() -> None:
    assert col.convergence(_rows([10.0, 9.0, 9.8, 10.2]))["passed"]
    failed = col.convergence(_rows([5.0, 5.0, 9.0, 10.0]))
    assert not failed["passed"] and failed["worst"][0]["key"] == "b0.ff.fc_in"
    assert failed["max_gap"] == pytest.approx(0.5)


def test_merge_orders_rows_and_rejects_duplicates() -> None:
    state = lambda rows: {"groups": ["all"], "rows": rows, "topk": {}, "hist": {}, "hist_log2_range": [0, 1, 1]}
    merged = col.merge_states([state(_rows([1.0, 2.0])[1:]), state(_rows([1.0, 2.0])[:1])])
    assert [r["plan_index"] for r in merged["rows"]] == [0, 1]
    with pytest.raises(ValueError, match="duplicate"):
        col.merge_states([state(_rows([1.0])), state(_rows([1.0]))])


def test_hist_percentile_brackets_values() -> None:
    values = torch.tensor([1.0] * 999 + [1000.0])
    counts = col._log2_hist(values)
    assert col.hist_percentile(counts, 0.5) == pytest.approx(1.0, rel=0.1)
    assert col.hist_percentile(counts, 1.0) >= 1000.0


def test_runtime_table_round_trips_through_converter_lookup(tmp_path) -> None:
    converter = _load("convert_minimax_h3_modelopt_nvfp4_dit", CONVERTER)
    table = col.runtime_amax_table(_rows([3.0, 4.0]))
    path = tmp_path / "amax.json"
    path.write_text(json.dumps(table))
    loaded = json.loads(path.read_text())
    assert converter.act_amax_for("transformer_blocks.0.ff.fc_in", loaded, required=True) == 4.0
    assert converter.act_amax_for("transformer_blocks.0.attn.to_q", loaded, required=False) is None
    with pytest.raises(SystemExit):
        converter.act_amax_for("transformer_blocks.1.ff.fc_in", loaded, required=True)


def test_calibration_plan_counts_balance_and_heldout_disjoint() -> None:
    calib = _load("calibrate_omniref_nvfp4", SCRIPTS / "calibrate_omniref_nvfp4.py")
    groups = {(case, res): [{"id": f"{case}:{res}-src{i}", "case": case, "resolution": res, "parquet": f"/p/{i}",
                             "frames": 124} for i in range(40)]
              for case in ("first_frame", "storyboard") for res in ("480p", "768p")}
    # Source ids are shared across resolutions, as in the OmniRef manifest.
    for (case, res), entries in groups.items():
        for entry in entries:
            entry["id"] = f"{case}:src{entry['id'].rsplit('src', 1)[1]}"
    plan = calib._group_plan(groups, 7, {"480p": 8, "768p": 2})
    assert len(plan) == 2 * (8 + 2)
    assert [p["plan_index"] for p in plan] == list(range(len(plan)))
    half = plan[:len(plan) // 2]
    assert {(p["case"], p["resolution"]) for p in half} == set(groups)  # every group in the first half
    held = calib._group_plan(groups, 8, {"480p": 4, "768p": 1}, frozenset(p["source"] for p in plan))
    assert not {p["source"] for p in held} & {p["source"] for p in plan}


def test_row_statistics_flags_single_outlier_row() -> None:
    rows = _rows([10.0] * 99 + [30.0])
    stats = col.row_statistics(rows, "b0.ff.fc_in", ["all"])
    assert stats["max"] == 30.0 and stats["p99"] == 10.0 and stats["max_over_p99"] == pytest.approx(3.0)
    assert stats["rows_within_5pct_of_max"] == 1 and stats["top_rows"][0]["id"] == "99"
    assert stats["prefix_max"] == [10.0, 10.0, 10.0, 30.0]


def test_t4_gate_fails_closed_without_rows(tmp_path) -> None:
    import argparse

    t4 = _load("omniref_t4_metrics", SCRIPTS / "omniref_t4_metrics.py")
    t4.report(argparse.Namespace(output_dir=str(tmp_path), gate_resolutions=["480p"]))
    result = json.loads((tmp_path / "gate_a_report.json").read_text())
    assert result["rows"] == 0 and result["gate_t4_passed"] is False


def test_variant_overrides_scales_and_restores() -> None:
    swap_mod = _load("h3_nvfp4_swap", SCRIPTS / "h3_nvfp4_swap.py")

    class _Method:
        _dynamic_act_cached = False

    layers = []
    for name in ("transformer_blocks.0.ff.fc_in", "transformer_blocks.0.ff.fc_out", "transformer_blocks.0.attn.to_q"):
        module = nn.Linear(4, 4)
        module.register_buffer("_nvfp4_weight", torch.zeros(4, 2, dtype=torch.uint8))
        module.register_buffer("_nvfp4_input_global_sf", torch.tensor(1.0))
        bf16, nvfp4 = _Method(), _Method()
        module.quant_method = bf16
        layers.append((name, module, bf16, nvfp4))
    swap = object.__new__(swap_mod.NVFP4Swap)
    swap.layers = layers
    variant = swap_mod.Variant(amax={"transformer_blocks.0.ff.fc_in": 2688.0},
                               dynamic=frozenset({"transformer_blocks.0.ff.fc_out"}),
                               bf16=frozenset({"transformer_blocks.0.attn.to_q"}))
    assert swap_mod.Variant.from_json(json.loads(json.dumps(variant.to_json()))) == variant
    (_, fc_in, _, m_in), (_, fc_out, _, m_out), (_, to_q, q_bf16, _) = layers
    with swap.enabled(variant=variant):
        assert fc_in.quant_method is m_in and fc_out.quant_method is m_out and to_q.quant_method is q_bf16
        assert fc_in._nvfp4_input_global_sf.item() == pytest.approx(1.0)  # 2688 / 2688
        assert "_nvfp4_input_global_sf" not in fc_out._buffers and m_out._dynamic_act_cached
    assert fc_out._nvfp4_input_global_sf.item() == 1.0 and not m_out._dynamic_act_cached
    assert all(module.quant_method is bf16 for _, module, bf16, _ in layers)
    with pytest.raises(ValueError, match="not exported"):
        with swap.enabled(variant=swap_mod.Variant(bf16=frozenset({"nope"}))):
            pass
