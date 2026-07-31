# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for generic FX capture over repeated module stacks."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from fastvideo.forward_context import get_forward_context, set_forward_context
from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.optimization import fx_capture
from fastvideo.optimization import profiler as optimization_profiler


class _Block(nn.Module):
    """Stand-in for a transformer block: no architecture-specific behavior."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(hidden_states * self.scale) + hidden_states


class _UntraceableBlock(nn.Module):
    """Data-dependent control flow — symbolic tracing must fail on this."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if bool(hidden_states.sum() > 0):
            return hidden_states * 2
        return hidden_states


class _ShapeBranchBlock(nn.Module):
    """Representative shape branch that FX cannot evaluate from a Proxy."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] > 2:
            return hidden_states * 2 * self.scale
        return (hidden_states + 1) * self.scale


class _ScalarInputBlock(nn.Module):
    """Export specializes safe scalar arguments instead of treating them as tensors."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_seq_len: int,
    ) -> torch.Tensor:
        if original_seq_len > 2:
            return hidden_states.flatten(0, 0) + original_seq_len
        return hidden_states


class _FakeLayerwiseOffloadHook(ForwardHook):
    class _State:
        def __init__(self, module: nn.Module) -> None:
            self.module = module
            self.gpu_named_parameters = {}

        def wait_and_replace_params(self) -> None:
            for name, parameter in self.module.named_parameters():
                if name in self.gpu_named_parameters:
                    continue
                self.gpu_named_parameters[name] = parameter.data

    def __init__(self, module: nn.Module) -> None:
        self.pre_calls = 0
        self.post_calls = 0
        self.state = self._State(module)

    @classmethod
    def name(cls) -> str:
        return "LayerwiseOffloadHook"

    def pre_forward(self, module, *args, **kwargs):
        self.pre_calls += 1
        self.state.wait_and_replace_params()
        return args, kwargs

    def post_forward(self, module, output):
        self.post_calls += 1
        self.state.gpu_named_parameters.clear()
        return output


class _ContextAttention(nn.Module):
    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_forward_context()
        return hidden_states + context.current_timestep


class _ContextShapeBranchBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.attention = _ContextAttention()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] > 2:
            hidden_states = hidden_states * self.scale
        return self.attention(hidden_states)


class _Transformer(nn.Module):

    def __init__(self, depth: int = 3, width: int = 4, block=_Block) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.blocks = nn.ModuleList([block(width) if block is _Block else block() for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _FakePipeline:
    """Minimal stand-in for a composed pipeline's module mapping + forward."""

    def __init__(self, transformer: nn.Module) -> None:
        self.modules = {"transformer": transformer, "scheduler": object()}
        self._optimization_profile_calls = 0

    def forward(self, hidden_states: torch.Tensor, steps: int = 2) -> torch.Tensor:
        call_index = self._optimization_profile_calls
        self._optimization_profile_calls += 1
        with optimization_profiler.optimization_profile(call_index, self.modules):
            for _ in range(steps):
                hidden_states = self.modules["transformer"](hidden_states)
        return hidden_states


class _Event:
    key = "aten::mul"
    count = 6
    device_time_total = 12.5
    self_device_time_total = 9.5
    cpu_time_total = 20.0
    input_shapes = [[2, 4]]


class _FakeProfiler:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def key_averages(self, *, group_by_input_shape):
        return [_Event()]


def _enable_profile(monkeypatch, output, *, capture: str = "1") -> None:
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT", str(output))
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_SKIP_RUNS", "0")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_WORKLOAD_ID", "unit")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_MODEL_ID", "model")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_CAPTURE_FX", capture)
    monkeypatch.setattr(
        optimization_profiler.torch.profiler,
        "profile",
        lambda **kwargs: _FakeProfiler(),
    )
    monkeypatch.setattr(optimization_profiler.torch.cuda, "is_available", lambda: False)


def test_capture_targets_are_repeated_stacks_only():
    model = _Transformer(depth=3)
    targets = fx_capture.default_capture_targets(model)

    assert {scope for scope, _ in targets} == {"blocks"}
    # Every stack member is hooked so call frequency covers the whole stack.
    assert len(targets) == 3


def test_fake_pipeline_export_carries_regions_and_frequencies(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)

    pipeline = _FakePipeline(_Transformer(depth=3))
    pipeline.forward(torch.randn(2, 4), steps=2)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["rows"], "existing profiler rows must be preserved"
    assert payload["capture"]["capture_schema_version"] == fx_capture.CAPTURE_SCHEMA_VERSION
    assert payload["capture"]["scopes"] == ["transformer.blocks"]
    # 3 blocks x 2 steps.
    assert payload["capture"]["scope_calls"]["transformer.blocks"] == 6
    assert payload["capture"]["errors"] == []

    regions = payload["regions"]
    assert len(regions) == 1
    region = regions[0]
    assert region["parent_module"] == "transformer.blocks"
    assert region["calls"] == 6
    assert sum(region["shape_frequency"].values()) == 6
    assert any(op.startswith("aten::") for op in region["operations"])
    assert region["dependencies"]
    assert region["inputs"][0]["shape"] == [2, 4]
    assert region["inputs"][0]["dtype"] == "float32"
    assert region["outputs"][0]["shape"] == [2, 4]
    assert region["attributes"]["module_class"] == "_Block"


def test_capture_is_off_unless_optimization_profile_requested(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output, capture="0")

    model = _Transformer(depth=2)
    pipeline = _FakePipeline(model)
    pipeline.forward(torch.randn(2, 4), steps=1)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "regions" not in payload
    assert "capture" not in payload
    # No hooks may survive a capture-disabled run.
    assert all(not block._forward_hooks for block in model.blocks)


def test_capture_disabled_without_profile_output(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_OUTPUT", "")
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_CAPTURE_FX", "1")

    model = _Transformer(depth=2)
    pipeline = _FakePipeline(model)
    hidden = torch.randn(2, 4)
    expected = model(model(hidden.clone()))
    result = pipeline.forward(hidden, steps=2)

    torch.testing.assert_close(result, expected)
    assert all(not block._forward_hooks for block in model.blocks)


def test_hooks_are_removed_and_outputs_unchanged(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)

    model = _Transformer(depth=2)
    hidden = torch.randn(2, 4)
    baseline = model(hidden.clone())

    pipeline = _FakePipeline(model)
    captured = pipeline.forward(hidden.clone(), steps=1)

    torch.testing.assert_close(captured, baseline)
    assert all(not block._forward_hooks for block in model.blocks)
    # A later, unprofiled call must run with no hooks attached at all.
    after = pipeline.forward(hidden.clone(), steps=1)
    torch.testing.assert_close(after, baseline)


def test_capture_emits_shape_specific_profiler_ranges(monkeypatch):
    entered: list[str] = []
    exited: list[str] = []

    class _Range:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            entered.append(self.name)

        def __exit__(self, exc_type, exc, traceback):
            exited.append(self.name)

    monkeypatch.setattr(
        fx_capture.torch.profiler,
        "record_function",
        lambda name: _Range(name),
    )
    session = fx_capture.FXCaptureSession()
    model = _Transformer(depth=2)
    assert session.attach(model, prefix="transformer") == 2

    model(torch.randn(2, 4))
    payload = session.finalize()
    region_name = payload["regions"][0]["name"]

    assert entered == [f"motionkernel::{region_name}"] * 2
    assert exited == entered


def test_trace_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)

    pipeline = _FakePipeline(_Transformer(depth=2, block=_UntraceableBlock))
    pipeline.forward(torch.randn(2, 4), steps=1)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["regions"] == []
    reasons = [item["reason"] for item in payload["graph_breaks"]]
    assert all(reason.startswith("capture_failed:") for reason in reasons)
    assert {reason.split(":", 2)[1] for reason in reasons} == {
        "symbolic",
        "export",
        "dynamo",
    }
    assert payload["rows"], "profiler rows still export after a capture failure"


def test_finalize_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)

    def _boom(self):
        raise RuntimeError("synthetic finalize failure")

    monkeypatch.setattr(fx_capture.FXCaptureSession, "finalize", _boom)

    model = _Transformer(depth=2)
    pipeline = _FakePipeline(model)
    pipeline.forward(torch.randn(2, 4), steps=1)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["regions"] == []
    assert payload["capture"]["errors"] == ["finalize_failed:RuntimeError"]
    assert all(not block._forward_hooks for block in model.blocks)


def test_auto_falls_back_to_export_for_shape_control_flow(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)
    pipeline = _FakePipeline(
        _Transformer(depth=2, block=_ShapeBranchBlock)
    )

    pipeline.forward(torch.randn(2, 4), steps=1)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["regions"]) == 1
    attributes = payload["regions"][0]["attributes"]
    assert attributes["capture_mode"] == "export"
    assert attributes["capture_attempts"] == ["symbolic", "export"]
    assert attributes["capture_failures"] == [
        "capture_failed:symbolic:dynamic_python_control_flow:TraceError"
    ]
    executable_ir = attributes["executable_ir"]
    assert executable_ir["schema_version"] == 1
    assert len(executable_ir["nodes"]) == len(payload["regions"][0]["operations"])
    assert {item["kind"] for item in executable_ir["inputs"]} == {
        "runtime",
        "lifted",
    }
    assert [item["name"] for item in executable_ir["inputs"]] == [
        "lifted_0",
        "input_0",
    ]
    assert all(
        set(item["meta"])
        == {"shape", "stride", "dtype", "device_type", "requires_grad"}
        for item in executable_ir["inputs"]
    )
    assert "scale" not in json.dumps(executable_ir)
    assert payload["capture"]["capture_mode_breakdown"]["export"] == 1


def test_fallback_bypasses_eager_only_offload_wrapper():
    model = _Transformer(depth=2, block=_ShapeBranchBlock)
    hooks = []
    for block in model.blocks:
        manager = ModuleHookManager.get_from_or_default(block)
        hook = _FakeLayerwiseOffloadHook(block)
        manager.append_forward_hook(hook)
        hooks.append(hook)

    session = fx_capture.FXCaptureSession()
    assert session.attach(model, prefix="transformer") == 2
    model(torch.randn(2, 4))
    prefetched = model.blocks[0].scale.data
    hooks[0].state.gpu_named_parameters["scale"] = prefetched
    payload = session.finalize()

    assert payload["regions"]
    assert payload["regions"][0]["attributes"]["capture_mode"] == "export"
    assert all(hook.pre_calls == 1 for hook in hooks)
    assert all(hook.pre_calls == hook.post_calls for hook in hooks)
    assert hooks[0].state.gpu_named_parameters == {"scale": prefetched}
    assert not hooks[1].state.gpu_named_parameters
    model(torch.randn(2, 4))
    assert all(hook.pre_calls == 2 for hook in hooks)
    assert all(hook.pre_calls == hook.post_calls for hook in hooks)
    assert all(not hook.state.gpu_named_parameters for hook in hooks)
    assert all(
        callable(block.forward)
        and ModuleHookManager.get_from(block) is not None
        for block in model.blocks
    )


def test_export_ir_specializes_safe_non_tensor_inputs():
    class _ScalarTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_ScalarInputBlock(), _ScalarInputBlock()]
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            original_seq_len: int,
        ) -> torch.Tensor:
            for block in self.blocks:
                hidden_states = block(hidden_states, original_seq_len)
            return hidden_states

    model = _ScalarTransformer()
    session = fx_capture.FXCaptureSession()
    assert session.attach(model, prefix="transformer") == 2
    model(torch.randn(2, 4), 4)

    region = session.finalize()["regions"][0]
    executable_ir = region["attributes"]["executable_ir"]

    assert region["attributes"]["capture_mode"] == "export"
    assert len(executable_ir["inputs"]) == 1
    assert executable_ir["inputs"][0]["name"] == "input_0"
    assert any(
        argument == {"const": 4}
        for node in executable_ir["nodes"]
        for argument in node["args"]
    )


def test_fallback_restores_observed_context_and_disabled_forwards():
    model = _Transformer(depth=2, block=_ContextShapeBranchBlock)
    session = fx_capture.FXCaptureSession()
    assert session.attach(model, prefix="transformer") == 2

    with set_forward_context(current_timestep=3, attn_metadata=None):
        model(torch.randn(2, 4))
    payload = session.finalize()

    assert payload["regions"]
    assert payload["regions"][0]["attributes"]["capture_mode"] == "export"
    assert all(
        getattr(block.attention.forward, "_torchdynamo_disable", False)
        for block in model.blocks
    )
    assert all(
        variant.observed_context is None
        for record in session._scopes.values()
        for variant in record.variants.values()
    )


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_each_capture_mode_has_stable_metadata(mode):
    session = fx_capture.FXCaptureSession(tracer=mode)
    model = _Transformer(depth=2)
    assert session.attach(model, prefix="transformer") == 2

    model(torch.randn(2, 4))
    payload = session.finalize()

    assert payload["regions"]
    region = payload["regions"][0]
    assert region["attributes"]["capture_mode"] == mode
    assert region["attributes"]["capture_attempts"] == [mode]
    assert payload["capture"]["capture_mode_breakdown"][mode] == 1
    assert all(
        variant.example_args is None and variant.example_kwargs is None
        for record in session._scopes.values()
        for variant in record.variants.values()
    )


def test_nested_dataclass_tensor_inputs_contribute_shape_metadata():
    @dataclass
    class _State:
        hidden_states: torch.Tensor
        conditioning: tuple[torch.Tensor, torch.Tensor]
        config_name: str

    state = _State(
        hidden_states=torch.randn(2, 4),
        conditioning=(torch.randn(1, 4), torch.randn(1, 4)),
        config_name="private text is not metadata",
    )

    metas = fx_capture._input_metas((state,), {})

    assert [meta["name"] for meta in metas] == [
        "input_0_hidden_states",
        "input_0_conditioning_0",
        "input_0_conditioning_1",
    ]
    assert "private text" not in str(metas)


def test_shape_variants_are_tracked_and_bounded(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_FX_MAX_SHAPES", "1")

    pipeline = _FakePipeline(_Transformer(depth=2))
    call_index = pipeline._optimization_profile_calls
    pipeline._optimization_profile_calls += 1
    transformer = pipeline.modules["transformer"]
    with optimization_profiler.optimization_profile(call_index, pipeline.modules):
        transformer(torch.randn(2, 4))
        transformer(torch.randn(8, 4))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["regions"]) == 1
    assert payload["capture"]["dropped_shape_variants"] == 2


def test_export_contains_no_tensor_or_prompt_data(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)
    monkeypatch.setenv("FASTVIDEO_OPTIMIZATION_PROFILE_WORKLOAD_ID", "unit")

    pipeline = _FakePipeline(_Transformer(depth=2))
    pipeline.forward(torch.randn(2, 4), steps=1)

    raw = output.read_text(encoding="utf-8")
    payload = json.loads(raw)
    fx_capture.assert_metadata_only({
        "regions": payload["regions"],
        "graph_breaks": payload["graph_breaks"],
        "unsupported": payload["unsupported"],
        "capture": payload["capture"],
    })
    for forbidden in ("prompt", "weights", "tensor_values", "activations"):
        assert f'"{forbidden}"' not in raw


class _KwargBlock(nn.Module):
    """Blocks are usually called with keyword tensors, as DiT stages do."""

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + encoder_hidden_states


class _KwargTransformer(nn.Module):

    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_KwargBlock() for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=context)
        return hidden_states


def test_keyword_tensor_inputs_are_captured(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    _enable_profile(monkeypatch, output)

    model = _KwargTransformer(depth=2)
    pipeline = _FakePipeline(model)
    call_index = pipeline._optimization_profile_calls
    with optimization_profiler.optimization_profile(call_index, pipeline.modules | {"transformer": model}):
        model(torch.randn(2, 4), torch.randn(2, 4))

    payload = json.loads(output.read_text(encoding="utf-8"))
    names = [meta["name"] for meta in payload["regions"][0]["inputs"]]
    assert names == ["input_0", "kwarg_encoder_hidden_states"]


def test_fingerprint_is_stable_and_value_independent():
    inputs = [{"shape": [2, 4], "stride": [4, 1], "dtype": "float32", "device_type": "cpu", "requires_grad": False}]
    first = fx_capture.graph_fingerprint(operations=["aten::mul", "aten::silu"], input_signatures=inputs)
    second = fx_capture.graph_fingerprint(operations=["aten::mul", "aten::silu"], input_signatures=inputs)
    different = fx_capture.graph_fingerprint(operations=["aten::mul"], input_signatures=inputs)

    assert first == second
    assert first != different
    assert len(first) == 32


def test_assert_metadata_only_rejects_tensors_and_forbidden_keys():
    import pytest

    with pytest.raises(RuntimeError):
        fx_capture.assert_metadata_only({"regions": [{"prompt": "a cat"}]})
    with pytest.raises(RuntimeError):
        fx_capture.assert_metadata_only({"regions": [{"inputs": torch.zeros(2)}]})
