# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision parameter loading coverage for native model dtype policies."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.distributed.parallel_state import init_distributed_environment
from fastvideo.models.dits.lingbot_video import LingBotVideoRouter
from fastvideo.models.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    maybe_load_fsdp_model,
    shard_model,
)


class _MixedDtypeModel(torch.nn.Module):
    """Tiny model that keeps one checkpoint parameter in fp32."""

    def __init__(self) -> None:
        super().__init__()
        self.bulk = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.sensitive = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the sensitive test tensor in fp32."""
        return torch.float32 if name == "sensitive" else default_dtype


class _MixedDtypeBlock(torch.nn.Module):
    """Small FSDP child with one managed and one replicated parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.bulk = torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
        self.sensitive = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply both parameters so FSDP must initialize the mixed child."""
        return torch.nn.functional.linear(hidden_states, self.bulk) + self.sensitive.to(hidden_states.dtype)


class _NestedMixedDtypeModel(torch.nn.Module):
    """Nested model exercising both child and root ignored-parameter wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_MixedDtypeBlock()])
        self.root_bulk = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.root_sensitive = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep every sensitive test parameter replicated in fp32."""
        return torch.float32 if "sensitive" in name else default_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the sharded child and consume both root parameters."""
        hidden_states = self.blocks[0](hidden_states)
        return hidden_states * self.root_bulk + self.root_sensitive.to(hidden_states.dtype)


class _GroupedMixedDtypeModel(torch.nn.Module):
    """Tiny model declaring one FP32 compute module."""

    _keep_in_fp32_modules = frozenset({"sensitive"})

    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        self.sensitive = torch.nn.Linear(2, 2, dtype=torch.float32)

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if name.startswith("sensitive.") else default_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = self.sensitive(hidden_states.float()).to(input_dtype)
        return self.block(hidden_states)


class _RouterBufferModel(torch.nn.Module):
    """Wrap the released router to exercise its persistent correction buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.router = LingBotVideoRouter(2, 3, 1, "sigmoid", True, None, None, 1.0)

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Keep the released router state in fp32."""
        return torch.float32 if "router" in name else default_dtype


def test_full_state_dict_loader_honors_model_parameter_dtypes() -> None:
    """Load exact fp32 values for selected tensors while casting ordinary weights."""
    model = _MixedDtypeModel()
    bulk = torch.tensor([1.001, -2.003], dtype=torch.float32)
    sensitive = torch.tensor([3.001, -4.003], dtype=torch.float32)
    incompatible = load_model_from_full_model_state_dict(
        model,
        iter((("bulk", bulk), ("sensitive", sensitive))),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert model.bulk.dtype == torch.bfloat16
    assert model.sensitive.dtype == torch.float32
    torch.testing.assert_close(model.bulk, bulk.to(torch.bfloat16))
    torch.testing.assert_close(model.sensitive, sensitive)


def test_full_state_dict_loader_preserves_router_bias_buffer() -> None:
    """Keep the MoE correction bias registered as a non-trainable buffer."""
    model = _RouterBufferModel()
    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    incompatible = load_model_from_full_model_state_dict(
        model,
        iter((("router.weight", weight), ("router.e_score_correction_bias", bias))),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert "router.e_score_correction_bias" in dict(model.named_buffers())
    assert "router.e_score_correction_bias" not in dict(model.named_parameters())
    torch.testing.assert_close(model.router.e_score_correction_bias, bias)


def test_training_rejection_uses_fsdp_parameter_dtype() -> None:
    """Reject replicated fp32 training state when construction defaults to fp32."""
    with pytest.raises(NotImplementedError, match="separate gradient synchronization"):
        maybe_load_fsdp_model(
            model_cls=_MixedDtypeModel,
            init_params={},
            weight_dir_list=[],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            default_dtype=torch.float32,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            training_mode=True,
            pin_cpu_memory=False,
        )


def test_declared_fp32_module_gets_separate_fsdp_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _GroupedMixedDtypeModel()
    calls: list[tuple[torch.nn.Module, dict]] = []

    monkeypatch.setattr(
        "fastvideo.models.loader.fsdp_load.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    shard_model(
        model,
        cpu_offload=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        ),
        mesh=None,
        fsdp_shard_conditions=[lambda name, module: name == "block"],
        pin_cpu_memory=False,
    )

    assert [module for module, _ in calls] == [model.block, model.sensitive, model]
    assert [kwargs["mp_policy"].param_dtype for _, kwargs in calls] == [
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]
    assert all("ignored_params" not in kwargs for _, kwargs in calls)

    with pytest.raises(ValueError, match="contains a declared FP32 compute group"):
        shard_model(
            _GroupedMixedDtypeModel(),
            cpu_offload=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            mesh=None,
            fsdp_shard_conditions=[lambda name, module: name == ""],
            pin_cpu_memory=False,
        )


def test_nested_fsdp_ignores_selected_fp32_parameters() -> None:
    """Run a CUDA forward with bf16 DTensors and replicated fp32 parameters."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("mixed-dtype FSDP coverage requires CUDA")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    model = _NestedMixedDtypeModel().cuda()
    mesh = torch.distributed.init_device_mesh(
        "cuda",
        mesh_shape=(1, 1),
        mesh_dim_names=("replicate", "shard"),
    )
    shard_model(
        model,
        cpu_offload=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        ),
        mesh=mesh,
        fsdp_shard_conditions=[lambda name, module: isinstance(module, _MixedDtypeBlock)],
        pin_cpu_memory=False,
    )
    output = model(torch.ones(1, 2, device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(output).all()
    assert isinstance(model.blocks[0].bulk, DTensor)
    assert not isinstance(model.blocks[0].sensitive, DTensor)
    assert model.blocks[0].sensitive.dtype == torch.float32
    assert isinstance(model.root_bulk, DTensor)
    assert not isinstance(model.root_sensitive, DTensor)
    assert model.root_sensitive.dtype == torch.float32


def _run_grouped_fsdp_worker() -> None:
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    try:
        torch.manual_seed(7)
        model = _GroupedMixedDtypeModel().to(device)
        observed: dict[str, tuple[torch.dtype, torch.dtype]] = {}

        def record(name: str):
            def hook(module, inputs, output):
                observed[name] = (inputs[0].dtype, module.weight.dtype)
                assert output.dtype == module.weight.dtype

            return hook

        model.block.register_forward_hook(record("block"))
        model.sensitive.register_forward_hook(record("sensitive"))
        mesh = torch.distributed.init_device_mesh(
            "cuda",
            mesh_shape=(1, torch.distributed.get_world_size()),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
                cast_forward_inputs=False,
            ),
            mesh=mesh,
            fsdp_shard_conditions=[lambda name, module: name == "block"],
            pin_cpu_memory=False,
        )

        output = model(torch.randn(4, 2, device=device, dtype=torch.bfloat16))
        assert output.dtype == torch.bfloat16
        output.float().square().mean().backward()
        assert observed == {
            "block": (torch.bfloat16, torch.bfloat16),
            "sensitive": (torch.float32, torch.float32),
        }
        for parameter in model.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad.to_local()).all()
            full_grad = parameter.grad.full_tensor()
            rank_zero_grad = full_grad.clone()
            torch.distributed.broadcast(rank_zero_grad, src=0)
            torch.testing.assert_close(full_grad, rank_zero_grad)
    finally:
        torch.distributed.destroy_process_group()


def test_declared_fp32_group_distributed_forward_backward() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--grouped-fsdp-worker",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-fsdp-worker", action="store_true")
    if parser.parse_args().grouped_fsdp_worker:
        _run_grouped_fsdp_worker()
