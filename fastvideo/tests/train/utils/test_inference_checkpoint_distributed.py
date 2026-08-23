# SPDX-License-Identifier: Apache-2.0
"""Two-GPU FSDP2 gate for validation-triggered inference checkpoints.

This test intentionally launches a real two-rank NCCL worker and must run on a
compute node. It covers frozen parameters, DCP staging, bounded rank-zero
export, strict reload, RNG preservation, and rank-zero export failure
propagation without leaving a long-running NCCL collective outstanding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard

import fastvideo.train.utils.inference_checkpoint as inference_checkpoint
from fastvideo.train.utils.checkpoint import CheckpointConfig, CheckpointManager

WORLD_SIZE = 2


class _TinyTransformer(torch.nn.Module):

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(
            torch.tensor([1.25, -2.5, 3.75, -4.0], device=device),
            requires_grad=False,
        )
        self.trainable = torch.nn.Parameter(
            torch.tensor([[0.5, 1.0], [1.5, 2.0]], device=device),
        )
        self.reverse_param_names_mapping: dict[str, tuple[str, None, None]] = {}


class _Method:

    def __init__(self, transformer: torch.nn.Module, base_model_dir: Path, generator: torch.Generator) -> None:
        self.transformer = transformer
        self.base_model_dir = base_model_dir
        self.cuda_generator = generator

    def inference_checkpoint_modules(self, role: str) -> dict[str, torch.nn.Module]:
        if role != "student":
            raise ValueError(role)
        return {"transformer": self.transformer}

    def inference_checkpoint_base_model_path(self, role: str) -> str:
        if role != "student":
            raise ValueError(role)
        return str(self.base_model_dir)


def _write_base_model(base_model_dir: Path) -> None:
    transformer_dir = base_model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "TinyTransformer"}),
        encoding="utf-8",
    )
    filename = "diffusion_pytorch_model-00001-of-00001.safetensors"
    tensors = {
        "frozen": torch.empty(4),
        "trainable": torch.empty(2, 2),
    }
    save_file(tensors, transformer_dir / filename)
    index = {
        "metadata": {
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        },
        "weight_map": {key: filename for key in tensors},
    }
    (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (base_model_dir / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "TinyPipeline"}),
        encoding="utf-8",
    )


def _load_exported_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in sorted((checkpoint_dir / "transformer").glob("*.safetensors")):
        state.update(load_file(shard))
    return state


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "cuda": torch.cuda.get_rng_state().clone(),
        "generator": generator.get_state().clone(),
    }


def _rng_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_numpy = before["numpy"]
    after_numpy = after["numpy"]
    return bool(
        torch.equal(before["torch"], after["torch"])
        and before["python"] == after["python"]
        and before_numpy[0] == after_numpy[0]
        and np.array_equal(before_numpy[1], after_numpy[1])
        and before_numpy[2:] == after_numpy[2:]
        and torch.equal(before["cuda"], after["cuda"])
        and torch.equal(before["generator"], after["generator"])
    )


def _run_worker(work_dir: Path, result_path: Path) -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    try:
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError(f"Expected world size {WORLD_SIZE}, got {dist.get_world_size()}")

        base_model_dir = work_dir / "base"
        output_dir = work_dir / "run"
        if rank == 0:
            _write_base_model(base_model_dir)
        dist.barrier()

        torch.manual_seed(1000 + rank)
        torch.cuda.manual_seed(2000 + rank)
        random.seed(3000 + rank)
        np.random.seed(4000 + rank)
        generator = torch.Generator(device=device)
        generator.manual_seed(5000 + rank)

        transformer = _TinyTransformer(device=device)
        mesh = init_device_mesh("cuda", (WORLD_SIZE, ))
        fully_shard(transformer, mesh=mesh)
        method = _Method(transformer, base_model_dir, generator)
        manager = CheckpointManager(
            method=method,
            dataloader=None,
            output_dir=str(output_dir),
            config=CheckpointConfig(
                save_steps=0,
                keep_last=0,
                save_inference_on_validation=True,
                inference_role="student",
                inference_dtype="float32",
            ),
        )

        success_rng_before = _capture_rng(generator)
        manager.save_inference(1)
        success_rng_after = _capture_rng(generator)

        checkpoint_dir = output_dir / "inference" / "checkpoint-1"
        exported = _load_exported_state(checkpoint_dir)
        reloaded = _TinyTransformer()
        incompatible = reloaded.load_state_dict(exported, strict=True)
        expected = _TinyTransformer().state_dict()
        reload_equal = (not incompatible.missing_keys and not incompatible.unexpected_keys
                        and all(torch.equal(reloaded.state_dict()[key], value) for key, value in expected.items()))
        frozen_present = "frozen" in exported and torch.equal(exported["frozen"], expected["frozen"])

        real_export = inference_checkpoint.export_inference_checkpoint
        if rank == 0:

            def _injected_failure(**_: Any) -> Path:
                raise RuntimeError("injected rank-zero export failure")

            inference_checkpoint.export_inference_checkpoint = _injected_failure

        failure_rng_before = _capture_rng(generator)
        failure: str | None = None
        try:
            manager.save_inference(2)
        except RuntimeError as error:
            failure = str(error)
        finally:
            if rank == 0:
                inference_checkpoint.export_inference_checkpoint = real_export
        failure_rng_after = _capture_rng(generator)

        local_result = {
            "rank": rank,
            "success_rng_equal": _rng_equal(success_rng_before, success_rng_after),
            "failure_rng_equal": _rng_equal(failure_rng_before, failure_rng_after),
            "reload_equal": reload_equal,
            "frozen_present": frozen_present,
            "failure": failure,
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_result)
        if rank == 0:
            result_path.write_text(json.dumps(gathered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_gpu_fsdp2_inference_checkpoint_gate(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA and must run on a compute node.")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"This test requires at least {WORLD_SIZE} CUDA devices.")

    result_path = tmp_path / "results.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(WORLD_SIZE),
        str(Path(__file__).resolve()),
        "--worker",
        "--work-dir",
        str(tmp_path / "worker"),
        "--result",
        str(result_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Two-GPU inference checkpoint worker timed out after 180 seconds\n"
            f"STDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}") from error
    if process.returncode != 0:
        raise RuntimeError(
            f"Two-GPU inference checkpoint worker failed with code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")

    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(results) == WORLD_SIZE
    assert all(result["success_rng_equal"] for result in results)
    assert all(result["failure_rng_equal"] for result in results)
    assert all(result["reload_equal"] for result in results)
    assert all(result["frozen_present"] for result in results)
    errors = [result["failure"] for result in results]
    assert all(error is not None and "injected rank-zero export failure" in error for error in errors)
    assert len(set(errors)) == 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker or args.work_dir is None or args.result is None:
        raise SystemExit("--worker, --work-dir, and --result are required")
    _run_worker(args.work_dir, args.result)
