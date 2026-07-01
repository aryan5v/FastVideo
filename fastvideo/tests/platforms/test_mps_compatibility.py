# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the Apple Silicon compatibility baseline."""

import torch

import fastvideo.distributed.parallel_state as parallel_state
import fastvideo.platforms as platforms
from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.platforms.cpu import CpuPlatform
from fastvideo.platforms.mps import MpsPlatform


def test_local_torch_device_uses_mps_only_for_mps(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "_current_platform", MpsPlatform())
    assert parallel_state.get_local_torch_device() == torch.device("mps")

    monkeypatch.setattr(platforms, "_current_platform", CpuPlatform())
    assert parallel_state.get_local_torch_device() == torch.device("cpu")


def test_mps_uses_fp16_eager_compatibility_settings(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "_current_platform", MpsPlatform())
    pipeline_config = FastWan2_1_T2V_480P_Config()
    args = FastVideoArgs(
        model_path="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        pipeline_config=pipeline_config,
        enable_torch_compile=True,
        dit_cpu_offload=True,
        dit_layerwise_offload=True,
        pin_cpu_memory=True,
    )

    assert args.disable_autocast is True
    assert args.enable_torch_compile is False
    assert args.dit_cpu_offload is False
    assert args.dit_layerwise_offload is False
    assert args.pin_cpu_memory is False
    assert pipeline_config.dit_precision == "fp16"
    assert pipeline_config.precision == "fp16"
    assert pipeline_config.vae_tiling is True
    assert pipeline_config.vae_precision == "fp16"


def test_mps_reports_unified_memory_and_device_name() -> None:
    # These introspection helpers are dependency-free (os.sysconf / sysctl
    # fallback) so they return sane values even on non-Mac test runners. They
    # feed benchmark labels and memory-aware config tiers.
    total_memory = MpsPlatform.get_device_total_memory()
    assert isinstance(total_memory, int)
    assert total_memory > 0

    device_name = MpsPlatform.get_device_name()
    assert isinstance(device_name, str)
    assert device_name != ""

    # Metal exposes no CUDA-style compute capability; must be None, not a raise.
    assert MpsPlatform.get_device_capability() is None
