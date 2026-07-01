# SPDX-License-Identifier: Apache-2.0

import os

import torch

from fastvideo.logger import init_logger
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.platforms.interface import (DeviceCapability, Platform, PlatformEnum)

logger = init_logger(__name__)


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    device_control_env_var: str = "MPS_VISIBLE_DEVICES"
    simple_compile_backend: str = "eager"

    @classmethod
    def get_torch_device(cls):
        """Return the MPS module for callers that need a torch device handle."""
        return torch.mps

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        # Metal has no CUDA-style compute-capability tuple. Return ``None`` so
        # capability-gated code paths (e.g. FP4/FP8 kernels) treat MPS as
        # "unknown / unsupported" rather than crashing.
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Best-effort human-readable chip name (e.g. ``Apple M3 Max``).

        Used to label benchmark rows and pick memory-aware config tiers. Falls
        back to the machine architecture when the SoC brand string is not
        available (e.g. off-Mac test runners).
        """
        import platform as _platform
        import subprocess

        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
        return _platform.machine() or "mps"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        # Apple Silicon exposes no per-device UUID; the SoC name is the most
        # stable identifier available.
        return cls.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Total unified memory in bytes.

        On Apple Silicon the GPU shares system RAM, so the total unified-memory
        budget is the amount of physical RAM. Uses ``os.sysconf`` to stay
        dependency-free and portable across macOS and Linux CI.
        """
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError, AttributeError):
            return 0

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning("To see benefits of async output processing, enable MPS "
                           "graph. Since, enforce-eager is enabled, async output "
                           "processor cannot be used")
            return False
        return True

    @classmethod
    def get_current_memory_usage(cls, device: torch.types.Device | None = None) -> float:
        """Current MPS allocation in bytes (0.0 when unavailable)."""
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        if current_allocated is not None:
            try:
                return float(current_allocated())
            except (RuntimeError, AttributeError):
                return 0.0
        return 0.0

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: AttentionBackendEnum | None, head_size: int,
                             dtype: torch.dtype) -> str:
        # MPS supports SDPA (Scaled Dot-Product Attention) which is the most compatible
        logger.info("Using Torch SDPA backend for MPS.")
        return "fastvideo.attention.backends.sdpa.SDPABackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Use base communicator for MPS
        return "fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """Set the seed for MPS device."""
        if seed is not None:
            import random

            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # MPS doesn't have manual_seed_all like CUDA
            # The manual_seed above should be sufficient
