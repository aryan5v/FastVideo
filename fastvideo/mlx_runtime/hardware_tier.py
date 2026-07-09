# SPDX-License-Identifier: Apache-2.0
"""Hardware-adaptive model + quant tiering for Apple Silicon MLX.

Picks a practical FastWan config from the Mac's unified-memory size so a 16 GB
machine defaults to a small INT8 model with a tight MLX allocator cap, while
32/64 GB machines can run higher fidelity (fp16, fuller decoders, larger caps).

Pure recommendation helpers are unit-testable with injected memory sizes; the
detection path reads ``sysctl hw.memsize`` on macOS and falls back to MLX
``device_info`` / a safe default when Metal or sysctl is unavailable.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

GIB = 1024**3

# Thresholds are inclusive upper bounds for each tier band (easy to retune).
TIER_SMALL_MAX_GIB = 18.0
TIER_MEDIUM_MAX_GIB = 40.0

# Recommended MLX allocator caps leave headroom for the OS, torch/MPS encode,
# and decode. Distinct from the benchmark *stress* presets that pin closer to
# the machine class (e.g. mac-16gb uses a 16 GiB stress cap).
TIER_SMALL_MLX_CAP_GIB = 12.0
TIER_MEDIUM_MLX_CAP_GIB = 24.0
TIER_LARGE_MLX_CAP_GIB = 48.0

MODEL_1_3B_REPO = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"

# TODO(Track D): once the 5B MLX port is parity-green on Metal, set this to the
# Diffusers repo id (candidate: "FastVideo/FastWan2.2-TI2V-5B-Diffusers").
# Until then ``recommend_tier(..., prefer_5b=True)`` falls back to 1.3B configs
# when this is ``None``.
FIVE_B_MODEL_REPO: str | None = None

# Non-Mac / undetectable-memory fallback: choose the safest small tier.
DEFAULT_ASSUMED_MEMORY_GIB = 16.0

QuantName = Literal["int8", "none"]
DecoderName = Literal["taehv", "wan-vae"]


@dataclass(frozen=True)
class HardwareTier:
    """Immutable recommended config for one unified-memory band."""

    name: str
    """Human-readable tier id: ``small`` / ``medium`` / ``large``."""

    max_memory_gib: float | None
    """Inclusive upper bound of this band; ``None`` means unbounded (large)."""

    model_repo: str
    """Hugging Face Diffusers repo id for the recommended checkpoint."""

    quantization: QuantName
    """``"int8"`` for weight-only affine quant; ``"none"`` for fp16 weights."""

    decoder: DecoderName
    """Default decode path for this tier."""

    mlx_memory_limit_gib: float
    """Suggested MLX allocator cap (GiB)."""

    benchmark_preset: str
    """Matching key in ``BENCHMARK_PRESETS`` (``mac-16gb`` / ``mac-32gb`` / ``mac-64gb``)."""

    modes: str
    """Comma-separated benchmark modes applied by ``--auto-tier``."""

    decoders: str
    """Comma-separated benchmark decoders applied by ``--auto-tier``."""

    uses_5b: bool = False
    """True when the recommendation selected a 5B checkpoint (Track D)."""


def _tier_small() -> HardwareTier:
    return HardwareTier(
        name="small",
        max_memory_gib=TIER_SMALL_MAX_GIB,
        model_repo=MODEL_1_3B_REPO,
        quantization="int8",
        decoder="taehv",
        mlx_memory_limit_gib=TIER_SMALL_MLX_CAP_GIB,
        benchmark_preset="mac-16gb",
        modes="int8",
        decoders="taehv",
        uses_5b=False,
    )


def _tier_medium(*, prefer_5b: bool, five_b_model_repo: str | None) -> HardwareTier:
    use_5b = bool(prefer_5b and five_b_model_repo)
    if use_5b:
        # 5B INT8 + TAEHV once Track D lands.
        return HardwareTier(
            name="medium",
            max_memory_gib=TIER_MEDIUM_MAX_GIB,
            model_repo=five_b_model_repo,  # type: ignore[arg-type]
            quantization="int8",
            decoder="taehv",
            mlx_memory_limit_gib=TIER_MEDIUM_MLX_CAP_GIB,
            benchmark_preset="mac-32gb",
            modes="int8",
            decoders="taehv",
            uses_5b=True,
        )
    # Fallback until 5B is available: 1.3B fp16 (higher quality than small's int8).
    return HardwareTier(
        name="medium",
        max_memory_gib=TIER_MEDIUM_MAX_GIB,
        model_repo=MODEL_1_3B_REPO,
        quantization="none",
        decoder="taehv",
        mlx_memory_limit_gib=TIER_MEDIUM_MLX_CAP_GIB,
        benchmark_preset="mac-32gb",
        modes="fp16",
        decoders="taehv",
        uses_5b=False,
    )


def _tier_large(*, prefer_5b: bool, five_b_model_repo: str | None) -> HardwareTier:
    use_5b = bool(prefer_5b and five_b_model_repo)
    if use_5b:
        return HardwareTier(
            name="large",
            max_memory_gib=None,
            model_repo=five_b_model_repo,  # type: ignore[arg-type]
            quantization="none",
            decoder="wan-vae",
            mlx_memory_limit_gib=TIER_LARGE_MLX_CAP_GIB,
            benchmark_preset="mac-64gb",
            modes="fp16",
            decoders="wan-vae",
            uses_5b=True,
        )
    return HardwareTier(
        name="large",
        max_memory_gib=None,
        model_repo=MODEL_1_3B_REPO,
        quantization="none",
        decoder="wan-vae",
        mlx_memory_limit_gib=TIER_LARGE_MLX_CAP_GIB,
        benchmark_preset="mac-64gb",
        modes="fp16",
        decoders="wan-vae",
        uses_5b=False,
    )


def _bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / float(GIB)


def _sysctl_memsize_bytes() -> int | None:
    """Return total physical/unified memory via macOS ``sysctl``, or None."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2.0)
        value = int(out.strip())
        return value if value > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _mlx_device_memory_bytes(mx_module: Any | None = None) -> int | None:
    """Return ``memory_size`` from MLX device_info when Metal is available."""
    try:
        if mx_module is None:
            import mlx.core as mlx_core
            mx_module = mlx_core
        info = None
        if hasattr(mx_module, "device_info"):
            info = mx_module.device_info()
        elif hasattr(mx_module, "metal") and hasattr(mx_module.metal, "device_info"):
            # Older MLX: metal.device_info (deprecated in favour of mx.device_info).
            if not mx_module.metal.is_available():
                return None
            info = mx_module.metal.device_info()
        if not isinstance(info, dict):
            return None
        memory_size = info.get("memory_size")
        if memory_size is None:
            return None
        value = int(memory_size)
        return value if value > 0 else None
    except Exception:  # noqa: BLE001 - optional path; any failure → caller falls back.
        return None


def _proc_meminfo_bytes() -> int | None:
    """Linux fallback: parse MemTotal from ``/proc/meminfo``."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    # kB
                    parts = line.split()
                    kib = int(parts[1])
                    return kib * 1024 if kib > 0 else None
    except (OSError, ValueError, IndexError):
        return None
    return None


def detect_unified_memory_gib(*, mx_module: Any | None = None) -> float:
    """Detect total unified/system memory in GiB.

    Preference order:
    1. macOS ``sysctl -n hw.memsize`` (true unified-memory size)
    2. MLX ``device_info()["memory_size"]`` when Metal is present
    3. Linux ``/proc/meminfo`` MemTotal
    4. :data:`DEFAULT_ASSUMED_MEMORY_GIB` (safe small-tier assumption)
    """
    for probe in (
            _sysctl_memsize_bytes,
            lambda: _mlx_device_memory_bytes(mx_module),
            _proc_meminfo_bytes,
    ):
        num_bytes = probe()
        if num_bytes is not None and num_bytes > 0:
            return _bytes_to_gib(num_bytes)
    return DEFAULT_ASSUMED_MEMORY_GIB


def recommend_tier(
    memory_gib: float | None = None,
    *,
    prefer_5b: bool = True,
    five_b_model_repo: str | None = None,
    mx_module: Any | None = None,
) -> HardwareTier:
    """Recommend a :class:`HardwareTier` for the given (or detected) memory.

    Args:
        memory_gib: Injected unified memory in GiB. When ``None``, calls
            :func:`detect_unified_memory_gib`. Pure/unit tests should pass this
            explicitly so no Metal or sysctl is required.
        prefer_5b: When True and a 5B repo id is known, medium/large tiers select
            the 5B checkpoint. When the 5B id is unset (Track D not landed),
            falls back to 1.3B configs.
        five_b_model_repo: Explicit 5B Diffusers repo id for this call. When
            ``None``, falls back to :data:`FIVE_B_MODEL_REPO` (also ``None``
            until Track D). Tests inject a fake id here to exercise the 5B path.
        mx_module: Optional MLX module for detection (injection / tests).
    """
    repo = five_b_model_repo if five_b_model_repo is not None else FIVE_B_MODEL_REPO

    if memory_gib is None:
        memory_gib = detect_unified_memory_gib(mx_module=mx_module)
    if memory_gib <= 0:
        # Defensive: treat nonsense as the safe small tier.
        memory_gib = DEFAULT_ASSUMED_MEMORY_GIB

    if memory_gib <= TIER_SMALL_MAX_GIB:
        return _tier_small()
    if memory_gib <= TIER_MEDIUM_MAX_GIB:
        return _tier_medium(prefer_5b=prefer_5b, five_b_model_repo=repo)
    return _tier_large(prefer_5b=prefer_5b, five_b_model_repo=repo)


def apply_tier_to_namespace(args: Any, tier: HardwareTier) -> HardwareTier:
    """Mutate an argparse namespace with tier modes, decoders, and MLX caps.

    Leaves ``model_root`` alone (local path vs HF id); callers that resolve
    checkpoints from a repo id should read ``tier.model_repo``. Sets
    ``args.auto_tier_name`` / ``args.auto_tier_model_repo`` for metrics.
    """
    args.modes = tier.modes
    args.decoders = tier.decoders
    args.mlx_memory_limit_gib = tier.mlx_memory_limit_gib
    # Memory-tier runs disable the MLX cache so the allocator cap is meaningful.
    args.mlx_disable_cache = True
    args.auto_tier_name = tier.name
    args.auto_tier_model_repo = tier.model_repo
    args.auto_tier_quantization = tier.quantization
    args.auto_tier_benchmark_preset = tier.benchmark_preset
    return tier
