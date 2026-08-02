# SPDX-License-Identifier: Apache-2.0
"""Load explicitly trusted, verified MotionKernel artifacts."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from fastvideo import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _load_artifact(path: str, operation: str) -> Callable[..., Any]:
    module_name = f"fastvideo_promoted_{operation}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load promoted kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _validate_module(module, operation, path)
    return module.kernel_fn


def _validate_module(
    module: ModuleType,
    operation: str,
    path: str,
) -> None:
    if getattr(module, "KERNEL_TYPE", None) != operation:
        raise ImportError(f"{path} declares KERNEL_TYPE="
                          f"{getattr(module, 'KERNEL_TYPE', None)!r}, expected {operation!r}")
    if not callable(getattr(module, "kernel_fn", None)):
        raise ImportError(f"{path} does not expose callable kernel_fn")


@lru_cache(maxsize=32)
def _find_optimized_kernel(
    artifact_dir: str,
    operation: str,
) -> Callable[..., Any] | None:
    root = Path(artifact_dir).expanduser().resolve()
    matches = sorted(root.glob(f"kernel_{operation}_*_optimized.py"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple promoted kernels found for %s; using %s",
            operation,
            matches[-1],
        )
    selected = matches[-1]
    try:
        return _load_artifact(
            str(selected),
            operation,
        )
    except Exception:
        logger.exception(
            "Unable to load promoted %s kernel; using bundled fallback",
            operation,
        )
        return None


def get_optimized_kernel(operation: str) -> Callable[..., Any] | None:
    """Return a verified artifact selected by operation, or native fallback."""
    artifact_dir = envs.FASTVIDEO_AUTOKERNEL_ARTIFACT_DIR
    if not artifact_dir:
        return None
    return _find_optimized_kernel(artifact_dir, operation)
