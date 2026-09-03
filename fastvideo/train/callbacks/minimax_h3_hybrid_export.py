# SPDX-License-Identifier: Apache-2.0
"""Export trained MiniMax H3 hybrid extras in converter-compatible form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class MiniMaxH3HybridExportCallback(Callback):
    """Gather only trainable hybrid tensors and write ``linear_branch/``."""

    def __init__(self, *, output_dir: str) -> None:
        self._output_dir = Path(output_dir)

    def on_train_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        transformer = method.student.transformer
        patterns = tuple(getattr(method.student, "trainable_parameter_patterns", ()))
        if not patterns:
            raise ValueError("Hybrid export requires explicit trainable parameter patterns")

        state = get_model_state_dict(
            transformer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        if dist.get_rank() == 0:
            tensors = {
                name: value.detach().cpu().contiguous()
                for name, value in state.items()
                if isinstance(value, torch.Tensor) and any(pattern in name for pattern in patterns)
            }
            if not tensors:
                raise RuntimeError("Hybrid export found no trainable tensors")

            branch_dir = self._output_dir / "linear_branch"
            branch_dir.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(branch_dir / "model.safetensors"))

            arch = transformer.config.arch_config
            model_spec: dict[str, Any] = {
                "hybrid_attention": {
                    "softmax_attention": {
                        "radius": int(arch.hybrid_window_radius),
                        "chunk": int(arch.hybrid_window_chunk),
                    },
                    "anchor_frames": str(arch.hybrid_anchor_frames),
                    "enable_softmax_gate": bool(arch.hybrid_enable_softmax_gate),
                    "linear_attention": {
                        "delta_rule": str(arch.hybrid_delta_rule),
                        "enable_text_state": bool(arch.hybrid_enable_text_state),
                        "short_conv": {
                            "targets": list(arch.hybrid_short_conv_targets)
                        },
                    },
                },
                "training": {
                    "teacher": "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2",
                    "sigma_grid_points": 5,
                    "step": int(iteration),
                },
            }
            self._output_dir.mkdir(parents=True, exist_ok=True)
            (self._output_dir / "model_spec.json").write_text(json.dumps(model_spec, indent=2) + "\n")
            logger.info("Exported %d hybrid tensors to %s", len(tensors), self._output_dir)
        dist.barrier()


__all__ = ["MiniMaxH3HybridExportCallback"]
