# SPDX-License-Identifier: Apache-2.0
"""Export converter-ready MiniMax H3 hybrid weights from a DCP checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def export_hybrid(
    *,
    checkpoint_dir: str,
    output_dir: str,
    config_path: str | None = None,
    step: int | None = None,
) -> str:
    """Load one training checkpoint on one GPU and export hybrid tensors."""
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.train.callbacks.minimax_h3_hybrid_export import (
        MiniMaxH3HybridExportCallback,
    )
    from fastvideo.train.entrypoint.dcp_to_diffusers import (
        _ensure_distributed,
        _run_config_from_raw,
    )
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.checkpoint import (
        CheckpointManager,
        _parse_step_from_dir,
        _resolve_resume_checkpoint,
    )
    from fastvideo.train.utils.config import load_run_config

    import torch.distributed.checkpoint as dcp

    _ensure_distributed()
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if (checkpoint_path / "dcp").is_dir():
        # Node-local snapshots are deliberately staged without their original
        # checkpoint-<step> parent name so they can be streamed independently
        # of checkpoint rotation in the live training job.
        resolved = checkpoint_path
    else:
        resolved = _resolve_resume_checkpoint(
            checkpoint_dir,
            output_dir=checkpoint_dir,
        )
    if config_path is None:
        metadata = CheckpointManager.load_metadata(resolved)
        raw_config = metadata.get("config")
        if raw_config is None:
            raise ValueError(
                "Checkpoint metadata does not contain config; pass --config."
            )
        cfg = _run_config_from_raw(raw_config)
    else:
        cfg = load_run_config(config_path)

    tc = cfg.training
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    _, method, _, _ = build_from_config(cfg)
    logger.info("Loading DCP checkpoint from %s", resolved)
    dcp.load(method.checkpoint_state(), checkpoint_id=str(resolved / "dcp"))

    if step is None:
        step = _parse_step_from_dir(resolved)
    callback = MiniMaxH3HybridExportCallback(output_dir=output_dir)
    callback.on_train_end(method, iteration=step)
    logger.info("Hybrid step-%s export complete: %s", step, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step, required when the staged directory is not named checkpoint-<step>.",
    )
    args = parser.parse_args()
    export_hybrid(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        step=args.step,
    )


if __name__ == "__main__":
    main()
