# SPDX-License-Identifier: Apache-2.0
"""Convert a DCP training checkpoint to a diffusers-style model directory.

Works on a single GPU regardless of how many GPUs were used for training
(DCP handles resharding automatically).

Usage (no torchrun needed)::

    python -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint /path/to/checkpoint-1000 \
        --output-dir /path/to/diffusers_output

Or with torchrun (also fine)::

    torchrun --nproc_per_node=1 \
        -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint ... --output-dir ...

The checkpoint must contain ``metadata.json`` (written by
``CheckpointManager``).  If the checkpoint predates metadata
support, pass ``--config`` explicitly to provide the training
YAML.

Pass ``--verify`` to strictly reload the exported transformer immediately
after writing it, so a key-mapping bug fails here instead of deep inside a
later training/inference launch that loads the exported directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _remove_existing_weight_files(module_dir: Any) -> None:
    """Remove copied weight payloads and indexes before a single-file export."""
    for pattern in ("*.safetensors", "*.safetensors.index.json"):
        for path in module_dir.glob(pattern):
            path.unlink(missing_ok=True)


def _ensure_distributed() -> None:
    """Set up a single-process distributed env if needed.

    When running under ``torchrun`` the env vars are already set.
    For plain ``python`` we fill in the minimum required vars so
    that ``init_process_group`` succeeds with world_size=1.
    """
    for key, default in [
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ]:
        os.environ.setdefault(key, default)


def _role_model_checkpoint_state(method: Any, role: str) -> dict[str, Any]:
    """Return the single model-only DCP entry needed for an export."""
    from fastvideo.training.checkpointing_utils import ModelWrapper

    try:
        model = method._role_models[role]
    except KeyError as error:
        raise KeyError(f"Unknown role {role!r}; available roles: {sorted(method._role_models)}") from error
    if model.transformer is None:
        raise ValueError(f"Role {role!r} does not have a transformer to export")
    return {
        f"roles.{role}.transformer": ModelWrapper(model.transformer),
    }


def _save_role_pretrained(
    *,
    role: str,
    base_model_path: str,
    output_dir: str,
    module_names: list[str] | None = None,
    overwrite: bool = False,
    model: Any,
) -> str:
    """Export a role's modules into a diffusers-style model dir.

    Produces a ``model_path`` loadable by
    ``PipelineComponentLoader`` (``model_index.json``,
    ``transformer/``, ``vae/``, etc. copied from
    ``base_model_path``).
    """
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    from fastvideo.utils import maybe_download_model

    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    local_base = Path(maybe_download_model(str(base_model_path))).resolve()
    dst = Path(os.path.expanduser(str(output_dir))).resolve()

    if _rank() == 0:
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst, ignore_errors=True)
            else:
                raise FileExistsError(f"Refusing to overwrite existing "
                                      f"directory: {dst}. "
                                      "Pass --overwrite to replace it.")

        def _copy_or_link(src: str, dest: str) -> None:
            # Resolve symlinks ourselves: os.link's follow_symlinks=True
            # default isn't honored on all filesystems (e.g. some
            # network/overlay mounts), which can silently hard-link to
            # the symlink itself instead of its target.
            real_src = os.path.realpath(src)
            try:
                os.link(real_src, dest)
            except OSError:
                shutil.copy2(real_src, dest)

        logger.info(
            "Creating pretrained export dir at %s "
            "(base=%s)",
            dst,
            local_base,
        )
        shutil.copytree(
            local_base,
            dst,
            symlinks=False,
            copy_function=_copy_or_link,
        )

    _barrier()

    modules: dict[str, torch.nn.Module] = {}
    if model.transformer is not None:
        modules["transformer"] = model.transformer

    if module_names is None:
        module_names = sorted(modules.keys())

    for module_name in module_names:
        if module_name not in modules:
            raise KeyError(f"Role {role!r} does not have module "
                           f"{module_name!r}. "
                           f"Available: {sorted(modules.keys())}")

        module_dir = dst / module_name
        if not module_dir.is_dir():
            raise FileNotFoundError(f"Export directory missing component "
                                    f"dir {module_name!r}: {module_dir}")

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        state_dict = get_model_state_dict(
            modules[module_name],
            options=options,
        )

        if _rank() == 0:
            _remove_existing_weight_files(module_dir)

            # Convert internal parameter names back to HF format.
            # load_model_from_full_model_state_dict builds reverse_param_names_mapping
            # (internal_key → hf_key) and stores it on the module.  Without this,
            # the exported safetensors would have internal keys (e.g.
            # "patch_embedding.proj.bias") and the next load would double-map them
            # (e.g. → "patch_embedding.proj.proj.bias").
            reverse_mapping: dict = getattr(modules[module_name], "reverse_param_names_mapping", {})

            tensor_state: dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expected tensor in state_dict "
                                    f"for {module_name}.{key}, "
                                    f"got {type(value).__name__}")
                if key in reverse_mapping:
                    hf_key, merge_index, _ = reverse_mapping[key]
                    if merge_index is not None:
                        logger.warning(
                            "Skipping reverse-mapping for merged param %s "
                            "(merge_index=%s); saving under internal key.",
                            key,
                            merge_index,
                        )
                        hf_key = key
                    key = hf_key
                tensor_state[key] = value.detach().cpu()

            from safetensors.torch import save_file

            out_path = module_dir / "model.safetensors"
            logger.info(
                "Saving %s weights to %s (%s tensors)",
                module_name,
                out_path,
                len(tensor_state),
            )
            save_file(tensor_state, str(out_path))

        _barrier()

    return str(dst)


def _strict_reload_verify(
    *,
    output_dir: str,
    training_config: Any,
    attention_backend: Any = None,
) -> None:
    """Reload the just-exported transformer from disk and fail loudly on
    any key mismatch.

    ``TransformerLoader.load()`` already loads strictly (``strict=True``)
    for every non-Cosmos25 model, so this doesn't add leniency -- it moves
    the failure to right after export, with a clear "the export is broken"
    error, instead of surfacing deep inside a later training/inference
    launch that happens to load this directory.
    """
    from fastvideo.train.utils.moduleloader import load_module_from_path

    logger.info("Verifying export: strictly reloading transformer from %s", output_dir)
    load_module_from_path(
        model_path=output_dir,
        module_type="transformer",
        training_config=training_config,
        attention_backend=attention_backend,
    )
    logger.info("Strict reload verification passed.")


def convert(
    *,
    checkpoint_dir: str,
    output_dir: str,
    config_path: str | None = None,
    role: str = "student",
    overwrite: bool = False,
    verify: bool = False,
    load_precision: str | None = None,
) -> str:
    """Load a DCP checkpoint and export as a diffusers model.

    Returns the path to the exported model directory.
    """
    _ensure_distributed()

    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.checkpoint import (
        CheckpointManager,
        _resolve_resume_checkpoint,
    )
    from fastvideo.train.utils.config import (
        RunConfig,
        load_run_config,
    )

    import torch.distributed.checkpoint as dcp

    # -- Resolve checkpoint directory --
    resolved = _resolve_resume_checkpoint(
        checkpoint_dir,
        output_dir=checkpoint_dir,
    )
    dcp_dir = resolved / "dcp"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"Missing dcp/ under {resolved}")

    # -- Obtain config --
    cfg: RunConfig
    if config_path is not None:
        cfg = load_run_config(config_path)
    else:
        metadata = CheckpointManager.load_metadata(resolved)
        raw_config = metadata.get("config")
        if raw_config is None:
            raise ValueError("Checkpoint metadata.json does not "
                             "contain 'config'. Pass --config "
                             "explicitly.")
        cfg = _run_config_from_raw(raw_config)

    tc = cfg.training
    if load_precision is not None:
        if load_precision not in {"bf16", "fp32"}:
            raise ValueError("load_precision must be 'bf16' or 'fp32'")
        tc.dit_precision = load_precision
        if tc.pipeline_config is not None:
            tc.pipeline_config.dit_precision = load_precision
        # A deployment export may intentionally materialize an FP32-master
        # training checkpoint into BF16 inference tensors. Do not let a
        # training-only constructor assertion reject that role load.
        cfg.method["require_fp32_master"] = False

    # -- Init distributed (1 GPU is enough; DCP reshards) --
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
    )

    # Override distributed config so model loading uses 1 GPU.
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1

    # -- Build model (loads pretrained weights + FSDP) --
    _, method, _, _ = build_from_config(cfg)

    # -- Load only the requested role's model weights from DCP --
    #
    # Do not use method.checkpoint_state() here.  Besides the model it exposes
    # optimizer and scheduler state.  DCP materializes missing Adam moments by
    # taking a dummy optimizer step, which can require roughly another 2x the
    # model size and makes an inference-only export OOM unnecessarily.
    states = _role_model_checkpoint_state(method, role)
    model = method._role_models[role]
    logger.info(
        "Loading DCP checkpoint from %s",
        resolved,
    )
    dcp.load(states, checkpoint_id=str(dcp_dir))

    # -- Export to diffusers format --
    base_model_path = str(tc.model_path)
    if not base_model_path:
        raise ValueError("Cannot determine base_model_path from "
                         "config. Ensure models.student.init_from "
                         "is set.")

    logger.info(
        "Exporting role=%s to %s (base=%s)",
        role,
        output_dir,
        base_model_path,
    )
    result = _save_role_pretrained(
        role=role,
        base_model_path=base_model_path,
        output_dir=output_dir,
        overwrite=overwrite,
        model=model,
    )
    logger.info("Export complete: %s", result)

    if verify:
        _strict_reload_verify(
            output_dir=result,
            training_config=tc,
            attention_backend=getattr(model, "attention_backend", None),
        )

    return result


def _run_config_from_raw(raw: dict[str, Any], ) -> Any:
    """Reconstruct a RunConfig from a raw config dict.

    This mirrors ``load_run_config`` but operates on an
    already-parsed dict (from metadata.json) instead of
    reading from a YAML file.
    """
    from fastvideo.train.utils.config import (
        RunConfig,
        _build_training_config,
        _parse_pipeline_config,
        _require_mapping,
        _require_str,
    )

    models_raw = _require_mapping(
        raw.get("models"),
        where="models",
    )
    models: dict[str, dict[str, Any]] = {}
    for role_key, model_cfg_raw in models_raw.items():
        role_str = _require_str(
            role_key,
            where="models.<role>",
        )
        model_cfg = _require_mapping(
            model_cfg_raw,
            where=f"models.{role_str}",
        )
        models[role_str] = dict(model_cfg)

    method_raw = _require_mapping(
        raw.get("method"),
        where="method",
    )
    method = dict(method_raw)

    callbacks_raw = raw.get("callbacks")
    callbacks: dict[str, dict[str, Any]] = (_require_mapping(
        callbacks_raw,
        where="callbacks",
    ) if callbacks_raw is not None else {})

    pipeline_config = _parse_pipeline_config(
        raw,
        models=models,
    )

    training_raw = _require_mapping(
        raw.get("training"),
        where="training",
    )
    t = dict(training_raw)
    training = _build_training_config(
        t,
        models=models,
        pipeline_config=pipeline_config,
    )

    return RunConfig(
        models=models,
        method=method,
        training=training,
        callbacks=callbacks,
        raw=raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=("Convert a DCP training checkpoint to a "
                                                  "diffusers-style model directory. "
                                                  "Only 1 GPU needed (DCP reshards "
                                                  "automatically)."), )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=("Path to checkpoint-<step> dir, its dcp/ "
              "subdir, or an output_dir (auto-picks "
              "latest)."),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination for the diffusers model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Training YAML config. If omitted, read "
              "from checkpoint metadata.json."),
    )
    parser.add_argument(
        "--role",
        type=str,
        default="student",
        help="Role to export (default: student).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=("After exporting, strictly reload the transformer from "
              "the exported directory to catch key-mapping bugs "
              "immediately."),
    )
    parser.add_argument(
        "--load-precision",
        choices=("bf16", "fp32"),
        default=None,
        help=("Optional model materialization precision. Use bf16 to export "
              "an FP32-master training checkpoint for deployment/evaluation."),
    )
    args = parser.parse_args(sys.argv[1:])

    convert(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        role=args.role,
        overwrite=args.overwrite,
        verify=args.verify,
        load_precision=args.load_precision,
    )


if __name__ == "__main__":
    main()
