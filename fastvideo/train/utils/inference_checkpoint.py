# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory export of model-only DCP state for inference.

The modular trainer stores resumable state with PyTorch Distributed
Checkpoint (DCP), while FastVideo inference consumes a Diffusers-style model
directory.  This module bridges those formats without gathering a full model
in memory: rank 0 reads one bounded shard at a time from an already-complete
role/module-only DCP checkpoint and publishes an immutable inference directory
with an atomic rename.

The manager-facing entry point is :func:`export_inference_checkpoint`; the
lower-level :func:`export_inference_checkpoint_from_dcp` exposes the same
rank-0-only conversion with a run-root destination. The checkpoint manager
owns distributed coordination around the temporary DCP save and local export.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

DEFAULT_MAX_SHARD_SIZE_BYTES = 5 * 1024**3
_FORMAT_VERSION = 1
_ALLOWED_NATIVE_EXTRA_KEYS = (
    re.compile(r"(?:^|\.)attn\.to_gate_compress\.weight$"),
)


class InferenceCheckpointExportError(RuntimeError):
    """The model-only checkpoint could not be exported safely."""


class UnsupportedMergedReverseMappingError(InferenceCheckpointExportError):
    """A fused training parameter would need to be split for inference."""


def _is_allowed_native_extra(key: str) -> bool:
    return any(pattern.search(key) for pattern in _ALLOWED_NATIVE_EXTRA_KEYS)


@dataclass(frozen=True, slots=True)
class _TensorPlan:
    checkpoint_key: str
    internal_key: str
    output_key: str
    shape: tuple[int, ...]
    source_dtype: torch.dtype
    output_dtype: torch.dtype
    output_nbytes: int


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _normalize_output_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported inference checkpoint dtype: {dtype!r}")
        dtype = resolved
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or torch dtype name")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"Inference checkpoint dtype must be floating point, got {dtype}")
    return dtype


def _resolve_dcp_dir(checkpoint: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.name != "dcp" and (path / "dcp").is_dir():
        path = path / "dcp"
    if not path.is_dir():
        raise FileNotFoundError(f"Inference checkpoint DCP directory not found: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"Incomplete inference checkpoint DCP (missing .metadata): {path}")
    return path


def validate_complete_inference_checkpoint(path: Path, *, step: int) -> Path | None:
    """Validate a published inference checkpoint without loading its tensors."""
    if not (path.exists() or path.is_symlink()):
        return None
    complete_path = path / ".complete"
    metadata_path = path / "metadata.json"
    if not complete_path.is_file() or not metadata_path.is_file():
        raise InferenceCheckpointExportError(f"Refusing to overwrite incomplete inference checkpoint: {path}")
    try:
        complete_marker = complete_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InferenceCheckpointExportError(f"Cannot read inference completion marker: {complete_path}") from exc
    if complete_marker != "complete\n":
        raise InferenceCheckpointExportError(f"Invalid inference completion marker: {complete_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceCheckpointExportError(
            f"Invalid metadata for existing inference checkpoint: {metadata_path}") from exc
    if (metadata.get("format_version") != _FORMAT_VERSION or metadata.get("kind") != "inference"
            or metadata.get("step") != step):
        raise InferenceCheckpointExportError(
            f"Existing inference checkpoint metadata does not match step={step}: {metadata_path}")
    role = metadata.get("role")
    dtype = metadata.get("dtype")
    if not isinstance(role, str) or not role or "." in role or dtype not in {"bfloat16", "float16", "float32"}:
        raise InferenceCheckpointExportError(f"Inference checkpoint role/dtype metadata is invalid: {metadata_path}")
    module_name = str(metadata.get("module") or "")
    if not module_name or "/" in module_name or "\\" in module_name or module_name in {".", ".."}:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint has an invalid module name {module_name!r}: {metadata_path}")
    module_dir = path / module_name
    index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its module index: {index_path}")
    if not (module_dir / "config.json").is_file():
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its module config: {module_dir}")
    if not any((path / name).is_file() for name in ("model_index.json", "modular_model_index.json")):
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its model index: {path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceCheckpointExportError(f"Invalid inference checkpoint index: {index_path}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise InferenceCheckpointExportError(f"Inference checkpoint index has no weight map: {index_path}")
    expected_by_shard: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str) or not filename:
            raise InferenceCheckpointExportError(f"Invalid key/shard entry in {index_path}: {key!r} -> {filename!r}")
        if Path(filename).name != filename:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint index contains an invalid shard path: {filename!r}")
        expected_by_shard.setdefault(filename, set()).add(key)
    actual_shards = {shard.name for shard in module_dir.glob("*.safetensors") if shard.is_file()}
    expected_shards = set(expected_by_shard)
    if actual_shards != expected_shards:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shards differ from its index under {module_dir}: "
            f"missing={sorted(expected_shards - actual_shards)} extra={sorted(actual_shards - expected_shards)}")
    shard_sizes: list[int] = []
    output_shapes: dict[str, tuple[int, ...]] = {}
    for filename in sorted(expected_by_shard):
        expected_keys = expected_by_shard[filename]
        shard_path = module_dir / filename
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                for key in actual_keys:
                    output_shapes[key] = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
        except Exception as exc:
            raise InferenceCheckpointExportError(f"Cannot read inference checkpoint shard: {shard_path}") from exc
        if actual_keys != expected_keys:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint shard keys differ from its index for {shard_path}: "
                f"missing={sorted(expected_keys - actual_keys)[:10]} extra={sorted(actual_keys - expected_keys)[:10]}")
        shard_sizes.append(shard_path.stat().st_size)
    if int(metadata.get("tensor_count", -1)) != len(weight_map):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint tensor_count does not match its index: {metadata_path}")
    if int(metadata.get("shard_count", -1)) != len(expected_shards):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shard_count does not match its index: {metadata_path}")
    logical_shard_sizes = metadata.get("shard_sizes")
    total_size = metadata.get("total_size")
    max_shard_size = metadata.get("max_shard_size_bytes")
    index_metadata = index.get("metadata")
    index_total_size = index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    if (not isinstance(logical_shard_sizes, list) or len(logical_shard_sizes) != len(expected_shards)
            or any(not isinstance(size, int) or size < 0 for size in logical_shard_sizes)
            or not isinstance(max_shard_size, int) or max_shard_size <= 0
            or any(size > max_shard_size for size in logical_shard_sizes)
            or not isinstance(total_size, int) or total_size != sum(logical_shard_sizes)
            or index_total_size != total_size):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint logical shard sizes are inconsistent: {metadata_path}")
    recorded_shard_sizes = metadata.get("shard_file_sizes")
    if not isinstance(recorded_shard_sizes, list) or recorded_shard_sizes != shard_sizes:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shard_file_sizes do not match files on disk: {metadata_path}")

    base_model_dir = metadata.get("base_model_dir")
    if not isinstance(base_model_dir, str) or not base_model_dir:
        raise InferenceCheckpointExportError(f"Inference checkpoint has no base_model_dir: {metadata_path}")
    base_shapes = _component_tensor_shapes(Path(base_model_dir) / module_name)
    for key, shape in output_shapes.items():
        expected_shape = base_shapes.get(key)
        if expected_shape is None:
            if not _is_allowed_native_extra(key):
                raise InferenceCheckpointExportError(
                    f"Inference checkpoint contains unknown non-native tensor {key!r}: {path}")
        elif shape != expected_shape:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint tensor {key!r} shape {shape} != base transformer shape {expected_shape}")
    missing_base_keys = set(base_shapes) - set(output_shapes)
    if missing_base_keys:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint is missing {len(missing_base_keys)} base transformer tensors; "
            f"first={sorted(missing_base_keys)[:10]}")
    return path


def _component_tensor_shapes(module_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read the base component's exact key/shape contract from safetensors headers."""
    index_candidates = (
        module_dir / "diffusion_pytorch_model.safetensors.index.json",
        module_dir / "model.safetensors.index.json",
    )
    index_path = next((path for path in index_candidates if path.is_file()), None)
    expected_files: set[str] | None = None
    indexed_keys: set[str] | None = None
    if index_path is not None:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceCheckpointExportError(f"Invalid base transformer index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise InferenceCheckpointExportError(f"Base transformer index has no weight map: {index_path}")
        indexed_keys = set(weight_map)
        expected_files = {str(filename) for filename in weight_map.values()}

    files = sorted(module_dir.glob("*.safetensors"))
    if expected_files is not None:
        files = [module_dir / filename for filename in sorted(expected_files)]
    if not files or any(not path.is_file() for path in files):
        raise InferenceCheckpointExportError(f"Base transformer safetensors are incomplete under {module_dir}")

    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in shapes:
                        raise InferenceCheckpointExportError(f"Duplicate base transformer tensor key {key!r}")
                    shapes[key] = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
        except InferenceCheckpointExportError:
            raise
        except Exception as exc:
            raise InferenceCheckpointExportError(f"Cannot read base transformer safetensors: {path}") from exc
    if indexed_keys is not None and set(shapes) != indexed_keys:
        raise InferenceCheckpointExportError(
            f"Base transformer index/header mismatch under {module_dir}: "
            f"missing={sorted(indexed_keys - set(shapes))[:10]} extra={sorted(set(shapes) - indexed_keys)[:10]}")
    return shapes


def _mapping_output_key(
    internal_key: str,
    reverse_mapping: Mapping[str, Any],
) -> str:
    entry = reverse_mapping.get(internal_key)
    if entry is None:
        # FastVideo-native additions such as MiniMax-H3's learned
        # ``attn.to_gate_compress`` VSA parameters intentionally have no key in
        # the base checkpoint.  The component loader accepts their native name.
        return internal_key
    if not isinstance(entry, tuple | list) or len(entry) != 3:
        raise InferenceCheckpointExportError(f"Invalid reverse mapping for {internal_key!r}: expected "
                                             "(output_key, merge_index, num_params_to_merge)")
    output_key, merge_index, num_params_to_merge = entry
    if merge_index is not None or num_params_to_merge not in (None, 1):
        raise UnsupportedMergedReverseMappingError(f"Cannot stream merged reverse mapping for {internal_key!r}: "
                                                   f"output_key={output_key!r}, merge_index={merge_index!r}, "
                                                   f"num_params_to_merge={num_params_to_merge!r}. "
                                                   "A model-specific split exporter is required.")
    if not isinstance(output_key, str) or not output_key:
        raise InferenceCheckpointExportError(
            f"Invalid output key in reverse mapping for {internal_key!r}: {output_key!r}")
    return output_key


def _tensor_output_dtype(source_dtype: torch.dtype, configured_dtype: torch.dtype) -> torch.dtype:
    if torch.empty((), dtype=source_dtype).is_floating_point():
        return configured_dtype
    return source_dtype


def _build_tensor_plan(
    *,
    dcp_dir: Path,
    state_prefix: str,
    reverse_mapping: Mapping[str, Any],
    base_shapes: Mapping[str, tuple[int, ...]],
    output_dtype: torch.dtype,
    max_shard_size_bytes: int,
) -> list[_TensorPlan]:
    metadata = FileSystemReader(str(dcp_dir)).read_metadata()
    plans: list[_TensorPlan] = []
    output_keys: set[str] = set()

    for checkpoint_key in sorted(metadata.state_dict_metadata):
        if not checkpoint_key.startswith(state_prefix):
            continue
        tensor_metadata = metadata.state_dict_metadata[checkpoint_key]
        properties = getattr(tensor_metadata, "properties", None)
        shape = getattr(tensor_metadata, "size", None)
        source_dtype = getattr(properties, "dtype", None)
        if shape is None or not isinstance(source_dtype, torch.dtype):
            raise InferenceCheckpointExportError(f"Inference state contains a non-tensor value at {checkpoint_key!r}; "
                                                 "safetensors exports support tensors only")

        internal_key = checkpoint_key[len(state_prefix):]
        if not internal_key:
            raise InferenceCheckpointExportError(f"Empty module key under DCP prefix {state_prefix!r}")
        output_key = _mapping_output_key(internal_key, reverse_mapping)
        tensor_shape = tuple(int(dim) for dim in shape)
        expected_shape = base_shapes.get(output_key)
        if expected_shape is None:
            if internal_key in reverse_mapping or not _is_allowed_native_extra(output_key):
                raise InferenceCheckpointExportError(
                    f"Inference tensor {internal_key!r} maps to unknown base key {output_key!r}; "
                    "only MiniMax-H3 attn.to_gate_compress.weight may be exported as a native extra")
        elif tensor_shape != expected_shape:
            raise InferenceCheckpointExportError(
                f"Inference tensor {output_key!r} shape {tensor_shape} != base transformer shape {expected_shape}")
        if output_key in output_keys:
            raise InferenceCheckpointExportError(f"Reverse mapping produces duplicate inference key {output_key!r}")
        output_keys.add(output_key)

        tensor_output_dtype = _tensor_output_dtype(source_dtype, output_dtype)
        numel = 1
        for dim in tensor_shape:
            numel *= dim
        output_nbytes = numel * torch.empty((), dtype=tensor_output_dtype).element_size()
        if output_nbytes > max_shard_size_bytes:
            raise InferenceCheckpointExportError(
                f"Tensor {checkpoint_key!r} requires {output_nbytes} bytes after casting, "
                f"which exceeds max_shard_size_bytes={max_shard_size_bytes}")
        plans.append(
            _TensorPlan(
                checkpoint_key=checkpoint_key,
                internal_key=internal_key,
                output_key=output_key,
                shape=tensor_shape,
                source_dtype=source_dtype,
                output_dtype=tensor_output_dtype,
                output_nbytes=output_nbytes,
            ))

    if not plans:
        raise InferenceCheckpointExportError(f"No tensor keys found under DCP prefix {state_prefix!r} in {dcp_dir}")
    missing_base_keys = set(base_shapes) - output_keys
    if missing_base_keys:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint is missing {len(missing_base_keys)} base transformer tensors; "
            f"first={sorted(missing_base_keys)[:10]}")
    return plans


def _group_shards(plans: list[_TensorPlan], max_shard_size_bytes: int) -> list[list[_TensorPlan]]:
    shards: list[list[_TensorPlan]] = []
    current: list[_TensorPlan] = []
    current_nbytes = 0
    for plan in plans:
        if current and current_nbytes + plan.output_nbytes > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_nbytes = 0
        current.append(plan)
        current_nbytes += plan.output_nbytes
    if current:
        shards.append(current)
    return shards


def _prepare_model_layout(temp_dir: Path, base_model_dir: Path, module_name: str) -> Path:
    if not any((base_model_dir / name).is_file() for name in ("model_index.json", "modular_model_index.json")):
        raise FileNotFoundError(
            f"Base model directory has no model_index.json or modular_model_index.json: {base_model_dir}")
    base_module_dir = base_model_dir / module_name
    base_config = base_module_dir / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(f"Base model component config not found: {base_config}")

    module_dir = temp_dir / module_name
    module_dir.mkdir(parents=True)
    shutil.copy2(base_config, module_dir / "config.json")

    reserved = {module_name, "metadata.json", ".complete"}
    for entry in sorted(base_model_dir.iterdir(), key=lambda item: item.name):
        if entry.name in reserved or entry.name == ".cache" or entry.name.startswith(".git"):
            continue
        target = temp_dir / entry.name
        target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    return module_dir


def _write_tensor_shards(
    *,
    dcp_dir: Path,
    module_dir: Path,
    shards: list[list[_TensorPlan]],
) -> tuple[dict[str, str], int, list[int], list[int]]:
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_sizes: list[int] = []
    shard_file_sizes: list[int] = []
    shard_count = len(shards)

    for shard_index, shard in enumerate(shards, start=1):
        filename = (f"diffusion_pytorch_model-{shard_index:05d}-of-{shard_count:05d}.safetensors")
        state = {plan.checkpoint_key: torch.empty(plan.shape, dtype=plan.source_dtype, device="cpu") for plan in shard}
        # This export is deliberately rank-local.  DCP assembles the full CPU
        # tensors from its storage shards without using the live process group.
        dcp.load(state, checkpoint_id=str(dcp_dir), no_dist=True)

        output_tensors: dict[str, torch.Tensor] = {}
        shard_size = 0
        for plan in shard:
            tensor = state[plan.checkpoint_key]
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=plan.output_dtype)
            output_tensors[plan.output_key] = tensor.detach().cpu().contiguous()
            weight_map[plan.output_key] = filename
            shard_size += plan.output_nbytes
        save_file(output_tensors, module_dir / filename)
        total_size += shard_size
        shard_sizes.append(shard_size)
        shard_file_sizes.append((module_dir / filename).stat().st_size)
        del output_tensors, state

    index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map,
    }
    index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return weight_map, total_size, shard_sizes, shard_file_sizes


def export_inference_checkpoint_from_dcp(
    *,
    dcp_checkpoint: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    step: int,
    module: torch.nn.Module,
    base_model_dir: str | os.PathLike[str],
    role: str = "student",
    module_name: str = "transformer",
    dtype: torch.dtype | str = torch.bfloat16,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    raw_config: Mapping[str, Any] | None = None,
) -> Path:
    """Export one role/module-only DCP as an immutable inference checkpoint.

    ``CheckpointManager`` is expected to call this function on rank 0 after a
    collective model-only DCP save has completed. ``dcp_checkpoint`` may name
    that DCP directory directly or its parent containing ``dcp/``. The source
    is never modified or removed.

    Floating tensors are cast into ``dtype`` in independent CPU buffers;
    integer and boolean tensors retain their source dtype. Parameter names are
    converted with ``module.reverse_param_names_mapping``. Unmapped names are
    retained for FastVideo-native inference parameters such as MiniMax-H3 VSA
    gates. Merged mappings fail rather than silently writing incompatible
    weights.

    The completed model is atomically renamed to
    ``<output_dir>/inference/checkpoint-<step>``. A valid existing completed
    export is returned unchanged, making retries idempotent.
    """

    if _rank() != 0:
        raise InferenceCheckpointExportError("export_inference_checkpoint_from_dcp is rank-0-only; "
                                             "the checkpoint manager must coordinate other ranks")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if not role or "." in role:
        raise ValueError(f"role must be a non-empty DCP key segment, got {role!r}")
    if not module_name or "." in module_name:
        raise ValueError(f"module_name must be a non-empty DCP key segment, got {module_name!r}")
    if not 0 < max_shard_size_bytes <= DEFAULT_MAX_SHARD_SIZE_BYTES:
        raise ValueError("max_shard_size_bytes must be in "
                         f"[1, {DEFAULT_MAX_SHARD_SIZE_BYTES}], got {max_shard_size_bytes}")

    output_dtype = _normalize_output_dtype(dtype)
    run_output_dir = Path(output_dir).expanduser().resolve()
    inference_root = run_output_dir / "inference"
    final_dir = inference_root / f"checkpoint-{step}"
    existing = validate_complete_inference_checkpoint(final_dir, step=step)
    if existing is not None:
        return existing

    dcp_dir = _resolve_dcp_dir(dcp_checkpoint)
    base_dir = Path(base_model_dir).expanduser().resolve()
    base_shapes = _component_tensor_shapes(base_dir / module_name)
    reverse_mapping = getattr(module, "reverse_param_names_mapping", {})
    if reverse_mapping is None:
        reverse_mapping = {}
    if not isinstance(reverse_mapping, Mapping):
        raise InferenceCheckpointExportError("module.reverse_param_names_mapping must be a mapping")

    state_prefix = f"roles.{role}.{module_name}."
    plans = _build_tensor_plan(
        dcp_dir=dcp_dir,
        state_prefix=state_prefix,
        reverse_mapping=reverse_mapping,
        base_shapes=base_shapes,
        output_dtype=output_dtype,
        max_shard_size_bytes=max_shard_size_bytes,
    )
    shards = _group_shards(plans, max_shard_size_bytes)

    inference_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(
        prefix=f".checkpoint-{step}.tmp-",
        dir=str(inference_root),
    ))
    try:
        module_dir = _prepare_model_layout(temp_dir, base_dir, module_name)
        weight_map, total_size, shard_sizes, shard_file_sizes = _write_tensor_shards(
            dcp_dir=dcp_dir,
            module_dir=module_dir,
            shards=shards,
        )
        metadata = {
            "format_version": _FORMAT_VERSION,
            "kind": "inference",
            "step": step,
            "role": role,
            "module": module_name,
            "dtype": str(output_dtype).removeprefix("torch."),
            "base_model_dir": str(base_dir),
            "tensor_count": len(weight_map),
            "total_size": total_size,
            "shard_count": len(shards),
            "shard_sizes": shard_sizes,
            "shard_file_sizes": shard_file_sizes,
            "max_shard_size_bytes": max_shard_size_bytes,
        }
        if raw_config is not None:
            metadata["config"] = raw_config
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Written last inside the private temp directory. The subsequent
        # same-filesystem rename publishes the complete tree in one operation.
        (temp_dir / ".complete").write_text("complete\n", encoding="utf-8")
        try:
            temp_dir.rename(final_dir)
        except FileExistsError as exc:
            raise InferenceCheckpointExportError(f"Inference checkpoint appeared concurrently: {final_dir}") from exc
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    validated = validate_complete_inference_checkpoint(final_dir, step=step)
    if validated is None:
        raise InferenceCheckpointExportError(f"Published inference checkpoint disappeared: {final_dir}")
    return validated


def export_inference_checkpoint(
    *,
    dcp_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    base_model_path: str | os.PathLike[str],
    role: str,
    modules: Mapping[str, torch.nn.Module],
    dtype: torch.dtype | str,
    step: int,
    raw_config: Mapping[str, Any] | None = None,
) -> Path:
    """CheckpointManager adapter for one deployable role/module checkpoint.

    ``output_dir`` is the manager's fully resolved target,
    ``<run>/inference/checkpoint-<step>``. The temporary DCP may contain only
    the role/module state selected by ``modules``. Multi-component deployment
    is intentionally rejected until its component layout and atomicity
    contract are defined.

    ``raw_config`` is persisted in ``metadata.json`` when supplied, matching
    resumable checkpoint provenance. The ephemeral DCP staging path is not
    persisted because the manager removes it after a successful export.
    """

    if len(modules) != 1:
        raise InferenceCheckpointExportError("Inference checkpoint export currently supports exactly one module; "
                                             f"got {sorted(modules)}")
    module_name, module = next(iter(modules.items()))
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Inference checkpoint module {module_name!r} must be a torch.nn.Module")

    final_dir = Path(output_dir).expanduser().resolve()
    expected_name = f"checkpoint-{step}"
    if final_dir.name != expected_name or final_dir.parent.name != "inference":
        raise ValueError("CheckpointManager output_dir must be "
                         f"<run>/inference/{expected_name}, got {final_dir}")
    run_output_dir = final_dir.parent.parent
    return export_inference_checkpoint_from_dcp(
        dcp_checkpoint=dcp_dir,
        output_dir=run_output_dir,
        step=step,
        module=module,
        base_model_dir=base_model_path,
        role=role,
        module_name=module_name,
        dtype=dtype,
        raw_config=raw_config,
    )


__all__ = [
    "DEFAULT_MAX_SHARD_SIZE_BYTES",
    "InferenceCheckpointExportError",
    "UnsupportedMergedReverseMappingError",
    "export_inference_checkpoint",
    "export_inference_checkpoint_from_dcp",
    "validate_complete_inference_checkpoint",
]
