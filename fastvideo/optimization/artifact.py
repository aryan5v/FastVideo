# SPDX-License-Identifier: Apache-2.0
"""Trusted artifact bundle loading for generic graph dispatch.

A bundle is a directory holding an ``artifact.json`` manifest plus the files it
pins by SHA-256. The manifest is produced by MotionKernel, which owns the
normative schema; this module is the consumer half and deliberately re-derives
the checks rather than importing the producer, exactly as
:mod:`fastvideo.optimization.fx_capture` re-derives the graph fingerprint.

Three rules hold everywhere in this module:

* Executable code is imported only from inside the explicitly configured
  trusted root. Nothing here reads a path from the manifest and follows it.
* Every declared file is hashed and compared before the entry point is
  imported.
* Anything unexpected -- an unknown schema version, a missing field, a changed
  byte -- is a hard rejection with a structured reason, never a best-effort
  load.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

from fastvideo.logger import init_logger

logger = init_logger(__name__)

#: Manifest format understood here. A bundle declaring anything else is
#: rejected rather than parsed optimistically.
SUPPORTED_ARTIFACT_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "artifact.json"

#: Wildcard accepted by the string-valued compatibility fields.
ANY = "*"

#: Synthetic package the bundles are imported under, so they never shadow a
#: real module for the rest of the process.
_MODULE_NAMESPACE = "fastvideo._artifacts"

_IGNORED_DIRECTORIES = frozenset({"__pycache__"})
_READ_CHUNK_BYTES = 1 << 20

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RELATIVE_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,255}$")
_IR_NODE_PATTERN = re.compile(r"^n[0-9]+$")
_IR_REF_PATTERN = re.compile(r"^[pn][0-9]+$")

# Structured rejection reasons. These are logged and exported verbatim, so they
# are part of the contract with the producer.
REASON_FINGERPRINT_MISMATCH = "fingerprint_mismatch"
REASON_INPUT_SIGNATURE_MISMATCH = "input_signature_mismatch"
REASON_OUTPUT_SIGNATURE_MISMATCH = "output_signature_mismatch"
REASON_MODEL_MISMATCH = "model_mismatch"
REASON_REVISION_MISMATCH = "model_revision_mismatch"
REASON_ARCHITECTURE_MISMATCH = "gpu_architecture_mismatch"
REASON_TORCH_VERSION = "torch_version_unsupported"
REASON_CUDA_VERSION = "cuda_version_unsupported"
REASON_TRITON_VERSION = "triton_version_unsupported"
REASON_EXECUTION_MODE = "execution_mode_unsupported"
REASON_DISTRIBUTED_MODE = "distributed_mode_unsupported"
REASON_NOT_PROMOTED = "not_promoted"
REASON_EVIDENCE_INCOMPLETE = "evidence_incomplete"


class ArtifactError(ValueError):
    """Raised when a bundle is malformed, altered, or unsafe to load."""


def _fail(source: object, location: str, message: str) -> ArtifactError:
    return ArtifactError(f"artifact bundle {source!r}: {location}: {message}")


def _mapping(value: Any, source: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise _fail(source, location, "must be a non-empty object")
    for key in value:
        if not isinstance(key, str) or not key:
            raise _fail(source, location, "keys must be non-empty strings")
    return value


def _sequence(value: Any, source: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(source, location, "must be a list")
    return value


def _text(value: Any, source: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, location, "must be a non-empty string")
    return value


def _bool(value: Any, source: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(source, location, "must be a bool")
    return value


def _pattern(value: Any, pattern: re.Pattern[str], source: object, location: str, description: str) -> str:
    text = _text(value, source, location)
    if not pattern.fullmatch(text):
        raise _fail(source, location, f"must be {description}")
    return text


def _relative_path(value: Any, source: object, location: str) -> str:
    text = _pattern(value, _RELATIVE_FILE_PATTERN, source, location, "a relative POSIX path")
    if any(part in ("", ".", "..") for part in text.split("/")):
        raise _fail(source, location, "must not contain empty or relative segments")
    return text


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Parse a leading dotted-numeric version, ignoring local/pre-release tags.

    ``"2.8.0+cu128"`` and ``"2.8.0a0"`` both parse to ``(2, 8, 0)``. ``None`` is
    returned when there is no numeric prefix, which callers treat as "cannot
    compare" rather than "compatible".
    """
    if text is None:
        return None
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    padded_left = left + (0, ) * (width - len(left))
    padded_right = right + (0, ) * (width - len(right))
    if padded_left < padded_right:
        return -1
    return 0 if padded_left == padded_right else 1


@dataclass(frozen=True)
class VersionRange:
    """An inclusive-minimum, exclusive-maximum version window."""

    minimum: str | None = None
    maximum_exclusive: str | None = None

    @classmethod
    def from_dict(cls, raw: Any, *, source: object, location: str) -> VersionRange:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise _fail(source, location, "must be an object")
        unknown = sorted(set(raw) - {"min", "max_exclusive"})
        if unknown:
            raise _fail(source, location, f"unknown field(s) {unknown}")
        minimum = raw.get("min")
        maximum = raw.get("max_exclusive")
        if minimum is not None:
            minimum = _text(minimum, source, f"{location}.min")
        if maximum is not None:
            maximum = _text(maximum, source, f"{location}.max_exclusive")
        low = parse_version(minimum)
        high = parse_version(maximum)
        if minimum is not None and low is None:
            raise _fail(source, f"{location}.min", "must be a dotted version")
        if maximum is not None and high is None:
            raise _fail(
                source,
                f"{location}.max_exclusive",
                "must be a dotted version",
            )
        if low is not None and high is not None and _compare_versions(low, high) >= 0:
            raise _fail(source, location, "min must be lower than max_exclusive")
        return cls(minimum=minimum, maximum_exclusive=maximum)

    @property
    def unbounded(self) -> bool:
        return self.minimum is None and self.maximum_exclusive is None

    def contains(self, version: str | None) -> bool:
        """Whether ``version`` satisfies this range.

        A bound that cannot be evaluated -- because the runtime version is
        missing or unparsable -- is never assumed to hold.
        """
        if self.unbounded:
            return True
        observed = parse_version(version)
        if observed is None:
            return False
        if self.minimum is not None:
            low = parse_version(self.minimum)
            if low is None or _compare_versions(observed, low) < 0:
                return False
        if self.maximum_exclusive is not None:
            high = parse_version(self.maximum_exclusive)
            if high is None or _compare_versions(observed, high) >= 0:
                return False
        return True

    def describe(self) -> str:
        if self.unbounded:
            return "any"
        low = self.minimum if self.minimum is not None else "any"
        high = self.maximum_exclusive if self.maximum_exclusive is not None else "any"
        return f">={low},<{high}"


def signature_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    """The comparable identity of one tensor signature.

    Built from the capture module's tensor metadata so a live invocation and a
    packaged artifact are compared on exactly the same fields. ``name`` is
    excluded: argument naming is a runtime detail, not part of the layout.
    """
    return (
        tuple(int(dim) for dim in meta.get("shape", ())),
        tuple(int(step) for step in meta.get("stride", ())),
        str(meta.get("dtype", "")),
        str(meta.get("device_type", "")),
        bool(meta.get("requires_grad", False)),
    )


def _signature_keys(raw: Any, source: object, location: str) -> tuple[tuple[Any, ...], ...]:
    items = _sequence(raw, source, location)
    if not items:
        raise _fail(source, location, "must be a non-empty list")
    keys = []
    for index, item in enumerate(items):
        entry = _mapping(item, source, f"{location}[{index}]")
        unknown = sorted(set(entry) - {"name", "shape", "stride", "dtype", "device_type", "requires_grad"})
        if unknown:
            raise _fail(
                source,
                f"{location}[{index}]",
                f"unknown field(s) {unknown}",
            )
        shape = _sequence(entry.get("shape"), source, f"{location}[{index}].shape")
        stride = _sequence(entry.get("stride"), source, f"{location}[{index}].stride")
        for values, name in ((shape, "shape"), (stride, "stride")):
            for position, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise _fail(
                        source,
                        f"{location}[{index}].{name}[{position}]",
                        "must be an integer",
                    )
                if name == "shape" and value < 0:
                    raise _fail(
                        source,
                        f"{location}[{index}].shape[{position}]",
                        "must be non-negative",
                    )
        if len(shape) != len(stride):
            raise _fail(
                source,
                f"{location}[{index}].stride",
                "must have the same length as shape",
            )
        _text(entry.get("dtype"), source, f"{location}[{index}].dtype")
        _text(entry.get("device_type"), source, f"{location}[{index}].device_type")
        _bool(
            entry.get("requires_grad", False),
            source,
            f"{location}[{index}].requires_grad",
        )
        keys.append(signature_key(entry))
    return tuple(keys)


@dataclass(frozen=True)
class ArtifactFile:
    """One bundled file, pinned by size and content hash."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ArtifactManifest:
    """The subset of a bundle manifest this runtime acts on."""

    artifact_id: str
    graph_fingerprint: str
    operation_name: str
    parent_module: str
    target_kind: str
    capture_mode: str | None
    selected_node_ids: tuple[str, ...]
    boundary_refs: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    input_keys: tuple[tuple[Any, ...], ...]
    output_keys: tuple[tuple[Any, ...], ...]
    entry_file: str
    entry_symbol: str
    files: tuple[ArtifactFile, ...]
    model_id: str
    model_revision: str
    gpu_architectures: tuple[str, ...]
    torch_range: VersionRange
    cuda_range: VersionRange
    triton_range: VersionRange
    execution_modes: tuple[str, ...]
    distributed_modes: tuple[str, ...]
    promotion_decision: str
    benchmark_passed: bool
    generation_passed: bool
    evidence_passed: bool
    speedup: float
    directory: Path

    @classmethod
    def from_dict(cls, raw_value: Any, *, directory: Path) -> ArtifactManifest:
        source = str(directory)
        raw = _mapping(raw_value, source, "top level")
        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "schema_version", "must be an integer")
        if version != SUPPORTED_ARTIFACT_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version}; expected "
                f"{SUPPORTED_ARTIFACT_SCHEMA_VERSION}",
            )

        operation = _mapping(raw.get("operation"), source, "operation")
        signature = _mapping(raw.get("signature"), source, "signature")
        entry_point = _mapping(raw.get("entry_point"), source, "entry_point")
        compatibility = _mapping(raw.get("compatibility"), source, "compatibility")
        evidence = _mapping(raw.get("evidence"), source, "evidence")
        promotion = _mapping(raw.get("promotion"), source, "promotion")

        files = []
        for index, item in enumerate(_sequence(raw.get("files"), source, "files")):
            entry = _mapping(item, source, f"files[{index}]")
            size = entry.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise _fail(source, f"files[{index}].bytes", "must be a non-negative integer")
            files.append(
                ArtifactFile(
                    path=_relative_path(entry.get("path"), source, f"files[{index}].path"),
                    sha256=_pattern(
                        entry.get("sha256"),
                        _SHA256_PATTERN,
                        source,
                        f"files[{index}].sha256",
                        "64 lowercase hex characters",
                    ),
                    size=size,
                ))
        if not files:
            raise _fail(source, "files", "must be a non-empty list")
        paths = [item.path for item in files]
        if len(paths) != len(set(paths)):
            raise _fail(source, "files", "contains duplicate paths")

        entry_file = _relative_path(entry_point.get("file"), source, "entry_point.file")
        if entry_file not in paths:
            raise _fail(source, "entry_point.file", f"{entry_file!r} is not a declared file")
        if not entry_file.endswith(".py"):
            raise _fail(source, "entry_point.file", "must be a .py file")

        benchmark = _mapping(evidence.get("benchmark"), source, "evidence.benchmark")
        generation = _mapping(evidence.get("generation"), source, "evidence.generation")
        speedup = benchmark.get("speedup")
        if (isinstance(speedup, bool) or not isinstance(speedup, int | float) or not math.isfinite(float(speedup))
                or float(speedup) < 0):
            raise _fail(
                source,
                "evidence.benchmark.speedup",
                "must be a finite non-negative number",
            )

        architectures = _sequence(
            compatibility.get("gpu_architectures"),
            source,
            "compatibility.gpu_architectures",
        )
        execution_modes = _sequence(
            compatibility.get("execution_modes"),
            source,
            "compatibility.execution_modes",
        )
        distributed_modes = _sequence(
            compatibility.get("distributed_modes"),
            source,
            "compatibility.distributed_modes",
        )
        for items, location in (
            (architectures, "compatibility.gpu_architectures"),
            (execution_modes, "compatibility.execution_modes"),
            (distributed_modes, "compatibility.distributed_modes"),
        ):
            if not items:
                raise _fail(source, location, "must be a non-empty list")

        target_kind = operation.get("target_kind", "module")
        if target_kind not in {"module", "subgraph"}:
            raise _fail(source, "operation.target_kind", "must be 'module' or 'subgraph'")
        rewrite_fields = {
            "capture_mode",
            "selected_node_ids",
            "boundary_refs",
            "output_node_ids",
        }
        if target_kind == "module" and rewrite_fields.intersection(operation):
            raise _fail(
                source,
                "operation",
                "module targets must not declare subgraph rewrite fields",
            )

        capture_mode: str | None = None
        selected_node_ids: tuple[str, ...] = ()
        boundary_refs: tuple[str, ...] = ()
        output_node_ids: tuple[str, ...] = ()
        if target_kind == "subgraph":
            capture_mode = _text(operation.get("capture_mode"), source, "operation.capture_mode")
            if capture_mode != "export":
                raise _fail(
                    source,
                    "operation.capture_mode",
                    "subgraph dispatch currently requires 'export'",
                )

            def refs(field: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
                items = _sequence(operation.get(field), source, f"operation.{field}")
                if not items:
                    raise _fail(source, f"operation.{field}", "must be a non-empty list")
                result = tuple(
                    _pattern(
                        item,
                        pattern,
                        source,
                        f"operation.{field}[{index}]",
                        "a canonical executable-IR reference",
                    ) for index, item in enumerate(items))
                if len(result) != len(set(result)):
                    raise _fail(source, f"operation.{field}", "must not contain duplicates")
                return result

            selected_node_ids = refs("selected_node_ids", _IR_NODE_PATTERN)
            boundary_refs = refs("boundary_refs", _IR_REF_PATTERN)
            output_node_ids = refs("output_node_ids", _IR_NODE_PATTERN)
            if not set(output_node_ids).issubset(selected_node_ids):
                raise _fail(source, "operation.output_node_ids", "must be selected nodes")

        return cls(
            artifact_id=_text(raw.get("artifact_id"), source, "artifact_id"),
            graph_fingerprint=_pattern(
                operation.get("graph_fingerprint"),
                _FINGERPRINT_PATTERN,
                source,
                "operation.graph_fingerprint",
                "32 lowercase hex characters",
            ),
            operation_name=_text(operation.get("name"), source, "operation.name"),
            parent_module=_text(operation.get("parent_module"), source, "operation.parent_module"),
            target_kind=target_kind,
            capture_mode=capture_mode,
            selected_node_ids=selected_node_ids,
            boundary_refs=boundary_refs,
            output_node_ids=output_node_ids,
            input_keys=_signature_keys(signature.get("inputs"), source, "signature.inputs"),
            output_keys=_signature_keys(signature.get("outputs"), source, "signature.outputs"),
            entry_file=entry_file,
            entry_symbol=_pattern(
                entry_point.get("symbol"),
                _SYMBOL_PATTERN,
                source,
                "entry_point.symbol",
                "a Python identifier",
            ),
            files=tuple(files),
            model_id=_text(compatibility.get("model_id"), source, "compatibility.model_id"),
            model_revision=_text(
                compatibility.get("model_revision"),
                source,
                "compatibility.model_revision",
            ),
            gpu_architectures=tuple(
                _text(item, source, f"compatibility.gpu_architectures[{index}]")
                for index, item in enumerate(architectures)),
            torch_range=VersionRange.from_dict(
                compatibility.get("torch"),
                source=source,
                location="compatibility.torch",
            ),
            cuda_range=VersionRange.from_dict(
                compatibility.get("cuda"),
                source=source,
                location="compatibility.cuda",
            ),
            triton_range=VersionRange.from_dict(
                compatibility.get("triton"),
                source=source,
                location="compatibility.triton",
            ),
            execution_modes=tuple(
                _text(item, source, f"compatibility.execution_modes[{index}]")
                for index, item in enumerate(execution_modes)),
            distributed_modes=tuple(
                _text(item, source, f"compatibility.distributed_modes[{index}]")
                for index, item in enumerate(distributed_modes)),
            promotion_decision=_text(promotion.get("decision"), source, "promotion.decision"),
            benchmark_passed=_bool(benchmark.get("passed"), source, "evidence.benchmark.passed"),
            generation_passed=_bool(generation.get("passed"), source, "evidence.generation.passed"),
            evidence_passed=(_bool(benchmark.get("passed"), source, "evidence.benchmark.passed")
                             and _bool(generation.get("passed"), source, "evidence.generation.passed")),
            speedup=float(speedup),
            directory=directory,
        )


@dataclass(frozen=True)
class RuntimeProfile:
    """The environment this process is executing in."""

    model_id: str
    model_revision: str
    gpu_architecture: str
    torch_version: str
    cuda_version: str | None = None
    triton_version: str | None = None
    execution_mode: str = "inference"
    distributed_mode: str = "single"

    @classmethod
    def detect(
        cls,
        *,
        model_id: str,
        model_revision: str = ANY,
        execution_mode: str = "inference",
        distributed_mode: str = "single",
    ) -> RuntimeProfile:
        """Read torch/CUDA/Triton identity from the running process."""
        import torch

        architecture = "cpu"
        cuda_version = None
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(index)
            architecture = f"sm{major}{minor}"
            cuda_version = getattr(torch.version, "cuda", None)
        try:
            import triton

            triton_version = getattr(triton, "__version__", None)
        except Exception:  # noqa: BLE001 - Triton is optional on every platform
            triton_version = None
        return cls(
            model_id=model_id,
            model_revision=model_revision,
            gpu_architecture=architecture,
            torch_version=str(getattr(torch, "__version__", "")),
            cuda_version=cuda_version,
            triton_version=triton_version,
            execution_mode=execution_mode,
            distributed_mode=distributed_mode,
        )


def _wildcard_equal(declared: str, observed: str) -> bool:
    return declared in (ANY, observed)


def check_compatibility(
    manifest: ArtifactManifest,
    *,
    graph_fingerprint: str,
    input_keys: tuple[tuple[Any, ...], ...],
    output_keys: tuple[tuple[Any, ...], ...],
    runtime: RuntimeProfile,
    validation: bool = False,
) -> str | None:
    """Return the reason ``manifest`` cannot serve this call, or ``None``.

    Ordering is cheapest-first: graph identity, then tensor layout, then the
    declared environment window.
    """
    if manifest.graph_fingerprint != graph_fingerprint:
        return REASON_FINGERPRINT_MISMATCH
    if manifest.input_keys != input_keys:
        return REASON_INPUT_SIGNATURE_MISMATCH
    if manifest.output_keys != output_keys:
        return REASON_OUTPUT_SIGNATURE_MISMATCH
    if not _wildcard_equal(manifest.model_id, runtime.model_id):
        return REASON_MODEL_MISMATCH
    if not _wildcard_equal(manifest.model_revision, runtime.model_revision):
        return REASON_REVISION_MISMATCH
    if not any(_wildcard_equal(item, runtime.gpu_architecture) for item in manifest.gpu_architectures):
        return REASON_ARCHITECTURE_MISMATCH
    if not manifest.torch_range.contains(runtime.torch_version):
        return REASON_TORCH_VERSION
    if not manifest.cuda_range.contains(runtime.cuda_version):
        return REASON_CUDA_VERSION
    if not manifest.triton_range.contains(runtime.triton_version):
        return REASON_TRITON_VERSION
    if runtime.execution_mode not in manifest.execution_modes:
        return REASON_EXECUTION_MODE
    if runtime.distributed_mode not in manifest.distributed_modes:
        return REASON_DISTRIBUTED_MODE
    if validation:
        if manifest.promotion_decision not in {"promoted", "quarantined"}:
            return REASON_NOT_PROMOTED
        if not manifest.benchmark_passed:
            return REASON_EVIDENCE_INCOMPLETE
        return None
    if manifest.promotion_decision != "promoted":
        return REASON_NOT_PROMOTED
    if not manifest.evidence_passed:
        return REASON_EVIDENCE_INCOMPLETE
    return None


def file_sha256(path: Path) -> str:
    """Content hash of one file, streamed so large bundles stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_contents(directory: Path) -> set[str]:
    present = set()
    for path in directory.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(directory)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        present.add(relative.as_posix())
    return present


def verify_bundle(directory: Path) -> ArtifactManifest:
    """Parse a bundle and confirm every declared file is byte-for-byte intact.

    Undeclared files are refused as well: an attacker who can drop an extra
    module next to a signed entry point could otherwise have it imported.
    """
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise _fail(str(directory), MANIFEST_FILENAME, "not found")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(str(directory), MANIFEST_FILENAME, f"invalid JSON: {exc}") from exc
    manifest = ArtifactManifest.from_dict(raw, directory=directory)

    declared = {item.path for item in manifest.files}
    undeclared = sorted(_bundle_contents(directory) - declared - {MANIFEST_FILENAME})
    if undeclared:
        raise _fail(str(directory), "files", f"undeclared file(s) {undeclared}")

    for entry in sorted(manifest.files, key=lambda item: item.path):
        path = directory / entry.path
        if not path.is_file():
            raise _fail(str(directory), "files", f"{entry.path!r} is missing")
        size = path.stat().st_size
        if size != entry.size:
            raise _fail(
                str(directory),
                "files",
                f"{entry.path!r} is {size} bytes, manifest records {entry.size}",
            )
        actual = file_sha256(path)
        if actual != entry.sha256:
            raise _fail(
                str(directory),
                "files",
                f"{entry.path!r} hash {actual} does not match manifest {entry.sha256}",
            )
    return manifest


def _resolve_inside(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to stay under ``root``."""
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ArtifactError(f"artifact bundle {str(candidate)!r}: resolves outside the trusted "
                            f"root {str(resolved_root)!r}")
    return resolved


def load_entry_point(manifest: ArtifactManifest, *, trusted_root: Path) -> Callable[..., Any]:
    """Re-verify a bundle and import its candidate callable.

    The bundle is verified again here even if the caller already validated it:
    the gap between validation and import is precisely where a swapped file
    would land.
    """
    directory = _resolve_inside(trusted_root, manifest.directory)
    verified = verify_bundle(directory)
    if verified.artifact_id != manifest.artifact_id:
        raise _fail(
            str(directory),
            "artifact_id",
            f"changed from {manifest.artifact_id!r} to {verified.artifact_id!r} "
            "since validation",
        )

    entry_file = _resolve_inside(directory, directory / verified.entry_file)
    readable_id = re.sub(r"[^A-Za-z0-9_]", "_", verified.artifact_id)
    unique_id = hashlib.sha256(verified.artifact_id.encode("utf-8")).hexdigest()[:16]
    module_name = f"{_MODULE_NAMESPACE}.{readable_id}_{unique_id}"
    spec = importlib.util.spec_from_file_location(module_name, entry_file)
    if spec is None or spec.loader is None:
        raise _fail(str(directory), "entry_point", f"cannot load {verified.entry_file!r}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the module can look itself up, and removed
    # again on failure so a half-initialized module is never reachable.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - untrusted code, any failure is a rejection
        sys.modules.pop(module_name, None)
        raise _fail(
            str(directory),
            "entry_point",
            f"importing {verified.entry_file!r} raised {type(exc).__name__}",
        ) from exc

    candidate = getattr(module, verified.entry_symbol, None)
    if candidate is None or not callable(candidate):
        sys.modules.pop(module_name, None)
        raise _fail(
            str(directory),
            "entry_point",
            f"{verified.entry_symbol!r} is not a callable in {verified.entry_file!r}",
        )
    return candidate


def discover_bundles(root: Path) -> list[Path]:
    """List bundle directories under ``root``: the root itself plus children."""
    if not root.is_dir():
        return []
    found = []
    if (root / MANIFEST_FILENAME).is_file():
        found.append(root)
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file():
            found.append(child)
    return found


class ArtifactRegistry:
    """Every valid bundle found in the trusted artifact directory.

    Loading is eager and one-shot: bundles are parsed and hashed when the
    registry is built, so a corrupt artifact surfaces before generation starts
    rather than mid-run. Bundles that fail validation are recorded in
    :attr:`errors` and skipped; they never disable the ones that passed.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.manifests: list[ArtifactManifest] = []
        self.errors: list[str] = []
        if root is None:
            return
        if not root.is_dir():
            self.errors.append(f"artifact directory {str(root)!r}: not a directory")
            return
        for path in discover_bundles(root):
            try:
                self.manifests.append(verify_bundle(path))
            except ArtifactError as exc:
                self.errors.append(str(exc))

    @property
    def enabled(self) -> bool:
        return bool(self.manifests)

    def candidates_for(self, input_keys: tuple[tuple[Any, ...], ...]) -> list[ArtifactManifest]:
        """Bundles whose input layout matches, before any graph is traced.

        This is the cheap pre-filter that keeps dispatch from tracing a module
        when the store holds nothing shaped like the live call.
        """
        return [item for item in self.manifests if item.target_kind == "module" and item.input_keys == input_keys]

    def subgraph_candidates_for(self, scope: str) -> list[ArtifactManifest]:
        """Export-subgraph bundles declared for this repeated module scope."""
        return [item for item in self.manifests if item.target_kind == "subgraph" and item.parent_module == scope]

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root) if self.root is not None else None,
            "loaded": len(self.manifests),
            "artifact_ids": sorted(item.artifact_id for item in self.manifests),
            "errors": list(self.errors),
        }
