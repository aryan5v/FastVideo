# SPDX-License-Identifier: Apache-2.0
"""Generic graph dispatch: run a packaged kernel, or fall back to native.

Dispatch attaches to the same model-independent structure the capture pipeline
uses -- stacks of identically-typed children under an ``nn.ModuleList`` -- and
decides, per stack and per observed input signature, whether a trusted artifact
may replace the native forward. Nothing in this module knows the name of a
model, a block class, or an architecture; a new model is supported by
publishing an artifact, never by editing code here.

The decision sequence for one scope and input signature is:

1. The first call always runs natively. It is what reveals the output
   signature, which is part of the artifact's identity.
2. The registry is pre-filtered on the input layout. If nothing in the store
   is shaped like this call, no graph is ever traced.
3. The module is traced once to recompute its graph fingerprint, using the
   capture module so the value matches what the producer recorded.
4. Compatibility is checked, the winning bundle is re-verified and imported,
   and the callable is cached for every later call with that signature.

Every step is guarded. A failure at any point demotes the signature to native
execution permanently and records a structured reason; it never propagates to
the caller. With no artifact directory configured, nothing is attached at all
and the model runs byte-for-byte as it does today.

Candidate calling convention
----------------------------
A bundle's entry point is called as ``candidate(module, *args, **kwargs)``:
the native module comes first, then the exact arguments its ``forward`` was
given. Passing the module is what lets one artifact serve every block in a
stack -- the kernel reads the parameters it needs from the module it was handed
instead of the loader having to know which parameters exist.
"""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

from torch import nn

from fastvideo import envs
from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.logger import init_logger
from fastvideo.optimization import timing
from fastvideo.optimization.artifact import (
    ArtifactManifest,
    ArtifactRegistry,
    RuntimeProfile,
    check_compatibility,
    load_entry_point,
    signature_key,
)
from fastvideo.optimization.fx_capture import capture_export_invocation, default_capture_targets
from fastvideo.optimization.identity import (
    graph_identity,
    input_signatures,
    output_signatures,
    shape_key_for,
)
from fastvideo.optimization.subgraph import rewrite_exported_subgraph, subgraph_signature_keys

logger = init_logger(__name__)

#: Diagnostic report format. Bumped independently of the capture and profiler
#: schemas so a consumer can tell the three apart.
DISPATCH_SCHEMA_VERSION = 1

# Structured fallback reasons. Consumers group on these, so they are stable.
REASON_SELECTED = "artifact_selected"
REASON_NO_SIGNATURE_MATCH = "no_artifact_for_input_signature"
REASON_NO_COMPATIBLE_ARTIFACT = "no_compatible_artifact"
REASON_NO_TENSOR_INPUTS = "no_tensor_inputs"
_REASON_IDENTITY_PREFIX = "graph_identity_unavailable"
_REASON_LOAD_PREFIX = "artifact_load_failed"
_REASON_RUNTIME_PREFIX = "candidate_runtime_error"
_REASON_DECISION_PREFIX = "decision_failed"


@dataclass
class _Decision:
    """What to do for one scope and input signature, and why."""

    candidate: Callable[..., Any] | None
    reason: str
    artifact_id: str | None = None
    rejections: tuple[str, ...] = ()
    #: Calls made *after* this decision was resolved. The one call that
    #: produced it ran natively and is not counted here.
    calls: int = 0
    candidate_calls: int = 0
    runtime_fallbacks: int = 0

    def as_dict(self, scope: str, shape_key: str) -> dict[str, Any]:
        return {
            "scope": scope,
            "shape_key": shape_key,
            "reason": self.reason,
            "artifact_id": self.artifact_id,
            "rejections": list(self.rejections),
            "calls": self.calls,
            "candidate_calls": self.candidate_calls,
            "runtime_fallbacks": self.runtime_fallbacks,
            "active": self.candidate is not None,
        }


@dataclass
class _Wrapper:
    """One patched module, remembered so ``detach`` can restore it exactly."""

    scope: str
    module: nn.Module
    parameter_manager: nn.Module
    native_forward: Callable[..., Any]
    had_own_forward: bool = False


class GraphDispatchSession:
    """Route repeated module stacks through trusted artifacts when possible."""

    def __init__(
        self,
        registry: ArtifactRegistry,
        runtime: RuntimeProfile,
        *,
        tracer: str = "symbolic",
        max_scopes: int = 64,
        max_shape_variants: int = 8,
        validation: bool = False,
    ) -> None:
        self.registry = registry
        self.runtime = runtime
        self.tracer = tracer
        self.max_scopes = max_scopes
        self.max_shape_variants = max_shape_variants
        self.validation = validation
        self._wrappers: list[_Wrapper] = []
        self._scopes: set[str] = set()
        self._decisions: dict[tuple[str, str], _Decision] = {}
        self._dropped_variants: dict[str, int] = defaultdict(int)
        self._dropped_scopes = 0
        self._errors: list[str] = []

    # -- attach ---------------------------------------------------------

    def attach_modules(self, modules: dict[str, Any] | None) -> int:
        """Wrap every repeated block stack in a pipeline's module mapping."""
        if not modules or not self.registry.enabled:
            return 0
        wrapped = 0
        for name, module in modules.items():
            if isinstance(module, nn.Module):
                wrapped += self.attach(module, prefix=str(name))
        return wrapped

    def attach(self, root: nn.Module, *, prefix: str = "") -> int:
        """Wrap every repeated block stack under ``root``."""
        try:
            targets = default_capture_targets(root)
        except Exception as exc:  # noqa: BLE001 - dispatch never breaks generation
            self._record_exception("target_selection_failed", exc)
            return 0
        parents = {
            id(child): parent
            for parent in root.modules()
            for child in parent.children()
        }
        wrapped = 0
        for scope, module in targets:
            qualified = f"{prefix}.{scope}" if prefix else scope
            if qualified not in self._scopes and len(self._scopes) >= self.max_scopes:
                self._dropped_scopes += 1
                continue
            try:
                self._install(
                    qualified,
                    module,
                    parameter_manager=self._parameter_manager(
                        root, module, parents
                    ),
                )
                self._scopes.add(qualified)
                wrapped += 1
            except Exception as exc:  # noqa: BLE001
                self._record_exception("install_failed", exc, scope=qualified)
        return wrapped

    @staticmethod
    def _parameter_manager(
        root: nn.Module,
        module: nn.Module,
        parents: dict[int, nn.Module],
    ) -> nn.Module:
        current = module
        while True:
            if callable(getattr(current, "unshard", None)) and callable(
                getattr(current, "reshard", None)
            ):
                return current
            parent = parents.get(id(current))
            if parent is None or parent is current:
                return root
            current = parent

    def _install(
        self,
        scope: str,
        module: nn.Module,
        *,
        parameter_manager: nn.Module,
    ) -> None:
        """Replace ``module.forward`` with the dispatching wrapper."""
        native_forward = module.forward
        wrapper = _Wrapper(
            scope=scope,
            module=module,
            parameter_manager=parameter_manager,
            native_forward=native_forward,
            had_own_forward="forward" in vars(module),
        )

        def dispatching_forward(*args: Any, **kwargs: Any) -> Any:
            return self._dispatch(wrapper, args, kwargs)

        module.forward = dispatching_forward  # type: ignore[method-assign]
        self._wrappers.append(wrapper)

    def detach(self) -> None:
        """Restore every native forward. Safe to call more than once."""
        for wrapper in reversed(self._wrappers):
            try:
                if wrapper.had_own_forward:
                    wrapper.module.forward = wrapper.native_forward  # type: ignore[method-assign]
                else:
                    # The wrapper shadowed the class-level method; deleting the
                    # instance attribute restores the original binding exactly.
                    del wrapper.module.forward  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                self._record_exception("detach_failed", exc, scope=wrapper.scope)
        self._wrappers.clear()

    # -- dispatch -------------------------------------------------------

    def _dispatch(self, wrapper: _Wrapper, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        native = wrapper.native_forward
        try:
            with timing.phase("dispatch.shape_key"):
                input_metas = input_signatures(args, kwargs)
                shape_key = shape_key_for(input_metas)
        except Exception as exc:  # noqa: BLE001
            self._record_exception("shape_key_failed", exc, scope=wrapper.scope)
            return native(*args, **kwargs)

        key = (wrapper.scope, shape_key)
        decision = self._decisions.get(key)
        if decision is None:
            # First call for this signature: run native, learn the output
            # layout, then decide once for every later call.
            with timing.phase("dispatch.native_reference"):
                output = native(*args, **kwargs)
            self._decisions[key] = self._decide(wrapper, args, kwargs, output, input_metas)
            return output

        decision.calls += 1
        if decision.candidate is None:
            with timing.phase("dispatch.native_fallback"):
                return native(*args, **kwargs)

        def candidate_forward(*candidate_args: Any, **candidate_kwargs: Any) -> Any:
            assert decision.candidate is not None
            return decision.candidate(
                wrapper.module,
                *candidate_args,
                **candidate_kwargs,
            )

        if timing.SHADOW:
            # Same module, same live tensors, same stream: the only honest
            # comparison for "is the artifact path cheaper than what it
            # replaced". The result is discarded.
            with timing.phase("shadow.native_forward"):
                native(*args, **kwargs)

        try:
            with timing.phase("dispatch.candidate_total"), self._materialized_candidate_parameters(
                wrapper.parameter_manager
            ):
                hook_manager = ModuleHookManager.get_from(wrapper.module)
                if hook_manager is None:
                    result = candidate_forward(*args, **kwargs)
                else:
                    result = hook_manager.run_with_forward(
                        candidate_forward,
                        *args,
                        **kwargs,
                    )
        except Exception as exc:  # noqa: BLE001 - untrusted candidate code
            # Demote permanently: a candidate that raised once is not trusted
            # to be retried thousands of times over the rest of the run.
            decision.runtime_fallbacks += 1
            decision.candidate = None
            decision.reason = f"{_REASON_RUNTIME_PREFIX}:{type(exc).__name__}"
            # The FSDP-lifecycle snapshot is built here, on the failure path,
            # rather than three times per successful call. It exists to
            # diagnose parameter materialization; a run that never fails never
            # needs it, and paying for it on every dispatch charged the
            # candidate's measured saving for diagnostics it did not use.
            logger.warning(
                "Artifact %s failed at runtime for %s; falling back to native "
                "execution for the rest of this run; materialization=%s",
                decision.artifact_id,
                wrapper.scope,
                json.dumps(
                    {"at_failure": self._safe_parameter_snapshot(wrapper)},
                    sort_keys=True,
                ),
                exc_info=True,
            )
            return native(*args, **kwargs)
        decision.candidate_calls += 1
        return result

    @staticmethod
    def _parameter_snapshot(wrapper: _Wrapper) -> dict[str, Any]:
        """Return bounded tensor metadata for diagnosing FSDP lifecycle gaps."""
        module_parameters = tuple(wrapper.module.parameters(recurse=False))
        manager = wrapper.parameter_manager
        hook_manager = ModuleHookManager.get_from(wrapper.module)
        return {
            "hook_manager": hook_manager is not None,
            "hook_names": (
                sorted(str(name) for name in hook_manager.forward_hooks)
                if hook_manager is not None
                else []
            ),
            "manager_has_lifecycle": callable(getattr(manager, "unshard", None))
            and callable(getattr(manager, "reshard", None)),
            "manager_is_module": manager is wrapper.module,
            "manager_type": type(manager).__name__,
            "module_direct_parameter_count": len(module_parameters),
            "module_direct_parameter_shapes": [
                list(parameter.shape) for parameter in module_parameters[:16]
            ],
            "module_type": type(wrapper.module).__name__,
        }

    @staticmethod
    def _safe_parameter_snapshot(wrapper: _Wrapper) -> dict[str, Any]:
        """Never let best-effort failure diagnostics suppress native fallback."""
        try:
            return GraphDispatchSession._parameter_snapshot(wrapper)
        except Exception as exc:  # noqa: BLE001 - diagnostics are never authoritative
            return {"snapshot_error": type(exc).__name__}

    @contextmanager
    def _materialized_candidate_parameters(self, module: nn.Module):
        """Mirror FSDP2's forward lifecycle when dispatch bypasses its wrapper."""
        unshard = getattr(module, "unshard", None)
        reshard = getattr(module, "reshard", None)
        managed = callable(unshard) and callable(reshard)
        if not managed:
            yield
            return
        try:
            try:
                unshard(async_op=False)
            except TypeError:
                unshard()
            yield
        finally:
            reshard()

    def _decide(
        self,
        wrapper: _Wrapper,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        input_metas: list[dict[str, Any]],
    ) -> _Decision:
        """Resolve one signature to an artifact or to a native fallback."""
        try:
            return self._resolve(wrapper, args, kwargs, output, input_metas)
        except Exception as exc:  # noqa: BLE001 - resolution is best effort
            self._record_exception("decide_failed", exc, scope=wrapper.scope)
            return _Decision(None, f"{_REASON_DECISION_PREFIX}:{type(exc).__name__}")

    def _resolve(
        self,
        wrapper: _Wrapper,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        input_metas: list[dict[str, Any]],
    ) -> _Decision:
        scope = wrapper.scope
        if self._variant_count(scope) >= self.max_shape_variants:
            self._dropped_variants[scope] += 1
            return _Decision(None, "shape_variant_budget_exhausted")

        if not input_metas:
            return _Decision(None, REASON_NO_TENSOR_INPUTS)
        input_keys = tuple(signature_key(meta) for meta in input_metas)
        module_candidates = self.registry.candidates_for(input_keys)
        subgraph_candidates = self.registry.subgraph_candidates_for(scope)
        candidates = module_candidates + subgraph_candidates
        if not candidates:
            return _Decision(None, REASON_NO_SIGNATURE_MATCH)

        module_region: dict[str, Any] | None = None
        export_region: dict[str, Any] | None = None
        exported: Any | None = None
        try:
            with self._native_forward_for_identity(wrapper):
                if module_candidates:
                    module_region = graph_identity(
                        wrapper.module,
                        args,
                        kwargs,
                        output,
                        scope=scope,
                        tracer=self.tracer,
                    )
                if subgraph_candidates:
                    export_region, exported = capture_export_invocation(
                        wrapper.module,
                        args,
                        kwargs,
                        output,
                        scope=scope,
                    )
        except Exception as exc:  # noqa: BLE001 - untraceable modules stay native
            reason = f"{_REASON_IDENTITY_PREFIX}:{type(exc).__name__}"
            logger.info("Graph identity unavailable for %s: %s", scope, reason)
            return _Decision(None, reason)

        module_output_keys = tuple(signature_key(meta) for meta in output_signatures(output))
        matched: list[ArtifactManifest] = []
        rejections: list[str] = []
        for manifest in candidates:
            if manifest.target_kind == "subgraph":
                if export_region is None:
                    rejections.append(f"{manifest.artifact_id}:export_capture_missing")
                    continue
                candidate_input_keys, candidate_output_keys = subgraph_signature_keys(
                    export_region,
                    manifest,
                )
                fingerprint = str(export_region.get("fingerprint", ""))
            else:
                if module_region is None:
                    rejections.append(f"{manifest.artifact_id}:module_capture_missing")
                    continue
                candidate_input_keys = input_keys
                candidate_output_keys = module_output_keys
                fingerprint = str(module_region.get("fingerprint", ""))
            compatibility_reason = check_compatibility(
                manifest,
                graph_fingerprint=fingerprint,
                input_keys=candidate_input_keys,
                output_keys=candidate_output_keys,
                runtime=self.runtime,
                validation=self.validation,
            )
            if compatibility_reason is None:
                matched.append(manifest)
            else:
                rejections.append(f"{manifest.artifact_id}:{compatibility_reason}")
        if not matched:
            return _Decision(
                None,
                REASON_NO_COMPATIBLE_ARTIFACT,
                rejections=tuple(sorted(rejections)),
            )

        best = max(matched, key=lambda item: (item.speedup, item.artifact_id))
        rejections.extend(f"{item.artifact_id}:not_selected" for item in matched if item is not best)
        root = self.registry.root
        if root is None:
            return _Decision(None, f"{_REASON_LOAD_PREFIX}:NoTrustedRoot")
        try:
            candidate = load_entry_point(best, trusted_root=root)
            if best.target_kind == "subgraph":
                if exported is None:
                    raise RuntimeError("export capture missing")
                candidate = rewrite_exported_subgraph(
                    exported,
                    best,
                    candidate,
                )
        except Exception as exc:  # noqa: BLE001 - a bad bundle must not stop generation
            logger.warning("Artifact %s could not be loaded; staying native", best.artifact_id, exc_info=True)
            return _Decision(
                None,
                f"{_REASON_LOAD_PREFIX}:{type(exc).__name__}",
                artifact_id=best.artifact_id,
                rejections=tuple(sorted(rejections)),
            )
        logger.info(
            "Dispatching %s to artifact %s (fingerprint %s)",
            scope,
            best.artifact_id,
            best.graph_fingerprint,
        )
        return _Decision(
            candidate,
            REASON_SELECTED,
            artifact_id=best.artifact_id,
            rejections=tuple(sorted(rejections)),
        )

    @contextmanager
    def _native_forward_for_identity(self, wrapper: _Wrapper):
        """Expose the real forward while export/Dynamo recompute identity.

        Symbolic FX traces the class method, but export and Dynamo call the
        instance forward. Leaving the dispatch wrapper installed would recurse
        back into this session during identity capture.
        """
        installed_forward = wrapper.module.forward
        wrapper.module.forward = wrapper.native_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            wrapper.module.forward = installed_forward  # type: ignore[method-assign]

    def _variant_count(self, scope: str) -> int:
        return sum(1 for existing_scope, _ in self._decisions if existing_scope == scope)

    # -- diagnostics ----------------------------------------------------

    def _record_error(self, message: str) -> None:
        if len(self._errors) < 64:
            self._errors.append(message)

    def _record_exception(self, code: str, exc: Exception, *, scope: str | None = None) -> None:
        # Exception text can carry argument reprs; only the type is recorded.
        location = f"[{scope}]" if scope else ""
        self._record_error(f"{code}{location}:{type(exc).__name__}")

    def diagnostics(self) -> dict[str, Any]:
        """A metadata-only report of every dispatch decision made."""
        decisions = [
            decision.as_dict(scope, shape_key) for (scope, shape_key), decision in sorted(self._decisions.items())
        ]
        reason_counts: dict[str, int] = defaultdict(int)
        for decision in decisions:
            reason_counts[str(decision["reason"])] += 1
        return {
            "dispatch": {
                "dispatch_schema_version": DISPATCH_SCHEMA_VERSION,
                "registry": self.registry.summary(),
                "runtime": {
                    "model_id": self.runtime.model_id,
                    "model_revision": self.runtime.model_revision,
                    "gpu_architecture": self.runtime.gpu_architecture,
                    "torch_version": self.runtime.torch_version,
                    "cuda_version": self.runtime.cuda_version,
                    "triton_version": self.runtime.triton_version,
                    "execution_mode": self.runtime.execution_mode,
                    "distributed_mode": self.runtime.distributed_mode,
                },
                "tracer": self.tracer,
                "validation": self.validation,
                "scopes": sorted(self._scopes),
                "dropped_scopes": self._dropped_scopes,
                "dropped_shape_variants": dict(sorted(self._dropped_variants.items())),
                "errors": list(self._errors),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
            "decisions": decisions,
        }

    def write_diagnostics(self, path: str | Path) -> Path | None:
        """Write the diagnostic report, returning ``None`` if that failed."""
        output = Path(path).expanduser()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.diagnostics(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            logger.warning("Could not write dispatch diagnostics to %s", output, exc_info=True)
            return None
        return output


def _execution_mode(modules: dict[str, Any] | None) -> str:
    """Report ``training`` when any hooked module is in training mode."""
    for module in (modules or {}).values():
        if isinstance(module, nn.Module) and any(child.training for child in module.modules()):
            return "training"
    return "inference"


def _distributed_mode() -> str:
    """Report the sharding mode this process runs in.

    A multi-rank run whose mode is not declared explicitly reports
    ``unspecified``, which matches no artifact: an unsharded kernel silently
    applied to a sharded module would be a correctness bug, so the ambiguous
    case fails closed.
    """
    declared = envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_DISTRIBUTED_MODE
    if declared:
        return declared
    try:
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            return "single" if distributed.get_world_size() == 1 else "unspecified"
    except Exception:  # noqa: BLE001 - absence of distributed means single process
        return "single"
    return "single"


def attach_graph_dispatch(modules: dict[str, Any] | None) -> GraphDispatchSession | None:
    """Attach generic dispatch, or return ``None`` when it is not configured.

    Returning ``None`` is the zero-effect path: no forward is wrapped, no graph
    is traced, and no artifact code is read. This is what makes an unset
    ``FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR`` behaviorally identical to a build
    without this feature.
    """
    configured = envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR
    if not configured:
        return None
    timing.reset()
    try:
        root = Path(configured).expanduser()
        enabled_ids = tuple(
            part.strip()
            for part in envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_ENABLE.split(",")
            if part.strip()
        )
        registry = ArtifactRegistry(root, enabled_ids=enabled_ids or None)
        if enabled_ids:
            logger.info(
                "Artifact selection restricted to %s (%d of %d bundles admitted)",
                ", ".join(enabled_ids),
                len(registry.manifests),
                len(registry.manifests) + len(registry.excluded_ids),
            )
        for error in registry.errors:
            logger.warning("Skipping artifact bundle: %s", error)
        if not registry.enabled:
            logger.info("No usable artifacts in %s; running natively", root)
            return None
        runtime = RuntimeProfile.detect(
            model_id=(envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID or envs.FASTVIDEO_OPTIMIZATION_PROFILE_MODEL_ID),
            model_revision=envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_REVISION,
            execution_mode=_execution_mode(modules),
            distributed_mode=_distributed_mode(),
        )
        session = GraphDispatchSession(
            registry,
            runtime,
            tracer=envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_TRACER,
            max_scopes=envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SCOPES,
            max_shape_variants=envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_MAX_SHAPES,
            validation=envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_VALIDATION,
        )
        wrapped = session.attach_modules(modules)
    except Exception:  # noqa: BLE001 - dispatch setup never breaks generation
        logger.warning("Graph dispatch could not be started; running natively", exc_info=True)
        return None
    if not wrapped:
        logger.info("Graph dispatch found no repeated module stacks; running natively")
        return None
    logger.info("Graph dispatch attached to %d module(s) from %s", wrapped, registry.root)
    return session


def detach_graph_dispatch(session: GraphDispatchSession | None) -> None:
    """Restore every native forward and write diagnostics if requested."""
    if session is None:
        return
    try:
        session.detach()
    except Exception:  # noqa: BLE001
        logger.warning("Graph dispatch detach failed", exc_info=True)
    output = envs.FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS
    if output:
        diagnostics_path = Path(str(output))
        session.write_diagnostics(diagnostics_path)
        try:
            timing.write_report(diagnostics_path.with_name("timing.json"))
        except Exception:  # noqa: BLE001 - diagnostics must never break teardown
            logger.warning("Could not resolve timing diagnostics path", exc_info=True)
