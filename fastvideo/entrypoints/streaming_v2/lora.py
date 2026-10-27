# SPDX-License-Identifier: Apache-2.0
"""Per-request LoRA selection for the streaming_v2 server.

The server owns exactly one generator, and FastVideo keeps the active adapter on the pipeline, so a request
applies its adapter under the same lock as its generation. ``GET /v1/streamv2/capabilities`` reports the
allow-list; a request carries ``{"lora": {"id": ..., "strength": ...}}``.

Two FastVideo properties matter here:

* ``strength=0`` is the base model. A side-branch layer returns its base layer's output directly, and a merged
  adapter contributes a zero delta, so a request that omits ``lora`` after another request loaded an adapter
  still renders the base model.
* Not every checkpoint can switch at runtime. Packed NVFP4/FP8 checkpoints keep the adapter beside the packed
  weight and switch freely; a dense checkpoint merges it, and a checkpoint whose BF16 weights NVFP4 removed
  rejects the change. ``set_lora_adapter`` raises in that case, and the error reaches the client as an
  ``invalid_request`` error event.

@module fastvideo.entrypoints.streaming_v2.lora
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from fastvideo.api.schema import LoRAAdapterConfig, StreamingV2LoRAConfig
from fastvideo.entrypoints.streaming_v2.messages import LoraSelection
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class LoraRequestError(ValueError):
    """A request's LoRA selection cannot be served; the message is the client-facing reason."""


def _adapter_capability(adapter: LoRAAdapterConfig) -> dict[str, Any]:
    """The capabilities entry of one adapter: what a client may select, never the local load path."""
    entry: dict[str, Any] = {
        "id": adapter.id,
        "default_strength": adapter.default_strength,
        "min_strength": adapter.min_strength,
        "max_strength": adapter.max_strength,
    }
    if adapter.description:
        entry["description"] = adapter.description
    if adapter.license:
        entry["license"] = adapter.license
    if adapter.source_url:
        entry["source_url"] = adapter.source_url
    return entry


def lora_capabilities(config: StreamingV2LoRAConfig | None) -> dict[str, Any]:
    """The ``lora`` block of ``GET /v1/streamv2/capabilities``.

    ``supported`` is false without an allow-list, and the server then refuses any request that selects an
    adapter. ``runtime_switching`` says that a request may change the adapter or the strength of a running
    server; on a checkpoint that merges its adapter, such a change raises and the request fails with
    ``invalid_request`` instead of restarting.
    """
    if config is None or not config.adapters:
        return {"supported": False, "runtime_switching": False, "adapters": []}
    return {
        "supported": True,
        "runtime_switching": True,
        "base_strength": 0.0,
        "adapters": [_adapter_capability(adapter) for adapter in config.adapters],
    }


@dataclass
class LoraRuntime:
    """The adapters a request may select, and the adapter the generator currently carries."""
    config: StreamingV2LoRAConfig
    # The adapter the pipeline currently carries; ``None`` until the first request selects one.
    current_id: str | None = None
    current_strength: float | None = None
    adapters: dict[str, LoRAAdapterConfig] = field(init=False)

    def __post_init__(self) -> None:
        self.adapters = {adapter.id: adapter for adapter in self.config.adapters}

    def validate(self, selection: LoraSelection | None) -> None:
        """Refuse a selection that names no allowed adapter or a strength outside its range.

        This runs before the response starts, so an invalid selection is an HTTP 400 rather than an error event
        after a generation has begun.
        """
        if selection is None:
            return
        adapter = self.adapters.get(selection.id)
        if adapter is None:
            known = ", ".join(sorted(self.adapters)) or "none"
            raise LoraRequestError(f"Unknown LoRA adapter {selection.id!r}; this server serves: {known}.")
        strength = adapter.default_strength if selection.strength is None else selection.strength
        if not math.isfinite(strength) or not adapter.min_strength <= strength <= adapter.max_strength:
            raise LoraRequestError(f"LoRA adapter {adapter.id!r} takes a strength from {adapter.min_strength:g} to "
                                   f"{adapter.max_strength:g}; got {strength:g}.")

    def apply(self, generator: Any, selection: LoraSelection | None) -> None:
        """Make the selection current on ``generator``; ``None`` renders the base model.

        Runs inside the generation lock, so two requests cannot interleave adapter changes. A no-op when the
        requested adapter and strength are already current.
        """
        if selection is None:
            if self.current_id is None or self.current_strength == 0.0:
                return
            self._set(generator, self.current_id, 0.0)
            return
        self.validate(selection)
        adapter = self.adapters[selection.id]
        strength = adapter.default_strength if selection.strength is None else selection.strength
        if self.current_id == adapter.id and self.current_strength == strength:
            return
        self._set(generator, adapter.id, strength)

    def _set(self, generator: Any, adapter_id: str, strength: float) -> None:
        adapter = self.adapters[adapter_id]
        logger.info("streaming_v2 applies LoRA adapter %r at strength %g", adapter_id, strength)
        try:
            generator.set_lora_adapter(adapter_id, adapter.path, strength=strength)
        except Exception as error:  # a bad adapter, an unreachable repo, or a checkpoint that cannot switch
            # The pipeline may have left a partial adapter behind, so the next request must set one unconditionally.
            self.current_id = None
            self.current_strength = None
            raise LoraRequestError(f"LoRA adapter {adapter_id!r} could not be applied: {error}") from error
        self.current_id = adapter_id
        self.current_strength = strength
