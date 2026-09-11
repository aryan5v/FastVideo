from __future__ import annotations

from dataclasses import dataclass

from dreamverse.config import MODEL_REGISTRY

# FastLTX lobby models share the same creation-studio surface today.
LTX_LOBBY_MODEL_IDS = frozenset({"fast-ltx2", "fast-ltx23"})

# Canonical upstream wire IDs. FL2VA is tracked in #1834 but not implemented on
# FastLTX streaming yet (last-frame conditioning is not wired).
LTX_LOBBY_GENERATION_MODES = frozenset({"t2va", "ref2va"})

LTX_LOBBY_ASPECT_RATIOS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})

# Realtime FastLTX serving is validated through 1080p-class outputs; 4K is rejected
# until the runtime path is tested on Dreamverse GPUs.
LTX_LOBBY_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})

LTX_LOBBY_DURATION_SEC = frozenset({5, 10, 15})

UNSUPPORTED_GENERATION_MODE_MESSAGES = {
    "fl2va": "First/last frame mode (FL2VA) is not supported on FastLTX models yet.",
}


@dataclass(frozen=True)
class LobbyCreationCapabilities:
    model_ids: frozenset[str]
    generation_modes: frozenset[str]
    aspect_ratios: frozenset[str]
    resolutions: frozenset[str]
    duration_sec: frozenset[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "model_ids": sorted(self.model_ids),
            "generation_modes": sorted(self.generation_modes),
            "aspect_ratios": sorted(self.aspect_ratios),
            "resolutions": sorted(self.resolutions),
            "duration_sec": sorted(self.duration_sec),
            "unsupported_generation_modes": dict(UNSUPPORTED_GENERATION_MODE_MESSAGES),
            "reference_assets": {
                "mime_types": ["image/png", "image/jpeg", "image/webp"],
                "max_bytes": 15 * 1024 * 1024,
            },
        }


LOBBY_CREATION_CAPABILITIES = LobbyCreationCapabilities(
    model_ids=LTX_LOBBY_MODEL_IDS,
    generation_modes=LTX_LOBBY_GENERATION_MODES,
    aspect_ratios=LTX_LOBBY_ASPECT_RATIOS,
    resolutions=LTX_LOBBY_RESOLUTIONS,
    duration_sec=LTX_LOBBY_DURATION_SEC,
)


def validate_lobby_creation_config(
    *,
    model_id: str,
    generation_mode: str,
    aspect_ratio: str,
    resolution: str,
    duration_sec: int,
) -> None:
    if model_id not in LOBBY_CREATION_CAPABILITIES.model_ids:
        raise ValueError(f"Unsupported model_id: {model_id}")
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}")

    if generation_mode in UNSUPPORTED_GENERATION_MODE_MESSAGES:
        raise ValueError(UNSUPPORTED_GENERATION_MODE_MESSAGES[generation_mode])
    if generation_mode not in LOBBY_CREATION_CAPABILITIES.generation_modes:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    if aspect_ratio not in LOBBY_CREATION_CAPABILITIES.aspect_ratios:
        raise ValueError(f"Unsupported aspect_ratio: {aspect_ratio}")
    if resolution not in LOBBY_CREATION_CAPABILITIES.resolutions:
        raise ValueError(f"Unsupported resolution: {resolution}")
    if duration_sec not in LOBBY_CREATION_CAPABILITIES.duration_sec:
        raise ValueError("duration_sec must be 5, 10, or 15.")
