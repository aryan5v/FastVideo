from __future__ import annotations

from dataclasses import dataclass

from dreamverse.config import FRAME_HEIGHT, FRAME_WIDTH, GENERATION_SEGMENT_CAP, MODEL_REGISTRY, NUM_FRAMES

LOBBY_MODEL_IDS = frozenset({"fast-ltx2", "fast-ltx23"})
SUPPORTED_CREATION_MODES = frozenset({"t2v", "fl2av", "ref2av"})
SUPPORTED_ASPECT_RATIOS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p", "4k"})
SEGMENT_DURATION_SEC = 5


@dataclass(frozen=True)
class SessionCreationConfig:
    model_id: str
    creation_mode: str
    aspect_ratio: str
    resolution: str
    duration_sec: int
    frame_width: int
    frame_height: int
    num_frames: int
    generation_segment_cap: int

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "creation_mode": self.creation_mode,
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution,
            "duration_sec": self.duration_sec,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "num_frames": self.num_frames,
            "generation_segment_cap": self.generation_segment_cap,
        }


def _round_to_multiple(value: float, multiple: int = 32) -> int:
    rounded = int(round(value / multiple)) * multiple
    return max(multiple, rounded)


def _resolution_base(resolution: str) -> int:
    return {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "4k": 2160,
    }.get(resolution, 720)


def resolve_frame_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    if aspect_ratio == "16:9" and resolution == "1080p":
        return FRAME_WIDTH, FRAME_HEIGHT

    base = _resolution_base(resolution)
    width_ratio, height_ratio = {
        "21:9": (21, 9),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1),
        "3:4": (3, 4),
        "9:16": (9, 16),
    }.get(aspect_ratio, (16, 9))

    if width_ratio >= height_ratio:
        height = _round_to_multiple(base)
        width = _round_to_multiple(height * width_ratio / height_ratio)
    else:
        width = _round_to_multiple(base)
        height = _round_to_multiple(width * height_ratio / width_ratio)
    return width, height


def duration_sec_to_segment_cap(duration_sec: int, *, global_cap: int = GENERATION_SEGMENT_CAP) -> int:
    requested = max(1, int(round(duration_sec / SEGMENT_DURATION_SEC + 0.0001)))
    if global_cap <= 0:
        return requested
    return max(1, min(requested, global_cap))


def parse_session_creation_config(payload: dict[str, object]) -> SessionCreationConfig:
    raw_model_id = str(payload.get("model_id") or "").strip()
    model_id = raw_model_id if raw_model_id in LOBBY_MODEL_IDS else "fast-ltx23"

    creation_mode = str(payload.get("creation_mode") or "t2v").strip()
    if creation_mode not in SUPPORTED_CREATION_MODES:
        raise ValueError(f"Unsupported creation_mode: {creation_mode}")

    aspect_ratio = str(payload.get("aspect_ratio") or "16:9").strip()
    if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect_ratio: {aspect_ratio}")

    resolution = str(payload.get("resolution") or "720p").strip()
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}")

    try:
        duration_sec = int(payload.get("duration_sec") or SEGMENT_DURATION_SEC)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_sec must be an integer.") from exc
    if duration_sec not in {5, 10, 15}:
        raise ValueError("duration_sec must be 5, 10, or 15.")

    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_id: {model_id}")

    frame_width, frame_height = resolve_frame_size(aspect_ratio, resolution)
    return SessionCreationConfig(
        model_id=model_id,
        creation_mode=creation_mode,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration_sec=duration_sec,
        frame_width=frame_width,
        frame_height=frame_height,
        num_frames=NUM_FRAMES,
        generation_segment_cap=duration_sec_to_segment_cap(duration_sec),
    )


def validate_creation_mode_assets(
    creation_mode: str,
    *,
    has_initial_image: bool,
    has_last_frame_image: bool,
) -> None:
    if creation_mode == "ref2av" and not has_initial_image:
        raise ValueError("Omni reference mode requires a reference image.")
    if creation_mode == "fl2av" and (not has_initial_image or not has_last_frame_image):
        raise ValueError("First and last frame mode requires both frame images.")
