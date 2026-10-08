"""Bundle parsing for the blind A/B app.

Two layouts are supported:

* **Manifest layout**: ``manifest.jsonl`` (one row per prompt with ``index``,
  ``sample_id``, ``prompt`` and ``arms: {slug: {path, ...}}``), ``arms.json``
  and ``arms/<arm>/videos/<index>_<sample_id>.mp4``.
* **Simple layout**: ``arms.json`` plus ``arms/<arm>/<clip_id>.mp4``. Files with
  the same ``<clip_id>`` across arms are the same prompt + seed. Optional
  ``prompts.json`` maps ``clip_id -> prompt text``.

Both layouts share the ``arms.json`` schema: ``{"arms": [{"slug",
"display_name", "notes", "speed": {...}, ...}]}``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
ARMS_NAME = "arms.json"
PROMPTS_NAME = "prompts.json"
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".m4v")


class BundleError(ValueError):
    """Raised when a bundle directory cannot be parsed."""


@dataclass(frozen=True)
class Arm:
    slug: str
    display_name: str
    notes: str = ""
    speed: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def seconds_per_clip(self) -> float | None:
        value = self.speed.get("seconds_per_clip")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        return None


@dataclass(frozen=True)
class Clip:
    clip_id: str
    prompt: str
    videos: Mapping[str, Path]  # arm slug -> absolute video path

    def has_arms(self, a: str, b: str) -> bool:
        return a in self.videos and b in self.videos


@dataclass(frozen=True)
class Bundle:
    root: Path
    layout: str  # "manifest" or "simple"
    arms: tuple[Arm, ...]
    clips: tuple[Clip, ...]

    @property
    def arm_slugs(self) -> tuple[str, ...]:
        return tuple(arm.slug for arm in self.arms)

    def arm(self, slug: str) -> Arm:
        for arm in self.arms:
            if arm.slug == slug:
                return arm
        raise KeyError(slug)

    def clip(self, clip_id: str) -> Clip:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(clip_id)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {path}: {exc}") from exc


def _parse_arm(raw: Any, source: Path) -> Arm:
    if not isinstance(raw, dict) or not isinstance(raw.get("slug"), str) or not raw["slug"]:
        raise BundleError(f"{source}: every arm needs a non-empty string 'slug', got {raw!r}")
    slug = raw["slug"]
    if "/" in slug or "\\" in slug or slug in (".", ".."):
        raise BundleError(f"{source}: invalid arm slug {slug!r}")
    speed = raw.get("speed") or {}
    if not isinstance(speed, dict):
        raise BundleError(f"{source}: arm {slug!r} 'speed' must be an object")
    return Arm(
        slug=slug,
        display_name=str(raw.get("display_name") or slug),
        notes=str(raw.get("notes") or ""),
        speed=MappingProxyType(dict(speed)),
    )


def load_arms(root: Path) -> tuple[Arm, ...]:
    path = root / ARMS_NAME
    if not path.is_file():
        raise BundleError(f"missing {ARMS_NAME} in {root}")
    data = _read_json(path)
    raw_arms = data.get("arms") if isinstance(data, dict) else None
    if not isinstance(raw_arms, list) or not raw_arms:
        raise BundleError(f"{path}: expected a non-empty 'arms' list")
    arms = tuple(_parse_arm(raw, path) for raw in raw_arms)
    slugs = [arm.slug for arm in arms]
    if len(set(slugs)) != len(slugs):
        raise BundleError(f"{path}: duplicate arm slugs in {slugs}")
    return arms


def _manifest_clip_id(row: Mapping[str, Any]) -> str:
    index, sample_id = row.get("index"), row.get("sample_id")
    if isinstance(index, int) and sample_id:
        return f"{index:03d}_{sample_id}"
    if sample_id:
        return str(sample_id)
    if isinstance(index, int):
        return f"{index:03d}"
    raise BundleError(f"manifest row has neither 'index' nor 'sample_id': {sorted(row)}")


def _manifest_video(root: Path, slug: str, clip_id: str, entry: Any) -> Path | None:
    candidates = []
    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
        candidates.append(root / entry["path"])
    elif isinstance(entry, str):
        candidates.append(root / entry)
    candidates.append(root / "arms" / slug / "videos" / f"{clip_id}.mp4")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_manifest_clips(root: Path, arms: tuple[Arm, ...]) -> tuple[Clip, ...]:
    slugs = {arm.slug for arm in arms}
    clips = []
    path = root / MANIFEST_NAME
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise BundleError(f"{path}:{line_no}: expected an object")
            clip_id = _manifest_clip_id(row)
            arm_entries = row.get("arms") if isinstance(row.get("arms"), dict) else {}
            videos = {}
            for slug in sorted(slugs):
                video = _manifest_video(root, slug, clip_id, arm_entries.get(slug))
                if video is None:
                    logger.warning("missing video for arm %s clip %s", slug, clip_id)
                    continue
                videos[slug] = video
            clips.append(Clip(clip_id, str(row.get("prompt") or ""), MappingProxyType(videos)))
    return tuple(clips)


def _load_simple_clips(root: Path, arms: tuple[Arm, ...]) -> tuple[Clip, ...]:
    prompts_path = root / PROMPTS_NAME
    prompts = _read_json(prompts_path) if prompts_path.is_file() else {}
    if not isinstance(prompts, dict):
        raise BundleError(f"{prompts_path}: expected an object mapping clip_id -> prompt")
    by_clip: dict[str, dict[str, Path]] = {}
    for arm in arms:
        arm_dir = root / "arms" / arm.slug
        if not arm_dir.is_dir():
            logger.warning("arm directory missing: %s", arm_dir)
            continue
        for video in sorted(arm_dir.iterdir()):
            if video.suffix.lower() in VIDEO_SUFFIXES and video.is_file():
                by_clip.setdefault(video.stem, {})[arm.slug] = video.resolve()
    return tuple(
        Clip(clip_id, str(prompts.get(clip_id, "")), MappingProxyType(videos))
        for clip_id, videos in sorted(by_clip.items()))


def load_bundle(root: str | Path) -> Bundle:
    """Parse a bundle directory in either supported layout."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"bundle directory not found: {root}")
    arms = load_arms(root)
    if (root / MANIFEST_NAME).is_file():
        layout, clips = "manifest", _load_manifest_clips(root, arms)
    else:
        layout, clips = "simple", _load_simple_clips(root, arms)
    usable = tuple(clip for clip in clips if len(clip.videos) >= 2)
    if not usable:
        raise BundleError(f"{root}: no clip has videos for at least two arms")
    if len(usable) < len(clips):
        logger.warning("skipping %d clips with fewer than two arms", len(clips) - len(usable))
    return Bundle(root=root, layout=layout, arms=arms, clips=usable)
