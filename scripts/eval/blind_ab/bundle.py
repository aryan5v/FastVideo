"""Bundle parsing for the blind A/B app.

Two layouts are supported:

* **Manifest layout**: ``manifest.jsonl`` (one row per prompt with ``index``,
  ``sample_id``, ``prompt`` and ``arms: {slug: {path, ...}}``), ``arms.json``
  and ``arms/<arm>/videos/<index>_<sample_id>.mp4``.
* **Simple layout**: ``arms.json`` plus ``arms/<arm>/<clip_id>.mp4``. Files with
  the same ``<clip_id>`` across arms are the same prompt + seed. Optional
  ``prompts.json`` maps ``clip_id`` to the prompt text, or to an object
  ``{"prompt", "guidance", "group", "meta"}``: ``guidance`` is a short "what to
  look for" note shown with the clip, ``group`` ties clips that share a source
  (e.g. one problem rendered with several seeds) so pairing spreads over groups,
  and ``meta`` (flat JSON object) is stored with every vote on that clip.

An optional ``site.json`` customizes the page text (title, intro paragraphs,
choice labels) and the side policy; see ``load_site``.

Both layouts share the ``arms.json`` schema: ``{"arms": [{"slug",
"display_name", "notes", "speed": {...}, ...}]}``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
ARMS_NAME = "arms.json"
PROMPTS_NAME = "prompts.json"
SITE_NAME = "site.json"
SIDE_POLICIES = ("balanced", "random")
SITE_TEXT_KEYS = ("title", "heading", "reveal_label", "guidance_label")
CHOICE_KEYS = ("left", "right", "tie", "both_bad")
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
    guidance: str = ""
    group: str = ""
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def has_arms(self, a: str, b: str) -> bool:
        return a in self.videos and b in self.videos


@dataclass(frozen=True)
class Bundle:
    root: Path
    layout: str  # "manifest" or "simple"
    arms: tuple[Arm, ...]
    clips: tuple[Clip, ...]
    site: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def side_policy(self) -> str:
        return str(self.site.get("sides") or "balanced")

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


def parse_prompt_entry(raw: Any, clip_id: str, source: Path) -> dict[str, Any]:
    """Normalize a prompts.json value (a string, or an object) into Clip keyword fields."""
    if raw is None:
        return {"prompt": ""}
    if isinstance(raw, str):
        return {"prompt": raw}
    if not isinstance(raw, dict):
        raise BundleError(f"{source}: entry for {clip_id!r} must be a string or an object")
    meta = raw.get("meta") or {}
    if not isinstance(meta, dict) or any(isinstance(v, (dict, list)) for v in meta.values()):
        raise BundleError(f"{source}: 'meta' for {clip_id!r} must be a flat object")
    return {
        "prompt": str(raw.get("prompt") or ""),
        "guidance": str(raw.get("guidance") or ""),
        "group": str(raw.get("group") or ""),
        "meta": MappingProxyType(dict(meta)),
    }


def load_site(root: Path) -> Mapping[str, Any]:
    """Read the optional site.json: {"title", "heading", "intro": [paragraphs], "choices": {...},
    "reveal_label", "guidance_label", "sides": "balanced" | "random"}."""
    path = root / SITE_NAME
    if not path.is_file():
        return MappingProxyType({})
    data = _read_json(path)
    if not isinstance(data, dict):
        raise BundleError(f"{path}: expected an object")
    if data.get("sides", "balanced") not in SIDE_POLICIES:
        raise BundleError(f"{path}: 'sides' must be one of {SIDE_POLICIES}")
    intro = data.get("intro", [])
    if not isinstance(intro, list) or not all(isinstance(p, str) for p in intro):
        raise BundleError(f"{path}: 'intro' must be a list of paragraph strings")
    choices = data.get("choices", {})
    if not isinstance(choices, dict) or not set(choices) <= set(CHOICE_KEYS):
        raise BundleError(f"{path}: 'choices' may only relabel {CHOICE_KEYS}")
    for key in SITE_TEXT_KEYS:
        if key in data and not isinstance(data[key], str):
            raise BundleError(f"{path}: {key!r} must be a string")
    return MappingProxyType(dict(data))


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
        Clip(clip_id=clip_id, videos=MappingProxyType(videos),
             **parse_prompt_entry(prompts.get(clip_id), clip_id, prompts_path))
        for clip_id, videos in sorted(by_clip.items()))


def select_arms(arms: tuple[Arm, ...], only: Sequence[str] | None) -> tuple[Arm, ...]:
    """Keep only the arms named in ``only`` (in arms.json order); ``None`` or empty keeps all."""
    if not only:
        return arms
    unknown = sorted(set(only) - {arm.slug for arm in arms})
    if unknown:
        raise BundleError(f"unknown arm(s) {unknown}; arms.json has {[arm.slug for arm in arms]}")
    kept = tuple(arm for arm in arms if arm.slug in set(only))
    if len(kept) < 2:
        raise BundleError(f"need at least two arms, got {[arm.slug for arm in kept]}")
    return kept


def load_bundle(root: str | Path, only_arms: Sequence[str] | None = None) -> Bundle:
    """Parse a bundle directory in either supported layout, optionally restricted to ``only_arms``."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"bundle directory not found: {root}")
    arms = select_arms(load_arms(root), only_arms)
    if (root / MANIFEST_NAME).is_file():
        layout, clips = "manifest", _load_manifest_clips(root, arms)
    else:
        layout, clips = "simple", _load_simple_clips(root, arms)
    usable = tuple(clip for clip in clips if len(clip.videos) >= 2)
    if not usable:
        raise BundleError(f"{root}: no clip has videos for at least two arms")
    if len(usable) < len(clips):
        logger.warning("skipping %d clips with fewer than two arms", len(clips) - len(usable))
    return Bundle(root=root, layout=layout, arms=arms, clips=usable, site=load_site(root))
