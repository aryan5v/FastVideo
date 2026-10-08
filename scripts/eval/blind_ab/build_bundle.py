#!/usr/bin/env python3
"""Assemble a simple-layout blind A/B bundle from per-arm folders of videos.

Videos are matched across arms by file stem (``<clip_id>.mp4``), so each arm
folder should contain one file per prompt+seed with identical names.

Example:
    python scripts/eval/blind_ab/build_bundle.py --out /tmp/my_bundle \\
        --arm baseline=runs/baseline_videos --arm fp8_decoder=runs/fp8_videos \\
        --speed speed.json --prompts prompts.json

``speed.json`` maps arm slug to metadata, e.g.::

    {"baseline": {"display_name": "Release default", "notes": "50 steps",
                  "seconds_per_clip": 70.9, "hardware": "1x GPU", "resolution": "832x480"}}

``display_name`` and ``notes`` become arm fields; everything else goes under
the arm's ``speed`` object. ``prompts.json`` (optional) maps clip_id -> prompt.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger("build_bundle")

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".m4v")
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ARM_FIELDS = ("display_name", "notes")
MODES = ("symlink", "copy", "hardlink")


def parse_arm_spec(spec: str) -> tuple[str, Path]:
    slug, sep, folder = spec.partition("=")
    if not sep or not folder:
        raise ValueError(f"--arm expects SLUG=DIR, got {spec!r}")
    if not SLUG_RE.match(slug) or slug in (".", ".."):
        raise ValueError(f"invalid arm slug {slug!r} (letters, digits, '.', '_', '-')")
    return slug, Path(folder).expanduser()


def list_videos(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise ValueError(f"arm folder not found: {folder}")
    return {
        p.stem: p.resolve()
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    }


def select_clips(per_arm: Mapping[str, Mapping[str, Path]], require_all: bool) -> list[str]:
    """Clip ids present in every arm (``require_all``) or in at least two arms."""
    counts: dict[str, int] = {}
    for videos in per_arm.values():
        for clip_id in videos:
            counts[clip_id] = counts.get(clip_id, 0) + 1
    need = len(per_arm) if require_all else 2
    return sorted(clip_id for clip_id, n in counts.items() if n >= need)


def arm_entry(slug: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ValueError(f"speed entry for {slug!r} must be an object")
    entry: dict[str, Any] = {"slug": slug, "display_name": str(meta.get("display_name") or slug)}
    if meta.get("notes"):
        entry["notes"] = str(meta["notes"])
    speed = {k: v for k, v in meta.items() if k not in ARM_FIELDS}
    spc = speed.get("seconds_per_clip")
    if spc is not None and (isinstance(spc, bool) or not isinstance(spc, (int, float)) or spc <= 0):
        raise ValueError(f"{slug}: seconds_per_clip must be a positive number, got {spc!r}")
    if speed:
        entry["speed"] = speed
    return entry


def place_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def build_bundle(out: Path,
                 arms: Sequence[tuple[str, Path]],
                 speed: Mapping[str, Any] | None = None,
                 prompts: Mapping[str, str] | None = None,
                 mode: str = "symlink",
                 require_all: bool = False) -> dict[str, Any]:
    """Create ``out`` with ``arms.json``, ``arms/<slug>/<clip_id>.<ext>`` and ``prompts.json``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    slugs = [slug for slug, _ in arms]
    if len(arms) < 2 or len(set(slugs)) != len(slugs):
        raise ValueError(f"need at least two arms with unique slugs, got {slugs}")
    speed = dict(speed or {})
    unknown = sorted(set(speed) - set(slugs))
    if unknown:
        logger.warning("speed entries for unknown arms ignored: %s", unknown)
    per_arm = {slug: list_videos(folder) for slug, folder in arms}
    clip_ids = select_clips(per_arm, require_all)
    if not clip_ids:
        raise ValueError("no clip id is shared by enough arms; check that file names match across folders")
    for slug, videos in per_arm.items():
        dropped = sorted(set(videos) - set(clip_ids))
        if dropped:
            logger.warning("%s: %d unmatched files skipped (e.g. %s)", slug, len(dropped), dropped[0])
    out.mkdir(parents=True, exist_ok=True)
    for slug, videos in per_arm.items():
        arm_dir = out / "arms" / slug
        arm_dir.mkdir(parents=True, exist_ok=True)
        for clip_id in clip_ids:
            if clip_id in videos:
                src = videos[clip_id]
                place_file(src, arm_dir / f"{clip_id}{src.suffix.lower()}", mode)
    arms_doc = {
        "schema_version": "blind-ab-simple-v1",
        "arms": [arm_entry(slug, speed.get(slug, {})) for slug in slugs],
    }
    (out / "arms.json").write_text(json.dumps(arms_doc, indent=2) + "\n", encoding="utf-8")
    if prompts:
        kept = {cid: str(prompts[cid]) for cid in clip_ids if cid in prompts}
        (out / "prompts.json").write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"clips": len(clip_ids), "arms": slugs}


def _load_json_object(path: str | None, what: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{what} file must contain a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output bundle directory")
    parser.add_argument("--arm", action="append", required=True, metavar="SLUG=DIR", help="arm folder (repeat)")
    parser.add_argument("--speed", help="JSON: {slug: {seconds_per_clip, hardware, resolution, display_name, ...}}")
    parser.add_argument("--prompts", help="JSON: {clip_id: prompt text}")
    parser.add_argument("--mode", choices=MODES, default="symlink", help="how to place videos (default: symlink)")
    parser.add_argument("--require-all", action="store_true", help="keep only clips present in every arm")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        arms = [parse_arm_spec(spec) for spec in args.arm]
        result = build_bundle(Path(args.out).expanduser(), arms, _load_json_object(args.speed, "speed"),
                              _load_json_object(args.prompts, "prompts"), args.mode, args.require_all)
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("wrote %s: %d clips x %d arms", args.out, result["clips"], len(result["arms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
