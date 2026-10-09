"""Append-only JSONL vote storage."""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CHOICES = ("left", "right", "tie", "both_bad")
MAX_VOTER_LEN = 64
MAX_COMMENT_LEN = 1000
MAX_SECONDS = 24 * 3600.0


class VoteError(ValueError):
    """Raised for invalid vote payloads."""


@dataclass(frozen=True)
class Vote:
    voter: str
    timestamp: str
    clip_id: str
    left_arm: str
    right_arm: str
    choice: str
    comment: str = ""
    watch_seconds: float = 0.0
    played_seconds: float = 0.0
    ballot_id: str = ""

    @property
    def winner(self) -> str | None:
        if self.choice == "left":
            return self.left_arm
        if self.choice == "right":
            return self.right_arm
        return None

    def to_json(self, extra: Mapping[str, Any] | None = None) -> str:
        """One JSONL line; ``extra`` (e.g. the clip's problem and seed) is added without overriding vote fields."""
        return json.dumps({**(extra or {}), **asdict(self)}, ensure_ascii=False, sort_keys=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_voter(raw: Any) -> str:
    voter = str(raw or "").strip()
    if not voter:
        raise VoteError("voter name is required")
    if len(voter) > MAX_VOTER_LEN:
        raise VoteError(f"voter name longer than {MAX_VOTER_LEN} characters")
    return voter


def _clean_seconds(raw: Any, name: str) -> float:
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError) as exc:
        raise VoteError(f"{name} must be a number") from exc
    if value != value or value < 0:  # NaN or negative
        raise VoteError(f"{name} must be a non-negative number")
    return round(min(value, MAX_SECONDS), 2)


def clean_vote_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the client-supplied part of a vote."""
    choice = payload.get("choice")
    if choice not in CHOICES:
        raise VoteError(f"choice must be one of {CHOICES}")
    comment = str(payload.get("comment") or "").strip()
    if len(comment) > MAX_COMMENT_LEN:
        raise VoteError(f"comment longer than {MAX_COMMENT_LEN} characters")
    return {
        "voter": clean_voter(payload.get("voter")),
        "choice": choice,
        "comment": comment,
        "watch_seconds": _clean_seconds(payload.get("watch_seconds"), "watch_seconds"),
        "played_seconds": _clean_seconds(payload.get("played_seconds"), "played_seconds"),
    }


def vote_from_record(record: Mapping[str, Any]) -> Vote:
    """Build a Vote from a stored JSON record (unknown keys are ignored)."""
    try:
        vote = Vote(
            voter=str(record["voter"]),
            timestamp=str(record["timestamp"]),
            clip_id=str(record["clip_id"]),
            left_arm=str(record["left_arm"]),
            right_arm=str(record["right_arm"]),
            choice=str(record["choice"]),
            comment=str(record.get("comment") or ""),
            watch_seconds=float(record.get("watch_seconds") or 0.0),
            played_seconds=float(record.get("played_seconds") or 0.0),
            ballot_id=str(record.get("ballot_id") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VoteError(f"malformed vote record: {exc}") from exc
    if vote.choice not in CHOICES:
        raise VoteError(f"unknown choice {vote.choice!r}")
    return vote


class VoteStore:
    """Thread-safe append-only vote log backed by a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.Lock()
        self._votes: list[Vote] = list(self._read_existing())

    def _read_existing(self) -> list[Vote]:
        if not self.path.is_file():
            return []
        votes, bad = [], 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    votes.append(vote_from_record(json.loads(line)))
                except (json.JSONDecodeError, VoteError):
                    bad += 1
        if bad:
            logger.warning("skipped %d malformed lines in %s", bad, self.path)
        return votes

    def append(self, vote: Vote, extra: Mapping[str, Any] | None = None) -> None:
        line = vote.to_json(extra) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._votes = [*self._votes, vote]

    def all(self) -> tuple[Vote, ...]:
        with self._lock:
            return tuple(self._votes)
