"""Server-side state: bundle, vote store and issued (blinded) ballots."""
from __future__ import annotations

import random
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .bundle import Bundle
from .pairing import Matchup, PastMatchup, choose_matchup
from .stats import summarize
from .votes import Vote, VoteError, VoteStore, clean_vote_fields, clean_voter, utc_now

MAX_BALLOTS = 5000
PENDING_TTL_SECONDS = 15 * 60
SIDES = ("left", "right")


class BallotError(LookupError):
    """Raised for unknown or already-used ballots."""


@dataclass(frozen=True)
class Ballot:
    ballot_id: str
    matchup: Matchup
    voter: str
    issued_at: float
    voted: bool = False


class AppState:

    def __init__(self, bundle: Bundle, votes: VoteStore, baseline: str | None = None,
                 rng: random.Random | None = None) -> None:
        self.bundle = bundle
        self.votes = votes
        self.baseline = baseline
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._ballots: OrderedDict[str, Ballot] = OrderedDict()
        self._availability = {clip.clip_id: frozenset(clip.videos) for clip in bundle.clips}
        self._groups = {clip.clip_id: clip.group for clip in bundle.clips if clip.group}

    def _history(self, now: float) -> list[PastMatchup]:
        history = [PastMatchup(v.clip_id, v.left_arm, v.right_arm, v.voter) for v in self.votes.all()]
        for ballot in self._ballots.values():
            if not ballot.voted and now - ballot.issued_at < PENDING_TTL_SECONDS:
                m = ballot.matchup
                history.append(PastMatchup(m.clip_id, m.left, m.right, ballot.voter))
        return history

    def new_ballot(self, voter: Any) -> dict[str, Any]:
        voter_name = clean_voter(voter)
        now = time.time()
        with self._lock:
            matchup = choose_matchup(self._availability, self.bundle.arm_slugs, self._history(now), voter_name,
                                     self._rng, self._groups, self.bundle.side_policy)
            ballot = Ballot(secrets.token_urlsafe(12), matchup, voter_name, now)
            self._ballots[ballot.ballot_id] = ballot
            while len(self._ballots) > MAX_BALLOTS:
                self._ballots.popitem(last=False)
        clip = self.bundle.clip(matchup.clip_id)
        return {
            "ballot_id": ballot.ballot_id,
            "clip_id": clip.clip_id,
            "prompt": clip.prompt,
            "guidance": clip.guidance,
            "left_url": f"/video/{ballot.ballot_id}/left",
            "right_url": f"/video/{ballot.ballot_id}/right",
            "voter_votes": sum(1 for v in self.votes.all() if v.voter == voter_name),
        }

    def video_path(self, ballot_id: str, side: str) -> Path:
        if side not in SIDES:
            raise BallotError(f"unknown side {side!r}")
        with self._lock:
            ballot = self._ballots.get(ballot_id)
        if ballot is None:
            raise BallotError("unknown ballot")
        arm = ballot.matchup.left if side == "left" else ballot.matchup.right
        return self.bundle.clip(ballot.matchup.clip_id).videos[arm]

    def record_vote(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        fields = clean_vote_fields(payload)
        ballot_id = str(payload.get("ballot_id") or "")
        with self._lock:
            ballot = self._ballots.get(ballot_id)
            if ballot is None:
                raise BallotError("unknown or expired ballot; load a new one")
            if ballot.voted:
                raise BallotError("this ballot was already voted on")
            self._ballots[ballot_id] = replace(ballot, voted=True)
        m = ballot.matchup
        vote = Vote(timestamp=utc_now(), clip_id=m.clip_id, left_arm=m.left, right_arm=m.right, ballot_id=ballot_id,
                    **fields)
        try:
            self.votes.append(vote, self.bundle.clip(m.clip_id).meta)
        except OSError:
            with self._lock:
                self._ballots[ballot_id] = ballot
            raise
        return {
            "ok": True,
            "reveal": {
                side: {
                    "slug": slug,
                    "display_name": self.bundle.arm(slug).display_name
                }
                for side, slug in (("left", m.left), ("right", m.right))
            },
            "voter_votes": sum(1 for v in self.votes.all() if v.voter == fields["voter"]),
        }

    def results(self, baseline: str | None = None) -> dict[str, Any]:
        summary = summarize(self.bundle.arms, self.votes.all(), baseline or self.baseline)
        return {**summary, "bundle": self.bundle.root.name, "clips": len(self.bundle.clips)}

    def info(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.root.name,
            "layout": self.bundle.layout,
            "arms": len(self.bundle.arms),
            "clips": len(self.bundle.clips),
            "votes": len(self.votes.all()),
            "site": dict(self.bundle.site),
        }


__all__ = ["AppState", "BallotError", "VoteError"]
