"""Balanced matchup scheduling for blind pairwise voting.

Each round picks the arm pair with the fewest comparisons so far (all pairs
get covered evenly over time), skipping pairs the current voter has already
seen on every clip while other pairs remain, then the clip that pair has been
compared on least, preferring clips the current voter has not yet seen for
that pair.
Left/right placement goes to whichever arm has been shown on the left less
often (random on ties), so no arm is systematically favored by screen side.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Matchup:
    clip_id: str
    left: str
    right: str


@dataclass(frozen=True)
class PastMatchup:
    clip_id: str
    arm_a: str
    arm_b: str
    voter: str = ""


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def available_pairs(availability: Mapping[str, frozenset[str]],
                    arm_slugs: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """All unordered arm pairs that share at least one clip."""
    pairs = []
    for a, b in combinations(sorted(set(arm_slugs)), 2):
        if any(a in arms and b in arms for arms in availability.values()):
            pairs.append((a, b))
    return tuple(pairs)


def _argmin_choice(items: list, key, rng: random.Random):
    best = min(key(item) for item in items)
    return rng.choice([item for item in items if key(item) == best])


def choose_matchup(availability: Mapping[str, frozenset[str]],
                   arm_slugs: Iterable[str],
                   history: Iterable[PastMatchup],
                   voter: str = "",
                   rng: random.Random | None = None) -> Matchup:
    """Pick the next blinded matchup.

    ``availability`` maps clip_id to the set of arms that have a video for it.
    ``history`` holds past (and optionally pending) comparisons.
    """
    rng = rng or random.Random()
    pairs = available_pairs(availability, arm_slugs)
    if not pairs:
        raise ValueError("no arm pair shares a clip; need at least two arms per clip")

    pair_counts: Counter[tuple[str, str]] = Counter()
    clip_pair_counts: Counter[tuple[str, tuple[str, str]]] = Counter()
    clip_counts: Counter[str] = Counter()
    left_counts: Counter[str] = Counter()
    appearances: Counter[str] = Counter()
    seen_by_voter: set[tuple[str, tuple[str, str]]] = set()
    for past in history:
        key = pair_key(past.arm_a, past.arm_b)
        pair_counts[key] += 1
        clip_pair_counts[(past.clip_id, key)] += 1
        clip_counts[past.clip_id] += 1
        left_counts[past.arm_a] += 1  # arm_a is the left side
        appearances.update((past.arm_a, past.arm_b))
        if voter and past.voter == voter:
            seen_by_voter.add((past.clip_id, key))

    clips_by_pair = {
        p: sorted(cid for cid, arms in availability.items() if p[0] in arms and p[1] in arms)
        for p in pairs
    }

    def exhausted(p: tuple[str, str]) -> bool:
        # The voter has already seen this pair on every clip it shares.
        return all((cid, p) in seen_by_voter for cid in clips_by_pair[p])

    pair = _argmin_choice(list(pairs), lambda p: (exhausted(p), pair_counts[p]), rng)
    clips = clips_by_pair[pair]
    clip_id = _argmin_choice(
        clips,
        lambda cid: ((cid, pair) in seen_by_voter, clip_pair_counts[(cid, pair)], clip_counts[cid]),
        rng,
    )
    a, b = pair
    # Left-side surplus = left placements minus half of all appearances; the
    # arm with the smaller surplus goes on the left.
    lean = (left_counts[a] - appearances[a] / 2) - (left_counts[b] - appearances[b] / 2)
    if lean == 0:
        lean = rng.random() - 0.5
    left, right = (b, a) if lean > 0 else (a, b)
    return Matchup(clip_id=clip_id, left=left, right=right)
