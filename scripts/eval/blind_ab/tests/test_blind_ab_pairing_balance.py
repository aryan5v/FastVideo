"""Pairing scheduler: every pair covered evenly, clips spread, sides random. CPU only."""
import random
from collections import Counter
from itertools import combinations

import pytest

from blind_ab.pairing import Matchup, PastMatchup, available_pairs, choose_matchup, pair_key

ARMS = ("a", "b", "c", "d")
CLIPS = {f"clip{i}": frozenset(ARMS) for i in range(5)}


def _run(rounds, availability=CLIPS, arms=ARMS, voter="v", seed=0):
    rng = random.Random(seed)
    history = []
    for _ in range(rounds):
        m = choose_matchup(availability, arms, history, voter, rng)
        history.append(PastMatchup(m.clip_id, m.left, m.right, voter))
    return history


def test_pairs_are_balanced_after_each_full_cycle():
    n_pairs = len(list(combinations(ARMS, 2)))
    history = _run(n_pairs * 7)
    counts = Counter(pair_key(h.arm_a, h.arm_b) for h in history)
    assert set(counts) == set(combinations(ARMS, 2))
    assert set(counts.values()) == {7}


def test_pair_counts_never_differ_by_more_than_one():
    history = []
    rng = random.Random(1)
    for _ in range(50):
        m = choose_matchup(CLIPS, ARMS, history, "v", rng)
        history.append(PastMatchup(m.clip_id, m.left, m.right, "v"))
        counts = Counter(pair_key(h.arm_a, h.arm_b) for h in history)
        values = [counts.get(p, 0) for p in combinations(ARMS, 2)]
        assert max(values) - min(values) <= 1


def test_clips_spread_within_a_pair_and_voter_avoids_repeats():
    history = _run(6 * len(CLIPS))
    per_pair_clip = Counter((pair_key(h.arm_a, h.arm_b), h.clip_id) for h in history)
    # 6 pairs x 5 clips, 30 rounds: each (pair, clip) shown exactly once.
    assert set(per_pair_clip.values()) == {1}


def test_left_right_placement_is_balanced_per_arm():
    history = _run(400)
    left_counts = Counter(h.arm_a for h in history)
    right_counts = Counter(h.arm_b for h in history)
    for arm in ARMS:
        assert abs(left_counts[arm] - right_counts[arm]) <= 2


def test_left_right_order_is_not_deterministic():
    orders = {(m.left, m.right) for m in (choose_matchup({"c": frozenset("ab")}, "ab", [], "v", random.Random(s))
                                          for s in range(20))}
    assert orders == {("a", "b"), ("b", "a")}


def test_respects_missing_videos():
    availability = {"x": frozenset({"a", "b"}), "y": frozenset({"b", "c"})}
    assert available_pairs(availability, ["a", "b", "c"]) == (("a", "b"), ("b", "c"))
    for h in _run(20, availability, ("a", "b", "c")):
        assert {h.arm_a, h.arm_b} <= availability[h.clip_id]


def test_no_shared_clip_raises():
    with pytest.raises(ValueError):
        choose_matchup({"x": frozenset({"a"})}, ["a", "b"], [], "v", random.Random(0))


def test_returns_matchup_with_distinct_arms():
    m = choose_matchup(CLIPS, ARMS, [], "v", random.Random(3))
    assert isinstance(m, Matchup) and m.left != m.right and m.clip_id in CLIPS


def test_voter_never_repeats_pair_and_clip_while_unseen_combos_remain():
    # One voter, 6 pairs x 2 clips = 12 combos; another voter's votes skew the global pair counts.
    clips = {f"clip{i}": frozenset(ARMS) for i in range(2)}
    rng = random.Random(5)
    history = [PastMatchup("clip0", "a", "b", "other")] * 3
    seen = set()
    for _ in range(12):
        m = choose_matchup(clips, ARMS, history, "me", rng)
        combo = (m.clip_id, pair_key(m.left, m.right))
        assert combo not in seen
        seen.add(combo)
        history.append(PastMatchup(m.clip_id, m.left, m.right, "me"))
    assert len(seen) == 12
