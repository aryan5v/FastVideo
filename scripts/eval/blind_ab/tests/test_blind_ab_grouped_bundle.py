"""Grouped bundles (problem x seed clips with per-problem guidance), random sides, arm selection. CPU only."""
import json
import random
from collections import Counter

import pytest

from blind_ab.app_state import AppState
from blind_ab.bundle import BundleError, load_bundle
from blind_ab.pairing import PastMatchup, choose_matchup
from blind_ab.votes import VoteStore

from test_blind_ab_bundle_parsing import _write

ARMS = ("v2", "trim", "omni")
PROBLEMS = [f"P{n}" for n in range(1, 41)]
SEEDS = (42, 43, 44, 45)
GROUPS = {f"{p}_seed{s}": p for p in PROBLEMS for s in SEEDS}


def make_grouped_bundle(root, problems=("P1", "P2"), site=None):
    _write(root / "arms.json", json.dumps({"arms": [{"slug": a} for a in ARMS]}))
    prompts = {}
    for p in problems:
        for s in SEEDS:
            clip_id = f"{p}_seed{s}"
            for arm in ARMS:
                _write(root / "arms" / arm / f"{clip_id}.mp4")
            prompts[clip_id] = {"prompt": f"prompt {p}", "guidance": f"look at {p}", "group": p,
                                "meta": {"problem": p, "seed": s}}
    _write(root / "prompts.json", json.dumps(prompts))
    if site is not None:
        _write(root / "site.json", json.dumps(site))
    return root


def test_prompt_objects_carry_guidance_group_and_meta(tmp_path):
    bundle = load_bundle(make_grouped_bundle(tmp_path / "b", site={"title": "Physics", "sides": "random"}))
    clip = bundle.clip("P2_seed44")
    assert (clip.prompt, clip.guidance, clip.group) == ("prompt P2", "look at P2", "P2")
    assert dict(clip.meta) == {"problem": "P2", "seed": 44}
    assert bundle.side_policy == "random" and bundle.site["title"] == "Physics"


@pytest.mark.parametrize("site, match", [
    ({"sides": "left"}, "sides"),
    ({"intro": "one string"}, "intro"),
    ({"choices": {"maybe": "x"}}, "choices"),
])
def test_bad_site_json(tmp_path, site, match):
    with pytest.raises(BundleError, match=match):
        load_bundle(make_grouped_bundle(tmp_path / "b", site=site))


def test_only_arms_restricts_the_vote(tmp_path):
    root = make_grouped_bundle(tmp_path / "b")
    bundle = load_bundle(root, ["trim", "v2"])
    assert bundle.arm_slugs == ("v2", "trim")
    assert all(set(c.videos) == {"v2", "trim"} for c in bundle.clips)
    with pytest.raises(BundleError, match="unknown"):
        load_bundle(root, ["v2", "nope"])
    with pytest.raises(BundleError, match="two arms"):
        load_bundle(root, ["v2"])


def _schedule(rounds, arms=("v2", "trim"), voters=("ann", "bob"), seed=0):
    availability = {cid: frozenset(arms) for cid in GROUPS}
    rng, history = random.Random(seed), []
    for i in range(rounds):
        voter = voters[i % len(voters)]
        m = choose_matchup(availability, arms, history, voter, rng, GROUPS, "random")
        history.append(PastMatchup(m.clip_id, m.left, m.right, voter))
    return history


def test_grouped_pairing_spreads_over_problems_then_seeds():
    history = _schedule(160)
    assert len({GROUPS[h.clip_id] for h in history[:40]}) == 40  # every problem before any repeats
    assert len({h.clip_id for h in history}) == 160  # every problem x seed once
    assert all(h.arm_a != h.arm_b for h in history)


def test_each_voter_sees_every_problem_before_a_repeat():
    history = _schedule(240, arms=ARMS, voters=("ann", "bob", "cy"))
    for voter in ("ann", "bob", "cy"):
        mine = [GROUPS[h.clip_id] for h in history if h.voter == voter]
        assert len(set(mine[:40])) == 40


def test_random_sides_are_coin_flips():
    history = _schedule(400, seed=3)
    lefts = Counter(h.arm_a for h in history)
    assert 150 < lefts["v2"] < 250
    with pytest.raises(ValueError):
        choose_matchup({"c": frozenset("ab")}, "ab", [], "v", random.Random(0), sides="sideways")


def test_ballot_has_guidance_and_vote_stores_problem_and_seed(tmp_path):
    bundle = load_bundle(make_grouped_bundle(tmp_path / "b", site={"sides": "random"}))
    store = VoteStore(tmp_path / "votes.jsonl")
    state = AppState(bundle, store, rng=random.Random(0))
    ballot = state.new_ballot("ann")
    clip = bundle.clip(ballot["clip_id"])
    assert ballot["guidance"] == clip.guidance
    assert state.info()["site"] == {"sides": "random"}
    state.record_vote({"ballot_id": ballot["ballot_id"], "voter": "ann", "choice": "left"})
    record = json.loads((tmp_path / "votes.jsonl").read_text())
    assert (record["problem"], record["seed"]) == (clip.meta["problem"], clip.meta["seed"])
    assert record["clip_id"] == ballot["clip_id"] and record["left_arm"] != record["right_arm"]
    assert VoteStore(tmp_path / "votes.jsonl").all()[0].clip_id == ballot["clip_id"]  # extra keys re-read fine
