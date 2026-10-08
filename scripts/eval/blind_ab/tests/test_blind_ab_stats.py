"""Wilson intervals, Bradley-Terry fitting and the results summary. CPU only."""
import csv
import io
import math
import random
from types import MappingProxyType

import pytest

from blind_ab.bundle import Arm
from blind_ab.stats import (Outcome, bradley_terry, elo_scale, outcomes_from_votes, pairwise_matrix, summarize,
                            summary_to_csv, wilson_interval)
from blind_ab.votes import Vote


def test_wilson_known_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(2.5, 5)  # fractional counts from ties
    assert low < 0.5 < high and low == pytest.approx(1 - high)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


def test_wilson_narrows_with_more_data():
    w10 = wilson_interval(7, 10)
    w1000 = wilson_interval(700, 1000)
    assert (w1000[1] - w1000[0]) < (w10[1] - w10[0])


def test_bradley_terry_recovers_known_strengths():
    rng = random.Random(0)
    truth = {"a": 4.0, "b": 2.0, "c": 1.0, "d": 0.5}
    slugs = list(truth)
    outcomes = []
    for _ in range(6000):
        x, y = rng.sample(slugs, 2)
        p = truth[x] / (truth[x] + truth[y])
        outcomes.append(Outcome(x, y, 1.0 if rng.random() < p else 0.0))
    fit = bradley_terry(outcomes, slugs, prior=0.0)
    assert sorted(slugs, key=fit.get, reverse=True) == ["a", "b", "c", "d"]
    for x in slugs:
        for y in slugs:
            assert math.log(fit[x] / fit[y]) == pytest.approx(math.log(truth[x] / truth[y]), abs=0.2)
    assert sum(math.log(v) for v in fit.values()) == pytest.approx(0.0, abs=1e-9)


def test_bradley_terry_ties_and_undefeated_stay_finite():
    outcomes = [Outcome("a", "b", 1.0)] * 5 + [Outcome("b", "c", 0.5)] * 4
    fit = bradley_terry(outcomes, ["a", "b", "c"])
    assert all(math.isfinite(v) and v > 0 for v in fit.values())
    assert fit["a"] > fit["b"]
    assert fit["b"] == pytest.approx(fit["c"], rel=0.3)
    assert bradley_terry([], ["a", "b"]) == {"a": 1.0, "b": 1.0}


def test_elo_scale_centered():
    scores = elo_scale({"a": 10.0, "b": 0.1})
    assert scores["a"] == pytest.approx(1400.0) and scores["b"] == pytest.approx(600.0)


def _v(left, right, choice):
    return Vote("ann", "t", "c0", left, right, choice)


ARMS = (
    Arm("base", "Base", speed=MappingProxyType({"seconds_per_clip": 100.0, "hardware": "gpu"})),
    Arm("fast", "Fast", speed=MappingProxyType({"seconds_per_clip": 25.0})),
    Arm("nospeed", "No speed"),
)
VOTES = [_v("base", "fast", "left"), _v("fast", "base", "right"), _v("base", "fast", "tie"),
         _v("nospeed", "fast", "both_bad"), _v("base", "ghost", "left")]


def test_matrix_and_outcomes_ignore_unknown_arms():
    slugs = [a.slug for a in ARMS]
    assert len(outcomes_from_votes(VOTES, slugs)) == 4
    m = pairwise_matrix(VOTES, slugs)
    assert m["base"]["fast"] == {"wins": 2, "losses": 0, "ties": 1, "both_bad": 0}
    assert m["fast"]["base"] == {"wins": 0, "losses": 2, "ties": 1, "both_bad": 0}
    assert m["fast"]["nospeed"]["both_bad"] == 1


def test_summary_rows_and_speed_columns():
    summary = summarize(ARMS, VOTES, baseline="base")
    rows = {r["slug"]: r for r in summary["arms"]}
    assert summary["total_votes"] == 4 and summary["ignored_votes"] == 1
    assert summary["arms"][0]["slug"] == "base"
    assert rows["base"]["win_rate"] == pytest.approx(2.5 / 3)
    assert rows["fast"]["speedup_vs_baseline"] == pytest.approx(4.0)
    assert rows["base"]["speedup_vs_baseline"] == pytest.approx(1.0)
    assert rows["nospeed"]["speedup_vs_baseline"] is None
    assert rows["base"]["win_rate_ci_low"] < rows["base"]["win_rate"] < rows["base"]["win_rate_ci_high"]
    # Default baseline: first arm with speed metadata.
    assert summarize(ARMS, VOTES)["baseline"] == "base"
    assert summarize(ARMS, VOTES, baseline="fast")["arms"][0]["speedup_vs_baseline"] == pytest.approx(0.25)


def test_csv_export():
    text = summary_to_csv(summarize(ARMS, VOTES, baseline="base"))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["slug"] for r in rows][0] == "base"
    assert rows[0]["seconds_per_clip"] == "100.0"
    assert next(r for r in rows if r["slug"] == "nospeed")["speedup_vs_baseline"] == ""
