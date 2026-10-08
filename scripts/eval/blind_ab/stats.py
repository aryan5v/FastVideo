"""Preference statistics: pairwise tallies, Wilson intervals, Bradley-Terry.

Scoring convention: a win counts 1, a loss 0, and "tie" / "both bad" count
0.5 for each side. The per-arm win rate is ``(wins + 0.5 * ties) / votes``
with a Wilson 95% interval on that rate.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .bundle import Arm
from .votes import Vote

Z_95 = 1.959963984540054
ELO_BASE = 1000.0
ELO_SCALE = 400.0


@dataclass(frozen=True)
class Outcome:
    """One comparison: ``score_a`` is 1 (a wins), 0 (b wins) or 0.5 (tie)."""
    arm_a: str
    arm_b: str
    score_a: float


def wilson_interval(successes: float, n: float, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (fractional counts OK)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} must be within [0, n={n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def outcomes_from_votes(votes: Iterable[Vote], arm_slugs: Iterable[str]) -> tuple[Outcome, ...]:
    known = set(arm_slugs)
    result = []
    for vote in votes:
        if vote.left_arm not in known or vote.right_arm not in known or vote.left_arm == vote.right_arm:
            continue
        score = {"left": 1.0, "right": 0.0}.get(vote.choice, 0.5)
        result.append(Outcome(vote.left_arm, vote.right_arm, score))
    return tuple(result)


def bradley_terry(outcomes: Sequence[Outcome],
                  arm_slugs: Sequence[str],
                  prior: float = 1.0,
                  max_iter: int = 1000,
                  tol: float = 1e-10) -> dict[str, float]:
    """Fit Bradley-Terry strengths with the MM algorithm (Hunter, 2004).

    Ties count as half a win for each side. ``prior`` adds that many virtual
    games (split evenly) against a reference opponent of strength 1, which keeps
    the fit finite for undefeated/winless arms. Returns strengths normalized to
    geometric mean 1.
    """
    slugs = list(arm_slugs)
    wins = {s: prior * 0.5 for s in slugs}
    games: dict[tuple[str, str], float] = {}
    for o in outcomes:
        if o.arm_a not in wins or o.arm_b not in wins:
            continue
        wins[o.arm_a] += o.score_a
        wins[o.arm_b] += 1.0 - o.score_a
        for key in ((o.arm_a, o.arm_b), (o.arm_b, o.arm_a)):
            games[key] = games.get(key, 0.0) + 1.0
    strength = {s: 1.0 for s in slugs}
    for _ in range(max_iter):
        updated = {}
        for s in slugs:
            denom = prior / (strength[s] + 1.0)
            for t in slugs:
                n = games.get((s, t), 0.0)
                if n:
                    denom += n / (strength[s] + strength[t])
            updated[s] = wins[s] / denom if denom > 0 else strength[s]
        log_mean = sum(math.log(v) for v in updated.values()) / len(updated)
        updated = {s: v / math.exp(log_mean) for s, v in updated.items()}
        delta = max(abs(math.log(updated[s]) - math.log(strength[s])) for s in slugs)
        strength = updated
        if delta < tol:
            break
    return strength


def elo_scale(strengths: Mapping[str, float]) -> dict[str, float]:
    return {s: ELO_BASE + ELO_SCALE * math.log10(v) for s, v in strengths.items()}


def pairwise_matrix(votes: Iterable[Vote], arm_slugs: Sequence[str]) -> dict[str, dict[str, dict[str, int]]]:
    """``matrix[a][b] = {wins, losses, ties, both_bad}`` from a's perspective."""
    matrix = {
        a: {b: {"wins": 0, "losses": 0, "ties": 0, "both_bad": 0}
            for b in arm_slugs if b != a}
        for a in arm_slugs
    }
    for vote in votes:
        a, b = vote.left_arm, vote.right_arm
        if a not in matrix or b not in matrix or a == b:
            continue
        if vote.choice == "left":
            matrix[a][b]["wins"] += 1
            matrix[b][a]["losses"] += 1
        elif vote.choice == "right":
            matrix[a][b]["losses"] += 1
            matrix[b][a]["wins"] += 1
        else:
            key = "ties" if vote.choice == "tie" else "both_bad"
            matrix[a][b][key] += 1
            matrix[b][a][key] += 1
    return matrix


def _arm_row(arm: Arm, cells: Mapping[str, Mapping[str, int]], bt: float, baseline: Arm | None) -> dict[str, Any]:
    wins = sum(c["wins"] for c in cells.values())
    losses = sum(c["losses"] for c in cells.values())
    ties = sum(c["ties"] for c in cells.values())
    both_bad = sum(c["both_bad"] for c in cells.values())
    n = wins + losses + ties + both_bad
    score = wins + 0.5 * (ties + both_bad)
    low, high = wilson_interval(score, n)
    spc = arm.seconds_per_clip
    base_spc = baseline.seconds_per_clip if baseline else None
    return {
        "slug": arm.slug,
        "display_name": arm.display_name,
        "notes": arm.notes,
        "votes": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "both_bad": both_bad,
        "win_rate": score / n if n else None,
        "win_rate_ci_low": low if n else None,
        "win_rate_ci_high": high if n else None,
        "bt_score": round(bt, 1),
        "seconds_per_clip": spc,
        "speedup_vs_baseline": round(base_spc / spc, 3) if spc and base_spc else None,
        "hardware": arm.speed.get("hardware"),
        "resolution": arm.speed.get("resolution"),
    }


def summarize(arms: Sequence[Arm], votes: Sequence[Vote], baseline: str | None = None) -> dict[str, Any]:
    slugs = [arm.slug for arm in arms]
    base_arm = next((a for a in arms if a.slug == baseline), None)
    if base_arm is None:
        base_arm = next((a for a in arms if a.seconds_per_clip), None)
    outcomes = outcomes_from_votes(votes, slugs)
    bt = elo_scale(bradley_terry(outcomes, slugs))
    matrix = pairwise_matrix(votes, slugs)
    rows = [_arm_row(arm, matrix[arm.slug], bt[arm.slug], base_arm) for arm in arms]
    rows.sort(key=lambda r: r["bt_score"], reverse=True)
    return {
        "total_votes": len(outcomes),
        "ignored_votes": len(votes) - len(outcomes),
        "voters": sorted({v.voter for v in votes}),
        "baseline": base_arm.slug if base_arm else None,
        "arms": rows,
        "matrix": matrix,
    }


CSV_COLUMNS = ("slug", "display_name", "votes", "wins", "losses", "ties", "both_bad", "win_rate", "win_rate_ci_low",
               "win_rate_ci_high", "bt_score", "seconds_per_clip", "speedup_vs_baseline", "hardware", "resolution")


def summary_to_csv(summary: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in summary["arms"]:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})
    return buffer.getvalue()
