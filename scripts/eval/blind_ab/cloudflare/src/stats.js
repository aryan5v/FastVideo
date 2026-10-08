// Port of ../../stats.py: pairwise tallies, Wilson intervals, Bradley-Terry.
// Ties and "both bad" count as half a win for each side. Keep in sync with stats.py.

import { pyCsvValue, pyRound } from "./pyfmt.js";

export const Z_95 = 1.959963984540054;
export const ELO_BASE = 1000.0;
export const ELO_SCALE = 400.0;

/** Wilson score interval for a binomial proportion (fractional counts OK). */
export function wilsonInterval(successes, n, z = Z_95) {
  if (n <= 0) return [0.0, 1.0];
  if (!(successes >= 0 && successes <= n)) throw new RangeError(`successes=${successes} must be within [0, n=${n}]`);
  const p = successes / n;
  const z2 = z * z;
  const denom = 1.0 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0.0, center - half), Math.min(1.0, center + half)];
}

/** Votes -> outcomes {armA, armB, scoreA}; skips unknown arms and self-pairs. */
export function outcomesFromVotes(votes, armSlugs) {
  const known = new Set(armSlugs);
  const result = [];
  for (const vote of votes) {
    if (!known.has(vote.left_arm) || !known.has(vote.right_arm) || vote.left_arm === vote.right_arm) continue;
    const score = vote.choice === "left" ? 1.0 : vote.choice === "right" ? 0.0 : 0.5;
    result.push({ armA: vote.left_arm, armB: vote.right_arm, scoreA: score });
  }
  return result;
}

/** Bradley-Terry strengths by the MM algorithm (Hunter, 2004), normalized to geometric mean 1. */
export function bradleyTerry(outcomes, armSlugs, { prior = 1.0, maxIter = 1000, tol = 1e-10 } = {}) {
  const slugs = [...armSlugs];
  const wins = new Map(slugs.map((s) => [s, prior * 0.5]));
  const games = new Map();
  const gameKey = (a, b) => `${a}\u0000${b}`;
  for (const o of outcomes) {
    if (!wins.has(o.armA) || !wins.has(o.armB)) continue;
    wins.set(o.armA, wins.get(o.armA) + o.scoreA);
    wins.set(o.armB, wins.get(o.armB) + (1.0 - o.scoreA));
    for (const key of [gameKey(o.armA, o.armB), gameKey(o.armB, o.armA)]) games.set(key, (games.get(key) || 0.0) + 1.0);
  }
  let strength = new Map(slugs.map((s) => [s, 1.0]));
  for (let iter = 0; iter < maxIter; iter += 1) {
    const raw = new Map();
    for (const s of slugs) {
      let denom = prior / (strength.get(s) + 1.0);
      for (const t of slugs) {
        const n = games.get(gameKey(s, t)) || 0.0;
        if (n) denom += n / (strength.get(s) + strength.get(t));
      }
      raw.set(s, denom > 0 ? wins.get(s) / denom : strength.get(s));
    }
    let logSum = 0;
    for (const v of raw.values()) logSum += Math.log(v);
    const scale = Math.exp(logSum / raw.size);
    const updated = new Map([...raw].map(([s, v]) => [s, v / scale]));
    let delta = 0;
    for (const s of slugs) delta = Math.max(delta, Math.abs(Math.log(updated.get(s)) - Math.log(strength.get(s))));
    strength = updated;
    if (delta < tol) break;
  }
  return Object.fromEntries(strength);
}

export function eloScale(strengths) {
  return Object.fromEntries(Object.entries(strengths).map(([s, v]) => [s, ELO_BASE + ELO_SCALE * Math.log10(v)]));
}

/** matrix[a][b] = {wins, losses, ties, both_bad} from a's perspective. */
export function pairwiseMatrix(votes, armSlugs) {
  const matrix = {};
  for (const a of armSlugs) {
    matrix[a] = {};
    for (const b of armSlugs) if (b !== a) matrix[a][b] = { wins: 0, losses: 0, ties: 0, both_bad: 0 };
  }
  const has = (slug) => Object.prototype.hasOwnProperty.call(matrix, slug);
  for (const vote of votes) {
    const a = vote.left_arm;
    const b = vote.right_arm;
    if (!has(a) || !has(b) || a === b) continue;
    if (vote.choice === "left") {
      matrix[a][b].wins += 1;
      matrix[b][a].losses += 1;
    } else if (vote.choice === "right") {
      matrix[a][b].losses += 1;
      matrix[b][a].wins += 1;
    } else {
      const key = vote.choice === "tie" ? "ties" : "both_bad";
      matrix[a][b][key] += 1;
      matrix[b][a][key] += 1;
    }
  }
  return matrix;
}

/** Seconds per clip when it is a positive number (mirrors Arm.seconds_per_clip). */
export function secondsPerClip(arm) {
  const value = arm.speed ? arm.speed.seconds_per_clip : undefined;
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function armRow(arm, cells, bt, baseline) {
  const sum = (key) => Object.values(cells).reduce((acc, c) => acc + c[key], 0);
  const wins = sum("wins");
  const losses = sum("losses");
  const ties = sum("ties");
  const bothBad = sum("both_bad");
  const n = wins + losses + ties + bothBad;
  const score = wins + 0.5 * (ties + bothBad);
  const [low, high] = wilsonInterval(score, n);
  const spc = secondsPerClip(arm);
  const baseSpc = baseline ? secondsPerClip(baseline) : null;
  const speed = arm.speed || {};
  return {
    slug: arm.slug,
    display_name: arm.display_name,
    notes: arm.notes,
    votes: n,
    wins,
    losses,
    ties,
    both_bad: bothBad,
    win_rate: n ? score / n : null,
    win_rate_ci_low: n ? low : null,
    win_rate_ci_high: n ? high : null,
    bt_score: pyRound(bt, 1),
    seconds_per_clip: spc,
    speedup_vs_baseline: spc && baseSpc ? pyRound(baseSpc / spc, 3) : null,
    hardware: speed.hardware ?? null,
    resolution: speed.resolution ?? null,
  };
}

/** Port of stats.summarize: per-arm rows sorted by BT score, plus the head-to-head matrix. */
export function summarize(arms, votes, baseline = null) {
  const slugs = arms.map((a) => a.slug);
  let baseArm = arms.find((a) => a.slug === baseline) || null;
  if (!baseArm) baseArm = arms.find((a) => secondsPerClip(a)) || null;
  const outcomes = outcomesFromVotes(votes, slugs);
  const bt = eloScale(bradleyTerry(outcomes, slugs));
  const matrix = pairwiseMatrix(votes, slugs);
  const rows = arms.map((arm) => armRow(arm, matrix[arm.slug], bt[arm.slug], baseArm));
  // Stable sort, descending by score (same order as Python's sort(reverse=True) for unequal keys;
  // Python keeps original order for equal keys, as does this).
  const sorted = rows.map((row, i) => [row, i]).sort((x, y) => y[0].bt_score - x[0].bt_score || x[1] - y[1]);
  return {
    total_votes: outcomes.length,
    ignored_votes: votes.length - outcomes.length,
    voters: [...new Set(votes.map((v) => v.voter))].sort(comparePyStr),
    baseline: baseArm ? baseArm.slug : null,
    arms: sorted.map(([row]) => row),
    matrix,
  };
}

/** Python string ordering (by code point), unlike JS's default UTF-16 unit order. */
export function comparePyStr(a, b) {
  const ca = [...a];
  const cb = [...b];
  for (let i = 0; i < Math.min(ca.length, cb.length); i += 1) {
    const d = ca[i].codePointAt(0) - cb[i].codePointAt(0);
    if (d) return d;
  }
  return ca.length - cb.length;
}

export const CSV_COLUMNS = ["slug", "display_name", "votes", "wins", "losses", "ties", "both_bad", "win_rate",
  "win_rate_ci_low", "win_rate_ci_high", "bt_score", "seconds_per_clip", "speedup_vs_baseline", "hardware",
  "resolution"];
const FLOAT_COLUMNS = new Set(["win_rate", "win_rate_ci_low", "win_rate_ci_high", "bt_score", "seconds_per_clip",
  "speedup_vs_baseline"]);

export function summaryToCsv(summary) {
  const lines = [CSV_COLUMNS.join(",")];
  for (const row of summary.arms) {
    lines.push(CSV_COLUMNS.map((k) => pyCsvValue(row[k], FLOAT_COLUMNS.has(k))).join(","));
  }
  return `${lines.join("\n")}\n`;
}
