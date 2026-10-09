// Port of ../../pairing.py: balanced matchup scheduling for blind pairwise voting.
//
// Each round picks the arm pair with the fewest comparisons so far (skipping pairs this voter has seen on
// every clip while others remain), then the clip that pair has been compared on least (preferring clips
// this voter has not seen for that pair, then the clip whose group this voter, then everyone, has seen least).
// Sides: "balanced" puts the arm shown on the left less often on the left (random on ties); "random" flips a
// fair coin. Keep in sync with pairing.py.

import { comparePyStr } from "./stats.js";

export const SIDE_POLICIES = Object.freeze(["balanced", "random"]);

/** Default RNG with the two methods the scheduler needs (same interface as Python's random.Random). */
export const mathRandom = Object.freeze({
  random: () => Math.random(),
  choice: (items) => items[Math.floor(Math.random() * items.length)],
});

export function pairKey(a, b) {
  return comparePyStr(a, b) <= 0 ? [a, b] : [b, a];
}

const keyOf = (pair) => `${pair[0]}\u0000${pair[1]}`;

/** All unordered arm pairs (sorted) that share at least one clip. `availability`: clip_id -> Set of arms. */
export function availablePairs(availability, armSlugs) {
  const slugs = [...new Set(armSlugs)].sort(comparePyStr);
  const pairs = [];
  for (let i = 0; i < slugs.length; i += 1) {
    for (let j = i + 1; j < slugs.length; j += 1) {
      const [a, b] = [slugs[i], slugs[j]];
      if ([...availability.values()].some((arms) => arms.has(a) && arms.has(b))) pairs.push([a, b]);
    }
  }
  return pairs;
}

function compareKeys(x, y) {
  // Lexicographic comparison of tuples of booleans / numbers.
  for (let i = 0; i < x.length; i += 1) {
    const d = Number(x[i]) - Number(y[i]);
    if (d) return d;
  }
  return 0;
}

function argminChoice(items, key, rng) {
  const keys = items.map(key);
  const best = keys.reduce((acc, k) => (compareKeys(k, acc) < 0 ? k : acc));
  return rng.choice(items.filter((_, i) => compareKeys(keys[i], best) === 0));
}

function tally(history, voter, groups) {
  const counts = {
    pair: new Map(), clipPair: new Map(), clip: new Map(), left: new Map(), appear: new Map(), group: new Map(),
    voterGroup: new Map(),
  };
  const seen = new Set();
  const bump = (map, key) => map.set(key, (map.get(key) || 0) + 1);
  for (const past of history) {
    const group = groups.get(past.clip_id);
    if (group) {
      bump(counts.group, group);
      if (voter && past.voter === voter) bump(counts.voterGroup, group);
    }
    const key = keyOf(pairKey(past.arm_a, past.arm_b));
    bump(counts.pair, key);
    bump(counts.clipPair, `${past.clip_id}\u0001${key}`);
    bump(counts.clip, past.clip_id);
    bump(counts.left, past.arm_a); // arm_a is the left side
    bump(counts.appear, past.arm_a);
    bump(counts.appear, past.arm_b);
    if (voter && past.voter === voter) seen.add(`${past.clip_id}\u0001${key}`);
  }
  return { counts, seen };
}

/**
 * Pick the next blinded matchup.
 * @param {Map<string, Set<string>>} availability clip_id -> arms with a video for it
 * @param {string[]} armSlugs
 * @param {{clip_id, arm_a, arm_b, voter}[]} history past (and pending) comparisons; arm_a was on the left
 * @param {{groups?: Map<string, string>, sides?: "balanced" | "random"}} [options] clip_id -> group id, side policy
 * @returns {{clip_id: string, left: string, right: string}}
 */
export function chooseMatchup(availability, armSlugs, history, voter = "", rng = mathRandom, options = {}) {
  const groups = options.groups || new Map();
  const sides = options.sides || "balanced";
  if (!SIDE_POLICIES.includes(sides)) throw new Error(`sides must be one of ${SIDE_POLICIES}`);
  const pairs = availablePairs(availability, armSlugs);
  if (!pairs.length) throw new Error("no arm pair shares a clip; need at least two arms per clip");
  const { counts, seen } = tally(history, voter, groups);
  const get = (map, key) => map.get(key) || 0;

  const clipsFor = (p) => [...availability.entries()]
    .filter(([, arms]) => arms.has(p[0]) && arms.has(p[1]))
    .map(([cid]) => cid)
    .sort(comparePyStr);
  // A pair is exhausted when this voter has already seen it on every clip it shares.
  const exhausted = (p) => clipsFor(p).every((cid) => seen.has(`${cid}\u0001${keyOf(p)}`));
  const pair = argminChoice(pairs, (p) => [exhausted(p), get(counts.pair, keyOf(p))], rng);
  const pk = keyOf(pair);
  const clips = clipsFor(pair);
  const clipId = argminChoice(clips, (cid) => {
    const ck = `${cid}\u0001${pk}`;
    const group = groups.get(cid);
    const groupLoad = group ? [get(counts.voterGroup, group), get(counts.group, group)] : [0, 0];
    return [seen.has(ck), get(counts.clipPair, ck), ...groupLoad, get(counts.clip, cid)];
  }, rng);
  const [a, b] = pair;
  if (sides === "random") {
    const [left, right] = rng.random() < 0.5 ? [a, b] : [b, a];
    return { clip_id: clipId, left, right };
  }
  const surplus = (arm) => get(counts.left, arm) - get(counts.appear, arm) / 2;
  let lean = surplus(a) - surplus(b);
  if (lean === 0) lean = rng.random() - 0.5;
  const [left, right] = lean > 0 ? [b, a] : [a, b];
  return { clip_id: clipId, left, right };
}
