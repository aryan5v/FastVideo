// Parity tests: the JS port must reproduce the Python implementation on the same synthetic votes.
// Reference values come from make_fixtures.py (run it after changing stats.py / pairing.py / votes.py).
//
//   node --test scripts/eval/blind_ab/cloudflare/test/

import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import { parseRange } from "../src/http.js";
import { chooseMatchup } from "../src/pairing.js";
import { pyFloatStr, pyRound } from "../src/pyfmt.js";
import { bradleyTerry, eloScale, outcomesFromVotes, summarize, summaryToCsv, wilsonInterval } from "../src/stats.js";
import { cleanVoteFields, voteFromRecord, voteToJson } from "../src/votes.js";
import { recordToInsert } from "../tools/votes_to_sql.mjs";

const fx = JSON.parse(fs.readFileSync(new URL("./fixtures.json", import.meta.url), "utf8"));
const votes = fx.votes_jsonl.map((line) => voteFromRecord(JSON.parse(line)));
const arms = fx.arms;
const slugs = arms.map((a) => a.slug);
const TOL = 1e-12;

function assertClose(actual, expected, tol = TOL, label = "") {
  if (expected === null) return assert.equal(actual, null, label);
  assert.ok(Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)), `${label}: ${actual} vs ${expected}`);
}

function assertMapsClose(actual, expected, tol = TOL) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
  for (const key of Object.keys(expected)) assertClose(actual[key], expected[key], tol, key);
}

test("Wilson intervals match stats.wilson_interval", () => {
  for (const { args, expected } of fx.wilson) {
    const [low, high] = wilsonInterval(...args);
    assertClose(low, expected[0], TOL, `low ${args}`);
    assertClose(high, expected[1], TOL, `high ${args}`);
  }
  assert.throws(() => wilsonInterval(6, 5), RangeError);
});

test("Bradley-Terry strengths and Elo scores match stats.bradley_terry", () => {
  const outcomes = outcomesFromVotes(votes, slugs);
  assertMapsClose(bradleyTerry(outcomes, slugs), fx.bradley_terry.default_prior, 1e-9);
  assertMapsClose(bradleyTerry(outcomes, slugs, { prior: 0.5 }), fx.bradley_terry.half_prior, 1e-9);
  assertMapsClose(eloScale(bradleyTerry(outcomes, slugs)), fx.bradley_terry.elo, 1e-9);
  const toy = [...Array(5).fill({ left_arm: "a", right_arm: "b", choice: "left" }),
    ...Array(4).fill({ left_arm: "b", right_arm: "c", choice: "tie" })];
  assertMapsClose(bradleyTerry(outcomesFromVotes(toy, ["a", "b", "c"]), ["a", "b", "c"]), fx.bradley_terry.undefeated, 1e-9);
});

test("results summary matches stats.summarize for every baseline", () => {
  for (const [baseline, expected] of Object.entries(fx.summary)) {
    const actual = summarize(arms, votes, baseline || null);
    assert.equal(actual.total_votes, expected.total_votes);
    assert.equal(actual.ignored_votes, expected.ignored_votes);
    assert.deepEqual(actual.voters, expected.voters);
    assert.equal(actual.baseline, expected.baseline);
    assert.deepEqual(actual.matrix, expected.matrix);
    assert.deepEqual(actual.arms.map((r) => r.slug), expected.arms.map((r) => r.slug), "BT order");
    actual.arms.forEach((row, i) => {
      const want = expected.arms[i];
      assert.deepEqual(Object.keys(row).sort(), Object.keys(want).sort());
      for (const [key, value] of Object.entries(want)) {
        if (typeof value === "number" && !Number.isInteger(value)) assertClose(row[key], value, 1e-12, `${row.slug}.${key}`);
        else assert.deepEqual(row[key], value, `${row.slug}.${key}`);
      }
    });
  }
});

test("CSV export is byte-identical to stats.summary_to_csv", () => {
  for (const [baseline, expected] of Object.entries(fx.csv)) {
    assert.equal(summaryToCsv(summarize(arms, votes, baseline || null)), expected);
  }
});

test("pairing reproduces pairing.choose_matchup decision by decision", () => {
  const { availability, slugs: armSlugs, history, cases } = fx.pairing;
  const avail = new Map(Object.entries(availability).map(([cid, a]) => [cid, new Set(a)]));
  for (const [i, c] of cases.entries()) {
    const calls = [];
    const rng = {
      choice: (items) => { calls.push(["choice", items]); return items[0]; },
      random: () => { calls.push(["random"]); return 0.75; },
    };
    const got = chooseMatchup(avail, armSlugs, history.slice(0, c.history_len), c.voter, rng);
    assert.deepEqual(got, c.expected, `case ${i}`);
    assert.deepEqual(calls, c.calls, `case ${i} candidate sets`);
  }
});

test("pairing with the default RNG keeps every pair covered evenly", () => {
  const avail = new Map(["c1", "c2", "c3"].map((c) => [c, new Set(["a", "b", "c"])]));
  const history = [];
  for (let i = 0; i < 30; i += 1) {
    const m = chooseMatchup(avail, ["a", "b", "c"], history, "v");
    history.push({ clip_id: m.clip_id, arm_a: m.left, arm_b: m.right, voter: "v" });
  }
  const counts = {};
  for (const h of history) counts[[h.arm_a, h.arm_b].sort().join()] = (counts[[h.arm_a, h.arm_b].sort().join()] || 0) + 1;
  assert.deepEqual(Object.values(counts), [10, 10, 10]);
});

test("vote records serialize exactly like Vote.to_json", () => {
  for (const line of fx.votes_jsonl) assert.equal(voteToJson(voteFromRecord(JSON.parse(line))), line);
});

test("vote validation matches votes.clean_vote_fields", () => {
  for (const { payload, expected } of fx.clean_vote_fields) assert.deepEqual(cleanVoteFields(payload), expected);
  assert.throws(() => cleanVoteFields({ voter: "a", choice: "maybe" }), /choice must be one of/);
  assert.throws(() => cleanVoteFields({ voter: " ", choice: "tie" }), /voter name is required/);
  assert.throws(() => cleanVoteFields({ voter: "a", choice: "tie", watch_seconds: -1 }), /non-negative/);
});

test("Python round() and float repr are reproduced", () => {
  for (const { x, round1, round2, round3, repr } of fx.numbers) {
    assert.equal(pyRound(x, 1), round1, `round(${x}, 1)`);
    assert.equal(pyRound(x, 2), round2, `round(${x}, 2)`);
    assert.equal(pyRound(x, 3), round3, `round(${x}, 3)`);
    assert.equal(pyFloatStr(x), repr, `repr(${x})`);
  }
});

test("Range parsing matches serve.parse_range", () => {
  assert.equal(parseRange(null, 100), null);
  assert.deepEqual(parseRange("bytes=0-", 100), [0, 99]);
  assert.deepEqual(parseRange("bytes=10-19", 100), [10, 19]);
  assert.deepEqual(parseRange("bytes=90-500", 100), [90, 99]);
  assert.deepEqual(parseRange("bytes=-10", 100), [90, 99]);
  for (const bad of ["bytes=100-", "bytes=5-2", "bytes=-0", "bytes=-", "items=0-1"]) {
    assert.throws(() => parseRange(bad, 100), RangeError, bad);
  }
});

test("JSONL import keeps every field", () => {
  const sql = recordToInsert({ ...JSON.parse(fx.votes_jsonl[0]), comment: "it's", session: 7 });
  assert.match(sql, /^INSERT OR IGNORE INTO votes \(voter, timestamp, clip_id, left_arm, right_arm, choice, comment, /);
  assert.match(sql, /'it''s'/);
  assert.match(sql, /'\{"session":7\}'\);$/);
});
