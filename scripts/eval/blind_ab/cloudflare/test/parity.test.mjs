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

function checkPairing(sequence) {
  const { availability, slugs: armSlugs, history, cases } = sequence;
  const options = { groups: new Map(Object.entries(sequence.groups || {})), sides: sequence.sides || "balanced" };
  const avail = new Map(Object.entries(availability).map(([cid, a]) => [cid, new Set(a)]));
  for (const [i, c] of cases.entries()) {
    const calls = [];
    const rng = {
      choice: (items) => { calls.push(["choice", items]); return items[0]; },
      random: () => { calls.push(["random"]); return 0.75; },
    };
    const got = chooseMatchup(avail, armSlugs, history.slice(0, c.history_len), c.voter, rng, options);
    assert.deepEqual(got, c.expected, `case ${i}`);
    assert.deepEqual(calls, c.calls, `case ${i} candidate sets`);
  }
}

test("pairing reproduces pairing.choose_matchup decision by decision", () => checkPairing(fx.pairing));

test("pairing skips pairs a voter has exhausted, like pairing.choose_matchup", () => checkPairing(fx.pairing_exhaust));

test("grouped clips and random sides match pairing.choose_matchup", () => checkPairing(fx.pairing_groups));

test("grouped pairing covers every group, then every clip, with both arms on the same clip", () => {
  const avail = new Map();
  const groups = new Map();
  for (let p = 1; p <= 40; p += 1) {
    for (const seed of [42, 43, 44, 45]) {
      avail.set(`P${p}_seed${seed}`, new Set(["v2", "trim"]));
      groups.set(`P${p}_seed${seed}`, `P${p}`);
    }
  }
  const history = [];
  const lefts = { v2: 0, trim: 0 };
  for (let i = 0; i < 160; i += 1) {
    const m = chooseMatchup(avail, ["v2", "trim"], history, i % 2 ? "ann" : "bob", undefined, { groups, sides: "random" });
    assert.notEqual(m.left, m.right);
    lefts[m.left] += 1;
    history.push({ clip_id: m.clip_id, arm_a: m.left, arm_b: m.right, voter: i % 2 ? "ann" : "bob" });
    if (i === 39) assert.equal(new Set(history.map((h) => groups.get(h.clip_id))).size, 40, "first 40 hit 40 problems");
  }
  assert.equal(new Set(history.map((h) => h.clip_id)).size, 160, "160 ballots cover all 160 problem-seed clips");
  assert.ok(lefts.v2 > 50 && lefts.trim > 50, `sides look random: ${JSON.stringify(lefts)}`);
});

test("prompt entries, site.json and arm selection parse like bundle.py", async () => {
  const { parsePromptEntry, selectArms } = await import("../tools/load_bundle.mjs");
  assert.deepEqual(parsePromptEntry("a cat", "c", "p"), { prompt: "a cat", guidance: "", group: "", meta: {} });
  assert.deepEqual(parsePromptEntry({ prompt: "x", guidance: "g", group: "P1", meta: { seed: 42 } }, "c", "p"),
    { prompt: "x", guidance: "g", group: "P1", meta: { seed: 42 } });
  assert.throws(() => parsePromptEntry({ meta: { a: [1] } }, "c", "p"), /flat object/);
  const arms = [{ slug: "v2" }, { slug: "trim" }, { slug: "omni" }];
  assert.deepEqual(selectArms(arms, ["trim", "v2"]).map((a) => a.slug), ["v2", "trim"]);
  assert.throws(() => selectArms(arms, ["v2", "nope"]), /unknown arm/);
  assert.throws(() => selectArms(arms, ["v2"]), /at least two/);
});

test("vote export keeps flat extra fields such as problem and seed", () => {
  const line = voteToJson({ ...voteFromRecord(JSON.parse(fx.votes_jsonl[0])), problem: "P1", seed: 42 });
  const parsed = JSON.parse(line);
  assert.equal(parsed.problem, "P1");
  assert.equal(parsed.seed, 42);
  assert.equal(Object.keys(parsed).join(), Object.keys(parsed).sort().join());
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

test("build-time audio detection finds an MP4 'soun' handler", async () => {
  const { hasAudioTrack } = await import("../tools/media.mjs");
  const os = await import("node:os");
  const path = await import("node:path");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "blindab-media-"));
  const box = (handler) => Buffer.concat([Buffer.from([0, 0, 0, 33]), Buffer.from("hdlr"), Buffer.alloc(8),
    Buffer.from(handler), Buffer.alloc(13)]);
  const write = (name, ...handlers) => {
    const file = path.join(dir, name);
    fs.writeFileSync(file, Buffer.concat([Buffer.from("....ftypisom"), ...handlers.map(box)]));
    return file;
  };
  assert.equal(hasAudioTrack(write("av.mp4", "vide", "soun")), true);
  assert.equal(hasAudioTrack(write("v.mp4", "vide")), false);
  assert.equal(hasAudioTrack(write("x.webm", "soun")), null);
  fs.rmSync(dir, { recursive: true, force: true });
});
