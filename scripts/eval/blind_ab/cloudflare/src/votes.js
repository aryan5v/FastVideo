// Port of ../../votes.py: vote validation and the JSONL record format.

import { pyJsonFlat, pyRound } from "./pyfmt.js";

export const CHOICES = Object.freeze(["left", "right", "tie", "both_bad"]);
export const MAX_VOTER_LEN = 64;
export const MAX_COMMENT_LEN = 1000;
export const MAX_SECONDS = 24 * 3600.0;
export const VOTE_FIELDS = Object.freeze(["voter", "timestamp", "clip_id", "left_arm", "right_arm", "choice",
  "comment", "watch_seconds", "played_seconds", "ballot_id"]);
const FLOAT_FIELDS = new Set(["watch_seconds", "played_seconds"]);

export class VoteError extends Error {}

const codePointLength = (text) => [...text].length;
// Python: str(raw or "") -- falsy values become "".
const pyStr = (raw) => (raw === null || raw === undefined || raw === false || raw === 0 || raw === "" ? "" : String(raw));

export function utcNow(date = new Date()) {
  // Python: datetime.now(timezone.utc).isoformat(timespec="seconds") -> 2026-01-01T12:00:00+00:00
  return `${date.toISOString().slice(0, 19)}+00:00`;
}

export function cleanVoter(raw) {
  const voter = pyStr(raw).trim();
  if (!voter) throw new VoteError("voter name is required");
  if (codePointLength(voter) > MAX_VOTER_LEN) throw new VoteError(`voter name longer than ${MAX_VOTER_LEN} characters`);
  return voter;
}

function toFloat(raw, name) {
  if (raw === null || raw === undefined || raw === false || raw === 0 || raw === "") return 0.0;
  if (raw === true) return 1.0;
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const text = raw.trim().toLowerCase();
    if (/^[+-]?(inf|infinity)$/.test(text)) return text.startsWith("-") ? -Infinity : Infinity;
    if (/^[+-]?nan$/.test(text)) return NaN;
    const value = Number(text.replaceAll("_", ""));
    if (text && Number.isFinite(value)) return value;
  }
  throw new VoteError(`${name} must be a number`);
}

function cleanSeconds(raw, name) {
  const value = toFloat(raw, name);
  if (Number.isNaN(value) || value < 0) throw new VoteError(`${name} must be a non-negative number`);
  return pyRound(Math.min(value, MAX_SECONDS), 2);
}

/** Validate the client-supplied part of a vote. */
export function cleanVoteFields(payload) {
  const choice = payload.choice;
  if (!CHOICES.includes(choice)) throw new VoteError(`choice must be one of ('left', 'right', 'tie', 'both_bad')`);
  const comment = pyStr(payload.comment).trim();
  if (codePointLength(comment) > MAX_COMMENT_LEN) {
    throw new VoteError(`comment longer than ${MAX_COMMENT_LEN} characters`);
  }
  return {
    voter: cleanVoter(payload.voter),
    choice,
    comment,
    watch_seconds: cleanSeconds(payload.watch_seconds, "watch_seconds"),
    played_seconds: cleanSeconds(payload.played_seconds, "played_seconds"),
  };
}

/** Normalize a stored record (JSONL line or D1 row) into the canonical vote shape. */
export function voteFromRecord(record) {
  for (const key of ["voter", "timestamp", "clip_id", "left_arm", "right_arm", "choice"]) {
    if (record[key] === undefined || record[key] === null) throw new VoteError(`malformed vote record: '${key}'`);
  }
  const vote = {
    voter: String(record.voter),
    timestamp: String(record.timestamp),
    clip_id: String(record.clip_id),
    left_arm: String(record.left_arm),
    right_arm: String(record.right_arm),
    choice: String(record.choice),
    comment: pyStr(record.comment),
    watch_seconds: Number(record.watch_seconds || 0.0),
    played_seconds: Number(record.played_seconds || 0.0),
    ballot_id: pyStr(record.ballot_id),
  };
  if (!CHOICES.includes(vote.choice)) throw new VoteError(`unknown choice '${vote.choice}'`);
  if (Number.isNaN(vote.watch_seconds) || Number.isNaN(vote.played_seconds)) {
    throw new VoteError("malformed vote record: seconds must be numbers");
  }
  return Object.freeze(vote);
}

/** One JSONL line, identical to Vote.to_json() in votes.py. */
export function voteToJson(vote) {
  return pyJsonFlat(Object.fromEntries(VOTE_FIELDS.map((k) => [k, vote[k]])), FLOAT_FIELDS);
}
