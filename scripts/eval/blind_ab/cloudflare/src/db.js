// D1 access for votes and issued (blinded) ballots. All SQL lives here.

import { voteFromRecord } from "./votes.js";

export const PENDING_TTL_SECONDS = 15 * 60;
export const BALLOT_KEEP_SECONDS = 3 * 24 * 3600; // ballots older than this are pruned (votes are kept forever)

const VOTE_COLUMNS = "voter, timestamp, clip_id, left_arm, right_arm, choice, comment, watch_seconds, "
  + "played_seconds, ballot_id";

function rowToVote(row) {
  const vote = voteFromRecord(row);
  if (!row.extra) return vote;
  try {
    return Object.freeze({ ...JSON.parse(row.extra), ...vote });
  } catch (err) {
    console.error("ignoring malformed extra JSON on vote", row.id, err);
    return vote;
  }
}

export async function allVotes(db) {
  const { results } = await db.prepare(`SELECT id, ${VOTE_COLUMNS}, extra FROM votes ORDER BY id`).all();
  return results.map(rowToVote);
}

export async function countVotes(db, voter = null) {
  const stmt = voter === null
    ? db.prepare("SELECT COUNT(*) AS n FROM votes")
    : db.prepare("SELECT COUNT(*) AS n FROM votes WHERE voter = ?").bind(voter);
  const row = await stmt.first();
  return row ? row.n : 0;
}

/** Past votes plus recent unvoted ballots, as pairing history ({clip_id, arm_a, arm_b, voter}). */
export async function matchupHistory(db, nowSeconds) {
  const [votes, pending] = await db.batch([
    db.prepare("SELECT clip_id, left_arm AS arm_a, right_arm AS arm_b, voter FROM votes ORDER BY id"),
    db.prepare("SELECT clip_id, left_arm AS arm_a, right_arm AS arm_b, voter FROM ballots "
      + "WHERE voted = 0 AND issued_at > ? ORDER BY issued_at").bind(nowSeconds - PENDING_TTL_SECONDS),
  ]);
  return [...votes.results, ...pending.results];
}

export async function insertBallot(db, ballot, nowSeconds) {
  await db.batch([
    db.prepare("DELETE FROM ballots WHERE issued_at < ?").bind(nowSeconds - BALLOT_KEEP_SECONDS),
    db.prepare("INSERT INTO ballots (ballot_id, clip_id, left_arm, right_arm, voter, issued_at, voted) "
      + "VALUES (?, ?, ?, ?, ?, ?, 0)")
      .bind(ballot.ballot_id, ballot.clip_id, ballot.left, ballot.right, ballot.voter, nowSeconds),
  ]);
}

export async function getBallot(db, ballotId) {
  return db.prepare("SELECT ballot_id, clip_id, left_arm, right_arm, voter, voted FROM ballots WHERE ballot_id = ?")
    .bind(ballotId).first();
}

/** Store a vote for a ballot exactly once. Returns false if the ballot already has a vote. */
export async function insertVote(db, vote) {
  const [inserted] = await db.batch([
    db.prepare(`INSERT OR IGNORE INTO votes (${VOTE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).bind(
      vote.voter, vote.timestamp, vote.clip_id, vote.left_arm, vote.right_arm, vote.choice, vote.comment,
      vote.watch_seconds, vote.played_seconds, vote.ballot_id),
    db.prepare("UPDATE ballots SET voted = 1 WHERE ballot_id = ?").bind(vote.ballot_id),
  ]);
  return inserted.meta.changes === 1;
}
