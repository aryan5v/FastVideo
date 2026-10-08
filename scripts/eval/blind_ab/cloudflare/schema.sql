-- D1 schema for the blind A/B voting app. Safe to re-apply.

-- One row per vote; columns mirror the JSONL record written by votes.py.
CREATE TABLE IF NOT EXISTS votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  voter TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  clip_id TEXT NOT NULL,
  left_arm TEXT NOT NULL,
  right_arm TEXT NOT NULL,
  choice TEXT NOT NULL CHECK (choice IN ('left', 'right', 'tie', 'both_bad')),
  comment TEXT NOT NULL DEFAULT '',
  watch_seconds REAL NOT NULL DEFAULT 0,
  played_seconds REAL NOT NULL DEFAULT 0,
  ballot_id TEXT NOT NULL DEFAULT '',
  extra TEXT -- JSON object with any additional fields from imported records
);
-- A ballot can be voted on once; this also makes re-importing a JSONL file idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS votes_ballot_id ON votes (ballot_id) WHERE ballot_id <> '';
CREATE INDEX IF NOT EXISTS votes_voter ON votes (voter);

-- Issued blinded ballots: the opaque id maps to the clip and the arms on each side.
CREATE TABLE IF NOT EXISTS ballots (
  ballot_id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL,
  left_arm TEXT NOT NULL,
  right_arm TEXT NOT NULL,
  voter TEXT NOT NULL,
  issued_at REAL NOT NULL,
  voted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ballots_issued_at ON ballots (issued_at);
