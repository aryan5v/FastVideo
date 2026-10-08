#!/usr/bin/env node
// Convert a votes JSONL file (written by serve.py) into SQL for `wrangler d1 execute --file`.
//
//   node tools/votes_to_sql.mjs votes.jsonl > /tmp/import.sql
//   npx wrangler d1 execute <db> --remote --file /tmp/import.sql
//
// Every field is kept: the ten standard fields go into columns, anything else into the `extra` JSON column.
// Rows use INSERT OR IGNORE keyed on ballot_id, so re-importing the same file adds nothing.

import fs from "node:fs";

import { VOTE_FIELDS, voteFromRecord } from "../src/votes.js";

const COLUMNS = [...VOTE_FIELDS, "extra"];
const sqlString = (text) => `'${String(text).replaceAll("'", "''")}'`;

function sqlValue(key, value) {
  if (value === null || value === undefined) return "NULL";
  if (key === "watch_seconds" || key === "played_seconds") {
    if (!Number.isFinite(value)) throw new Error(`${key} must be finite, got ${value}`);
    return String(value);
  }
  return sqlString(value);
}

export function recordToInsert(record) {
  const vote = voteFromRecord(record);
  const extraEntries = Object.entries(record).filter(([k]) => !VOTE_FIELDS.includes(k));
  const row = { ...vote, extra: extraEntries.length ? JSON.stringify(Object.fromEntries(extraEntries)) : null };
  const values = COLUMNS.map((k) => sqlValue(k, row[k]));
  return `INSERT OR IGNORE INTO votes (${COLUMNS.join(", ")}) VALUES (${values.join(", ")});`;
}

function main(argv) {
  if (argv.length !== 1) {
    console.error("usage: votes_to_sql.mjs <votes.jsonl>");
    process.exit(2);
  }
  const lines = fs.readFileSync(argv[0], "utf8").split("\n");
  const out = [];
  lines.forEach((line, i) => {
    if (!line.trim()) return;
    try {
      out.push(recordToInsert(JSON.parse(line)));
    } catch (err) {
      console.error(`${argv[0]}:${i + 1}: skipped: ${err.message}`);
    }
  });
  process.stdout.write(`${out.join("\n")}\n`);
  console.error(`${out.length} votes converted`);
}

if (import.meta.url === `file://${process.argv[1]}`) main(process.argv.slice(2));
