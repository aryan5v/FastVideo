// Minimal media probing used at build time (no ffprobe dependency).

import fs from "node:fs";

const HDLR = Buffer.from("hdlr");
const SOUN = Buffer.from("soun");
const HANDLER_OFFSET = 8; // 'hdlr' + version/flags (4) + pre_defined (4), then handler_type

/**
 * True if an MP4/MOV file has an audio track (a 'hdlr' box with handler type 'soun'), false if it has
 * none, null for formats we cannot inspect (e.g. WebM).
 */
export function hasAudioTrack(file) {
  if (!/\.(mp4|m4v|mov)$/i.test(file)) return null;
  const data = fs.readFileSync(file);
  for (let i = data.indexOf(HDLR); i !== -1; i = data.indexOf(HDLR, i + 1)) {
    const start = i + HDLR.length + HANDLER_OFFSET;
    if (data.subarray(start, start + SOUN.length).equals(SOUN)) return true;
  }
  return false;
}
