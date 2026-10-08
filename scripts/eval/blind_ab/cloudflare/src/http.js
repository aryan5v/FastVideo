// Small HTTP helpers shared by the Pages Functions.

export const JSON_TYPE = "application/json; charset=utf-8";

export function sendBytes(status, body, contentType, extra = {}) {
  return new Response(body, {
    status,
    headers: { "Content-Type": contentType, "Cache-Control": "no-store", ...extra },
  });
}

export function sendJson(status, payload, extra = {}) {
  return sendBytes(status, JSON.stringify(payload, null, 1), JSON_TYPE, extra);
}

export function sendError(status, message) {
  return sendJson(status, { ok: false, error: message });
}

const RANGE_RE = /^bytes=(\d*)-(\d*)$/;

/**
 * Port of serve.parse_range: inclusive [start, end] for a Range header, null for the whole file.
 * Throws RangeError for malformed or unsatisfiable ranges.
 */
export function parseRange(header, size) {
  if (!header) return null;
  const match = RANGE_RE.exec(header.trim());
  if (!match || (!match[1] && !match[2])) throw new RangeError("malformed range");
  const [, first, last] = match;
  if (!first) {
    const length = Number(last);
    if (length === 0) throw new RangeError("empty suffix range");
    return [Math.max(0, size - length), size - 1];
  }
  const start = Number(first);
  const end = last ? Math.min(Number(last), size - 1) : size - 1;
  if (start >= size || end < start) throw new RangeError("range not satisfiable");
  return [start, end];
}
