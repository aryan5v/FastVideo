// Request handlers: the Pages Functions port of serve.py + app_state.py.
//
// `bundle` is the generated bundle description (see build.mjs):
//   {name, layout, baseline, arms: [{slug, display_name, notes, speed}], clips: [{clip_id, prompt, videos}]}
// where clips[].videos maps arm slug -> unguessable static asset path (e.g. "/v/<salted hash>.mp4") and
// clips[].audio maps arm slug -> whether that video has an audio track (true/false/null = unknown), and
// clips[].has_audio summarizes it per clip.

import { allVotes, countVotes, getBallot, insertBallot, insertVote, matchupHistory } from "./db.js";
import { parseRange, sendBytes, sendError, sendJson } from "./http.js";
import { chooseMatchup } from "./pairing.js";
import { summarize, summaryToCsv } from "./stats.js";
import { cleanVoteFields, cleanVoter, utcNow, VoteError, voteToJson } from "./votes.js";

export const MAX_BODY_BYTES = 16 * 1024;
const BALLOT_ID_BYTES = 12;
const VIDEO_ROUTE_RE = /^\/video\/([A-Za-z0-9_-]{1,64})\/(left|right)$/;
const VIDEO_CACHE = "private, max-age=3600";

class BallotError extends Error {}

function newBallotId() {
  // Same shape as secrets.token_urlsafe(12): 16 base64url characters.
  const bytes = crypto.getRandomValues(new Uint8Array(BALLOT_ID_BYTES));
  return btoa(String.fromCharCode(...bytes)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function bundleIndex(bundle) {
  const clips = new Map(bundle.clips.map((c) => [c.clip_id, c]));
  const availability = new Map(bundle.clips.map((c) => [c.clip_id, new Set(Object.keys(c.videos))]));
  const arms = new Map(bundle.arms.map((a) => [a.slug, a]));
  const groups = new Map(bundle.clips.filter((c) => c.group).map((c) => [c.clip_id, c.group]));
  const sides = (bundle.site && bundle.site.sides) || "balanced";
  return { clips, availability, arms, groups, sides, slugs: bundle.arms.map((a) => a.slug) };
}

const indexCache = new WeakMap();
function indexOf(bundle) {
  if (!indexCache.has(bundle)) indexCache.set(bundle, bundleIndex(bundle));
  return indexCache.get(bundle);
}

async function results(env, bundle, baseline) {
  const summary = summarize(bundle.arms, await allVotes(env.DB), baseline || bundle.baseline || null);
  return { ...summary, bundle: bundle.name, clips: bundle.clips.length };
}

async function newBallot(env, bundle, rawVoter) {
  const voter = cleanVoter(rawVoter);
  const idx = indexOf(bundle);
  const now = Date.now() / 1000;
  const history = await matchupHistory(env.DB, now);
  const matchup = chooseMatchup(idx.availability, idx.slugs, history, voter, undefined,
    { groups: idx.groups, sides: idx.sides });
  const ballotId = newBallotId();
  await insertBallot(env.DB, { ballot_id: ballotId, ...matchup, voter }, now);
  const clip = idx.clips.get(matchup.clip_id);
  return {
    ballot_id: ballotId,
    clip_id: clip.clip_id,
    prompt: clip.prompt,
    guidance: clip.guidance || "",
    // Whether the clip has an audio track; null means unknown (the page then probes the media).
    has_audio: clip.has_audio ?? null,
    left_url: `/video/${ballotId}/left`,
    right_url: `/video/${ballotId}/right`,
    voter_votes: await countVotes(env.DB, voter),
  };
}

async function recordVote(env, bundle, payload) {
  const fields = cleanVoteFields(payload);
  const ballotId = payload.ballot_id ? String(payload.ballot_id) : "";
  const ballot = ballotId ? await getBallot(env.DB, ballotId) : null;
  if (!ballot) throw new BallotError("unknown or expired ballot; load a new one");
  if (ballot.voted) throw new BallotError("this ballot was already voted on");
  const vote = {
    ...fields,
    timestamp: utcNow(),
    clip_id: ballot.clip_id,
    left_arm: ballot.left_arm,
    right_arm: ballot.right_arm,
    ballot_id: ballotId,
  };
  const meta = indexOf(bundle).clips.get(ballot.clip_id)?.meta || {};
  if (!(await insertVote(env.DB, vote, meta))) throw new BallotError("this ballot was already voted on");
  const arms = indexOf(bundle).arms;
  const reveal = (slug) => ({ slug, display_name: arms.has(slug) ? arms.get(slug).display_name : slug });
  return {
    ok: true,
    reveal: { left: reveal(vote.left_arm), right: reveal(vote.right_arm) },
    voter_votes: await countVotes(env.DB, fields.voter),
  };
}

async function readJsonBody(request) {
  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > MAX_BODY_BYTES) throw new VoteError("missing or oversized request body");
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.length === 0 || bytes.length > MAX_BODY_BYTES) throw new VoteError("missing or oversized request body");
  let payload;
  try {
    payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (err) {
    throw new VoteError(`invalid JSON: ${err.message}`);
  }
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new VoteError("expected a JSON object");
  }
  return payload;
}

async function routeApiGet(env, bundle, path, query) {
  const attach = (name) => ({ "Content-Disposition": `attachment; filename="${name}"` });
  switch (path) {
    case "/api/info":
      return sendJson(200, {
        bundle: bundle.name, layout: bundle.layout, arms: bundle.arms.length, clips: bundle.clips.length,
        has_audio: bundle.clips.some((c) => c.has_audio === true),
        votes: await countVotes(env.DB),
        site: bundle.site || {},
      });
    case "/api/ballot":
      return sendJson(200, await newBallot(env, bundle, query.get("voter")));
    case "/api/results":
      return sendJson(200, await results(env, bundle, query.get("baseline")));
    case "/api/results.json":
      return sendJson(200, await results(env, bundle, query.get("baseline")), attach("blind_ab_results.json"));
    case "/api/results.csv":
      return sendBytes(200, summaryToCsv(await results(env, bundle, query.get("baseline"))),
        "text/csv; charset=utf-8", attach("blind_ab_results.csv"));
    case "/api/votes.jsonl": {
      const body = (await allVotes(env.DB)).map((v) => `${voteToJson(v)}\n`).join("");
      return sendBytes(200, body, "application/x-ndjson; charset=utf-8", attach("votes.jsonl"));
    }
    default:
      return sendError(404, "not found");
  }
}

/** Entry point for /api/*. */
export async function handleApi(request, env, bundle) {
  const url = new URL(request.url);
  try {
    if (request.method === "POST") {
      if (url.pathname !== "/api/vote") return sendError(404, "not found");
      return sendJson(200, await recordVote(env, bundle, await readJsonBody(request)));
    }
    if (request.method !== "GET" && request.method !== "HEAD") return sendError(405, "method not allowed");
    return await routeApiGet(env, bundle, url.pathname, url.searchParams);
  } catch (err) {
    if (err instanceof VoteError) return sendError(400, err.message);
    if (err instanceof BallotError) return sendError(410, err.message);
    console.error(`${request.method} ${url.pathname} failed`, err);
    return sendError(500, "internal error; see server log");
  }
}

function videoHeaders(source, extra) {
  return {
    "Content-Type": source.headers.get("Content-Type") || "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": VIDEO_CACHE,
    ...extra,
  };
}

async function sliceWholeAsset(asset, rangeHeader) {
  // Fallback when the asset server ignored the Range header: slice the bytes ourselves.
  const body = new Uint8Array(await asset.arrayBuffer());
  const size = body.length;
  let range;
  try {
    range = parseRange(rangeHeader, size);
  } catch {
    return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}`, "Content-Length": "0" } });
  }
  if (!range) return new Response(body, { status: 200, headers: videoHeaders(asset, { "Content-Length": `${size}` }) });
  const [start, end] = range;
  return new Response(body.slice(start, end + 1), {
    status: 206,
    headers: videoHeaders(asset, { "Content-Length": `${end - start + 1}`, "Content-Range": `bytes ${start}-${end}/${size}` }),
  });
}

/** Entry point for /video/<ballot_id>/<side>: resolve the ballot and stream the hidden static asset. */
export async function handleVideo(request, env, bundle) {
  const url = new URL(request.url);
  const match = VIDEO_ROUTE_RE.exec(url.pathname);
  if (!match) return sendError(404, "not found");
  if (request.method !== "GET" && request.method !== "HEAD") return sendError(405, "method not allowed");
  try {
    const ballot = await getBallot(env.DB, match[1]);
    const clip = ballot ? indexOf(bundle).clips.get(ballot.clip_id) : null;
    const assetPath = clip ? clip.videos[match[2] === "left" ? ballot.left_arm : ballot.right_arm] : null;
    if (!assetPath) return sendError(404, "unknown ballot");
    const rangeHeader = request.headers.get("Range");
    const headers = rangeHeader ? { Range: rangeHeader } : {};
    const asset = await env.ASSETS.fetch(new Request(new URL(assetPath, url.origin), { method: request.method, headers }));
    if (asset.status === 200 && rangeHeader) return sliceWholeAsset(asset, rangeHeader);
    if (asset.status !== 200 && asset.status !== 206 && asset.status !== 416) {
      console.error("asset fetch failed", asset.status, assetPath);
      return sendError(502, "video unavailable");
    }
    const keep = {};
    for (const name of ["Content-Length", "Content-Range"]) {
      if (asset.headers.has(name)) keep[name] = asset.headers.get(name);
    }
    // Drop ETag / Last-Modified so identical files cannot be correlated across ballots.
    return new Response(asset.body, { status: asset.status, headers: videoHeaders(asset, keep) });
  } catch (err) {
    console.error(`video ${url.pathname} failed`, err);
    return sendError(500, "internal error; see server log");
  }
}
