# Blind A/B video voting

A small local web app for blinded pairwise preference votes between video
generation variants ("arms": decoders, step counts, token compression,
attention backends, ...), with a results page that puts preference next to
measured speed.

Standard library only (Python 3.10+, `http.server`), vanilla HTML/JS, no build
step. Runs on macOS and Linux.

## Quick start

```bash
python scripts/eval/blind_ab/serve.py --bundle /path/to/bundle [--votes votes.jsonl] [--port 8765]
```

- Vote: <http://localhost:8765/>
- Results: <http://localhost:8765/results>

Options: `--votes` (default `<bundle>/votes.jsonl`; set it when the bundle is
read-only), `--host` (default `127.0.0.1`), `--port` (default `8765`),
`--baseline <arm slug>` (speedup reference; also selectable on the results
page), `--verbose` (log every request).

## Ballot

Each round shows one prompt and two arms side by side. The server picks the
arm pair with the fewest votes so far (all pairs get covered evenly), then the
clip that pair has been compared on least (avoiding repeats for the same
voter), and balances which arm appears on the left. Video URLs are opaque
ballot ids, so arm names are hidden until after the vote.

- Playback is synchronized: play/pause, seek, loop, step with the arrow keys.
- Audio: off / left / right (`m` cycles).
- `1:1 crop` (`z`) shows pixels at native size; panning one view pans both.
- Vote with `1` left better, `2` right better, `3` tie, `4` both bad, and an
  optional comment. `space` plays/pauses, `p` toggles the prompt, `Enter` or
  `n` loads the next pair.
- "Reveal after vote" shows the arm names after each vote (stored per browser).
- The voter name is asked once and kept in the browser's localStorage.

Each vote is appended to the votes JSONL file as one line:

```json
{"voter": "ann", "timestamp": "2026-01-01T12:00:00+00:00", "clip_id": "003_sample-0042",
 "left_arm": "fp8_decoder", "right_arm": "baseline", "choice": "left", "comment": "sharper text",
 "watch_seconds": 21.4, "played_seconds": 18.2, "ballot_id": "..."}
```

`choice` is one of `left`, `right`, `tie`, `both_bad`. `watch_seconds` is the time
from the ballot appearing to the vote; `played_seconds` is time spent playing.
Several voters can vote at once: the single server process appends under a
lock. Restarting the server keeps all votes (the file is re-read on start).

## Results

- Per arm: votes, wins / ties / losses / both bad, win rate with a Wilson 95%
  interval, Bradley-Terry score (Elo-like scale, 1000 = average, +400 = 10x
  odds), seconds per clip, speedup vs the chosen baseline, hardware and
  resolution from `arms.json`.
- A quality-vs-speed scatter (speedup vs Bradley-Terry score).
- A head-to-head win/tie/loss matrix.
- Export: `/api/results.csv`, `/api/results.json` (both accept `?baseline=<slug>`),
  and the raw `/api/votes.jsonl`.

Ties and "both bad" count as half a win for each side in the win rate and the
Bradley-Terry fit. The fit uses a weak prior (one virtual tie against an
average opponent) so undefeated arms stay finite.

## Input layouts

### 1. Manifest layout

```text
bundle/
  arms.json                # {"arms": [{"slug", "display_name", "notes", ...}]}
  manifest.jsonl           # one row per prompt
  arms/<arm>/videos/<index>_<sample_id>.mp4
```

Each `manifest.jsonl` row has `index`, `sample_id`, `prompt` and
`arms: {<slug>: {"path": "arms/<slug>/videos/...mp4", "sha256": ..., ...}}`.
Extra fields are ignored. If an arm's `path` is missing, the app looks for
`arms/<arm>/videos/<index:03d>_<sample_id>.mp4`.

### 2. Simple layout

```text
bundle/
  arms.json
  prompts.json             # optional: {"<clip_id>": "prompt text"}
  arms/<arm>/<clip_id>.mp4 # same clip_id across arms = same prompt + seed
```

`arms.json` uses the same schema plus optional per-arm speed metadata:

```json
{"arms": [
  {"slug": "baseline", "display_name": "Release default", "notes": "50 steps",
   "speed": {"seconds_per_clip": 70.9, "hardware": "1x GPU", "resolution": "832x480"}},
  {"slug": "fp8_decoder", "display_name": "FP8 decoder",
   "speed": {"seconds_per_clip": 52.3, "hardware": "1x GPU", "resolution": "832x480"}}
]}
```

Speed metadata is also read from manifest-layout bundles if present. Clips
with fewer than two arms are skipped.

### Building a simple bundle

`build_bundle.py` assembles the simple layout from per-arm folders of videos,
matching files by name:

```bash
python scripts/eval/blind_ab/build_bundle.py --out /tmp/my_bundle \
    --arm baseline=runs/baseline/videos \
    --arm fp8_decoder=runs/fp8_decoder/videos \
    --speed speed.json [--prompts prompts.json] [--mode symlink|copy|hardlink] [--require-all]
```

`speed.json` maps slug to metadata, e.g.
`{"baseline": {"display_name": "Release default", "seconds_per_clip": 70.9, "hardware": "1x GPU"}}`.
`display_name` and `notes` become arm fields; all other keys go under `speed`.
Videos are symlinked by default (use `--mode copy` to make a portable bundle).
By default a clip is kept when at least two arms have it; `--require-all`
keeps only clips present in every arm.

## Sharing with teammates

On a LAN, bind to all interfaces and share your machine's address:

```bash
python scripts/eval/blind_ab/serve.py --bundle /path/to/bundle --host 0.0.0.0
# teammates open http://<your-ip>:8765/
```

There is no authentication: only do this on a trusted network.

When the bundle lives on a remote machine (for example a cluster login node),
run the server there on localhost and forward the port over SSH:

```bash
# on the remote machine
python scripts/eval/blind_ab/serve.py --bundle /path/to/bundle --votes ~/votes.jsonl --port 8765
# on your laptop
ssh -N -L 8765:localhost:8765 user@remote-host
# then open http://localhost:8765/
```

Each teammate can open their own tunnel to the same server; votes from all of
them land in the same file.

## Serverless deployment (Cloudflare Pages + D1)

`cloudflare/` is a port of the server to Cloudflare Pages Functions with votes
in a D1 database, so a vote keeps running with no machine of yours online. It
fits the free tier. The UI is the same `static/` files, and the API and
exports are the same; the numbers match `stats.py` (see the parity test below).

- Videos are uploaded as static assets under `v/<salted sha256>.mp4`. Browsers
  only ever see `/video/<ballot_id>/<left|right>`. A Function looks the ballot
  up in D1 and streams the hidden asset through `env.ASSETS`, so arm names stay
  hidden until after the vote. Range requests (seeking) work. Pages allows at
  most 25 MiB per file and 20,000 files per deployment.
- Ballots and votes live in D1 (`schema.sql`). Votes are stored once per ballot.

One-time setup (needs Node 18+ and `npx wrangler login`):

```bash
cd scripts/eval/blind_ab/cloudflare
npx wrangler d1 create <db-name>                 # note the database_id it prints
npx wrangler pages project create <project-name> --production-branch main
```

Deploy a bundle (either layout). Pass values as flags, or put them in the
git-ignored `local.json`: `{"bundle", "databaseId", "databaseName",
"projectName", "baseline"}`.

```bash
node build.mjs --bundle /path/to/bundle --database-id <uuid> \
    --database-name <db-name> --project-name <project-name> [--baseline <slug>]
cd dist                                           # git-ignored build output
npx wrangler d1 execute <db-name> --remote --file schema.sql
npx wrangler pages deploy --branch main
```

The site is at `https://<project-name>.pages.dev` (`/` to vote, `/results` for
results). Rebuilding keeps the same video salt (`dist/.video-salt`), so
redeploys only upload changed files. To start a fresh vote on a new bundle,
use a new D1 database or clear the tables.

Import votes from a local `votes.jsonl`. All fields are kept, and importing
the same file twice adds nothing:

```bash
node tools/votes_to_sql.mjs votes.jsonl > /tmp/import.sql
npx wrangler d1 execute <db-name> --remote --file /tmp/import.sql
```

To delete votes (for example a test vote), run `npx wrangler d1 execute <db-name>
--remote --command "DELETE FROM votes WHERE voter = '<name>'"`. To use a custom
domain, add it under the Pages project's *Custom domains*. If the domain's
zone is in the same account, the hostname must be a proxied CNAME to
`<project-name>.pages.dev`.

Local preview: run `npx wrangler d1 execute <db-name> --local --file schema.sql`,
then `npx wrangler pages dev`, both inside `dist/`.

## Tests

```bash
pytest scripts/eval/blind_ab/tests -v
node --test scripts/eval/blind_ab/cloudflare/test/parity.test.mjs
```

CPU only, no GPU or real videos needed (placeholder files are used). The
Node test checks that the Cloudflare port reproduces the Python Wilson
intervals, Bradley-Terry fit, results summary and CSV, pairing decisions, and
vote records. It compares against `cloudflare/test/fixtures.json`. After
changing `stats.py`, `pairing.py` or `votes.py`, regenerate that file with
`python scripts/eval/blind_ab/cloudflare/test/make_fixtures.py`.
