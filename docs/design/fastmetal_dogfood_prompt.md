# FastMetal dogfooding — agent prompt

**We are collecting performance data for the FastMetal Apple Silicon video
models across as many different Macs as possible.** This takes about 20 minutes
of your time, most of it waiting.

## How to use it

1. Open a coding agent (Claude Code, Codex, Cursor) on the Mac you want to
   benchmark, in an empty directory.
2. Copy **everything inside the fenced block below** and paste it in.
3. Replace `<PASTE YOUR TOKEN HERE>` with your Hugging Face token first.
   If a ready-made download command was sent with this doc, paste that in too —
   the agent is told to prefer it.
4. Answer the four questions the agent asks. It will not download or run
   anything before you do.
5. Leave it alone. It runs the benchmark and posts the results to
   **https://github.com/aryan5v/FastVideo/issues/31** itself.

The only thing that may interrupt you is a one-time `gh auth login` browser
prompt, if the GitHub CLI is not already signed in on this machine.

You do not need to know anything about the codebase. The agent sizes the run to
your machine's memory and will tell you straight if your Mac cannot run a given
model.

**Generation runs entirely on your machine.** Nothing is uploaded except the
report the agent posts to the issue, which you approve up front and which
contains your machine profile and timings — no prompts, no video.

---

````markdown
You are running a benchmark of the FastMetal Apple Silicon video-generation
models on this machine. The goal is to collect performance data across a range
of Macs, so accurate measurement matters more than pretty video.

HUGGING_FACE_TOKEN: <PASTE YOUR TOKEN HERE>

## Rules

- **Ask before you run anything.** The four questions in step 3 are mandatory.
  Do not download a model or start a generation before the user has answered.
- Never exceed what the user approves, even if the machine looks capable.
- If a run fails, record the failure and carry on. A crash on a small machine
  is data, not a problem to debug.
- Do not modify the FastVideo source. You are measuring it, not changing it.

## 1. Profile the machine

Collect and print:

```bash
sysctl -n machdep.cpu.brand_string          # e.g. Apple M4 Max
sysctl -n hw.memsize                        # unified memory, bytes
sw_vers -productVersion                     # macOS version
system_profiler SPDisplaysDataType | grep -E "Chipset|Total Number of Cores"
```

Also record whether the machine is on AC power and whether anything else large
is running — both move the numbers.

## 2. Work out which models this machine can take

Peak memory during denoise, measured on an M4 Max:

| Model | Resolution | Peak | Entrypoint |
|---|---|---:|---|
| FastMetal-1.3B-QAD | 832x480x81 | 3.9 GiB | `mlx_wan_prompt_to_video.py` |
| FastMetal-5B-QAD | 1280x704x81 | 9.3 GiB | `mlx_wan22_generate.py` |
| FastMetal-14B-QAD | 832x480x81 | 21.7 GiB | `mlx_wan_prompt_to_video.py` |

Add roughly 2 GiB for the UMT5 text encoder on the first run of a prompt, and
leave macOS 4-6 GiB. Recommend accordingly:

- **8 GB — recommend nothing.** Say so plainly. Even the 1.3B will swap once
  the text encoder loads, which makes the timings meaningless. Offer to record
  the machine profile only, so we know the model was declined here.
- **16 GB — recommend 1.3B. 5B is worth *trying* but may swap.** If the user
  wants 5B, run it last and flag any run where wall-clock is wildly out of line
  with denoise time, which is the signature of swapping.
- **24 GB — recommend 1.3B and 5B.** Do not attempt 14B.
- **36 GB and above — recommend all three,** 1.3B, 5B and 14B.

## 3. Ask the user (mandatory, before any download)

Present your recommendation from step 2, then ask:

1. **Which models are you comfortable running?** Show the recommended set as the
   default and let them cut it down or override it. If they pick something above
   the recommendation for their RAM, warn once, then respect the choice.
2. **How many runs total?** Each run is one model plus one mode combination.
   Give them the cost: at 1.3B a run is roughly 15-120s of compute depending on
   mode; at 14B, 5-10 minutes. Suggest 6-10 runs as a useful sample.
3. **Anything to avoid?** Time limits, thermal concerns, disk space (the three
   checkpoints together are large), or modes they do not care about.
4. **May I post the results to the public issue when I am done?** Say that you
   will post the machine profile and the timing table to
   `https://github.com/aryan5v/FastVideo/issues/31`, that it is public, and that
   it contains no prompts or video unless they ask for those to be included.
   Default to yes. If they decline, run the benchmark anyway and leave the
   report on disk.

Wait for answers. Then confirm the plan back to them before starting.

After this point, run to completion without further questions. The user should
not need to do anything else.

## 4. Set up

**Check out the exact head of PR #1638.** Not `main` — the code being measured
only exists on that PR, and it moves. `pull/1638/head` always resolves to its
current tip:

```bash
brew install ffmpeg
git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo
git fetch origin pull/1638/head:pr1638
git checkout pr1638
git rev-parse HEAD          # RECORD THIS — it goes in the report
```

The head must be at or after `8be306e07`, which is the commit that fixed spatial
fast mode. Anything earlier produces a blurred veil in `--fast-spatial` output
and the numbers will not be comparable. Verify:

```bash
git merge-base --is-ancestor 8be306e07ddac456918d6cf55dfa1ed06143f6ba HEAD \
  && echo "OK: includes the fast-spatial fix" \
  || echo "TOO OLD: re-fetch the PR head"
```

Then:

```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e '.[mlx]'
```

Download only the approved models:

```bash
export HF_TOKEN="<the token above>"
huggingface-cli download FastVideo/FastMetal-1.3B-QAD --local-dir models/fastmetal-1.3b
huggingface-cli download FastVideo/FastMetal-5B-QAD  --local-dir models/fastmetal-5b
huggingface-cli download FastVideo/FastMetal-14B-QAD --local-dir models/fastmetal-14b
```

If a download command is supplied alongside this prompt, prefer it over the
above — the repo names may have moved. If a download 401s or 404s, stop and tell
the user rather than guessing at an alternative name.

## 5. Pick prompts at random

Choose a different prompt per run, at random, from this pool. They are chosen to
exercise motion, texture and lighting rather than to be easy:

1. A red fox trots through a snowy pine forest at golden hour, breath visible in the cold air, cinematic.
2. A potter's hands shape a spinning clay bowl, wet clay glistening under warm studio light, close up.
3. Waves break over black volcanic rock as sea spray catches the late afternoon sun, slow motion.
4. A neon-lit Tokyo alley in the rain, reflections rippling in puddles as a figure walks past with an umbrella.
5. A hot air balloon rises over patchwork farmland at dawn, mist pooling in the valleys below.
6. A chef tosses vegetables in a wok, flames leaping, steam and oil catching the kitchen light.
7. A hummingbird hovers at a bright red flower, wings blurred, garden bokeh behind it.
8. Time-lapse of storm clouds rolling over a desert mesa, lightning flickering on the horizon.
9. A cyclist carves down a wet mountain road through pine trees, spray kicking off the tyres.
10. An astronaut floats in a space station cupola, Earth turning slowly in the window behind them.

## 6. Pick a mode combination at random

Cover all seven over the course of the session — random order, no repeats until
each has been used once. Baseline first so there is something to compare to.

| # | Mode | 1.3B / 14B flags | 5B flags |
|---|---|---|---|
| 1 | baseline | *(none)* | *(none)* |
| 2 | temporal fast | `--fast` | `--fast` |
| 3 | spatial fast | `--fast-spatial` | `--fast-spatial` |
| 4 | refine | `--refine` | `--refine` |
| 5 | refine + fast | `--refine --fast` | `--refine --fast` |
| 6 | spatial + temporal fast | `--fast-spatial --fast` | `--fast-spatial --fast` |
| 7 | prompt improvement | `--enhance-prompt` | `--enhance-prompt` |

## 7. Run

1.3B and 14B:

```bash
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root models/fastmetal-1.3b --mlx-checkpoint models/fastmetal-1.3b \
  --height 480 --width 832 --num-frames 81 --seed 0 \
  --prompt "<chosen prompt>" <mode flags> \
  --output-path runs/<run-id>.mp4 --metrics-json runs/<run-id>.json
```

5B (different entrypoint, different resolution, and note `--compile` rather
than `--mlx-compile`):

```bash
python examples/inference/basic/mlx_wan22_generate.py \
  --mlx-checkpoint models/fastmetal-5b --text-encoder-root models/fastmetal-5b \
  --vae-root models/fastmetal-5b/vae \
  --height 704 --width 1280 --num-frames 81 --fps 16 --seed 0 \
  --prompt "<chosen prompt>" <mode flags> \
  --output-path runs/<run-id>.mp4 --metrics-json runs/<run-id>.json
```

Both entrypoints write the numbers we want to `--metrics-json` — use it rather
than scraping stdout. Between runs, let the machine settle for ~30s so thermal
throttling does not bleed from one run into the next.

Note that the first run for a given prompt pays a cold UMT5 encode (18s at
1.3B/14B, 47s at 5B); repeats hit the cache. Say which runs were cold.

## 8. Report

Write `runs/REPORT.md` containing:

- the machine profile from step 1
- **the commit SHA from step 4** — without it the numbers cannot be compared
- one row per run: model, mode, prompt index, denoise seconds, total seconds,
  peak GiB, cold or warm prompt, pass or fail
- for each model, the speedup of each mode over its own baseline
- anything anomalous: swapping, thermal throttling, failures, visible quality
  problems in the output video

Then, in two or three sentences, say what this machine is comfortably good for.

## 9. Post the results yourself

If the user approved posting in step 3, **you post it.** Do not ask them to do
it by hand.

Check authentication first:

```bash
gh auth status
```

If `gh` is not installed, install it (`brew install gh`). If it is not
authenticated, run `gh auth login` and walk the user through the browser prompt
— that is the one point where they must touch the keyboard.

Then post:

```bash
gh issue comment 31 --repo aryan5v/FastVideo --body-file runs/REPORT.md
```

Print the URL of the comment you created so the user can see it landed.

Post even when the results are bad. A machine that swapped, throttled, or
refused to run a model is a real data point, and an 8 GB Mac that ran nothing at
all should still post its profile.

If posting fails, say so plainly, print the path to `runs/REPORT.md`, and give
the user the issue URL so they can paste it themselves.

**Videos.** The GitHub API cannot attach video to a comment, so you cannot
upload them. If any output looks wrong — blur, noise, colour artefacts, motion
breaking down — say so in the comment, name the model and mode, and print the
file path. Tell the user they can drag that `.mp4` into the issue in a browser
if they want to share it. Treat this as optional.
````
