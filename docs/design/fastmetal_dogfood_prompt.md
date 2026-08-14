# FastMetal dogfooding — agent prompt

Paste everything in the fenced block below into a coding agent (Claude Code,
Codex, Cursor) running on the Mac you want to benchmark. It will size the run to
the machine, ask you what you are willing to run, then collect timing and memory
data for us.

Fill in your Hugging Face token first. Everything else the agent works out.

---

````markdown
You are running a benchmark of the FastMetal Apple Silicon video-generation
models on this machine. The goal is to collect performance data across a range
of Macs, so accurate measurement matters more than pretty video.

HUGGING_FACE_TOKEN: <PASTE YOUR TOKEN HERE>

## Rules

- **Ask before you run anything.** The three questions in step 3 are mandatory.
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

Wait for answers. Then confirm the plan back to them before starting.

## 4. Set up

```bash
brew install ffmpeg
git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo
git fetch origin pull/1638/head:pr1638 && git checkout pr1638
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e '.[mlx]'
```

Download only the approved models:

```bash
export HF_TOKEN="<the token above>"
huggingface-cli download aryan5v/FastMetal-1.3B-QAD --local-dir models/fastmetal-1.3b
huggingface-cli download aryan5v/FastMetal-5B-QAD  --local-dir models/fastmetal-5b
huggingface-cli download aryan5v/FastMetal-14B-QAD --local-dir models/fastmetal-14b
```

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
- one row per run: model, mode, prompt index, denoise seconds, total seconds,
  peak GiB, cold or warm prompt, pass or fail
- for each model, the speedup of each mode over its own baseline
- anything anomalous: swapping, thermal throttling, failures, visible quality
  problems in the output video

Then, in two or three sentences, say what this machine is comfortably good for.

Finally, tell the user where the videos and `REPORT.md` are, and ask them to
send the report and the machine profile back to the team.
````
