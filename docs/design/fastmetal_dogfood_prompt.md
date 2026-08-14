# FastMetal dogfooding — agent prompt

Collecting FastMetal performance data across our Macs. Takes about 20 minutes,
mostly waiting.

Open a coding agent on the Mac you want to benchmark, in an empty directory.
Paste everything in the fenced block below, with your Hugging Face token filled
in. Answer three questions, then leave it — it runs the benchmark and posts the
results itself.

---

````markdown
Benchmark the FastMetal Apple Silicon video models on this machine. Accurate
measurement matters more than pretty video.

HUGGING_FACE_TOKEN: <PASTE YOUR TOKEN HERE>

If a run fails, record it and carry on — a crash on a small machine is data.
Do not modify the FastVideo source; you are measuring it.

## 1. Profile the machine

```bash
sysctl -n machdep.cpu.brand_string          # e.g. Apple M4 Max
sysctl -n hw.memsize                        # unified memory, bytes
sw_vers -productVersion
system_profiler SPDisplaysDataType | grep -E "Chipset|Total Number of Cores"
```

Note whether it is on AC power and whether anything else heavy is running.

## 2. Decide what this machine can take

Peak memory during denoise, measured on an M4 Max:

| Model | Resolution | Peak | Entrypoint |
|---|---|---:|---|
| FastMetal-1.3B-QAD | 832x480x81 | 3.9 GiB | `mlx_wan_prompt_to_video.py` |
| FastMetal-5B-QAD | 1280x704x81 | 9.3 GiB | `mlx_wan22_generate.py` |
| FastMetal-14B-QAD | 832x480x81 | 21.7 GiB | `mlx_wan_prompt_to_video.py` |

Add ~2 GiB for the UMT5 text encoder on a cold prompt, and leave macOS 4-6 GiB:

- **8 GB** — recommend nothing. Record the profile so we know it was declined.
- **16 GB** — recommend 1.3B. 5B is worth trying but may swap; run it last.
- **24 GB** — recommend 1.3B and 5B. Skip 14B.
- **36 GB+** — recommend all three.

Flag any run whose wall-clock is far above its denoise time: that is swapping.

## 3. Ask, then go

Show your recommendation from step 2 and ask:

1. **Which models do you want to run?** Recommended set is the default.
2. **How many runs?** One run is one model plus one mode. At 1.3B a run is
   15-120s depending on mode; at 14B, 5-10 minutes. Suggest 6-10.
3. **Anything to avoid?** Time, thermals, disk space, modes you do not care about.

Then run to completion without further questions.

## 4. Set up

Check out the exact head of PR #1638 — not `main`. The code only exists there.

```bash
brew install ffmpeg
git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo
git fetch origin pull/1638/head:pr1638
git checkout pr1638
git rev-parse HEAD          # record this, it goes in the report
```

The head must include `8be306e07`, which fixed spatial fast mode. Earlier
commits produce a blurred veil in `--fast-spatial` and the numbers will not be
comparable:

```bash
git merge-base --is-ancestor 8be306e07ddac456918d6cf55dfa1ed06143f6ba HEAD \
  && echo "OK" || echo "TOO OLD: re-fetch the PR head"
```

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

Prefer any download command supplied alongside this prompt. If one 404s, stop
and say so rather than guessing at another name.

## 5. Pick a prompt at random

A different one per run, chosen to exercise motion, texture and lighting:

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

## 6. Pick a mode at random

Cover all seven across the session, random order, no repeats until each is used.
Baseline first so the rest have something to compare against.

| Mode | Flags (same on both entrypoints) |
|---|---|
| baseline | *(none)* |
| temporal fast | `--fast` |
| spatial fast | `--fast-spatial` |
| refine | `--refine` |
| refine + fast | `--refine --fast` |
| spatial + temporal fast | `--fast-spatial --fast` |
| prompt improvement | `--enhance-prompt` |

## 7. Run

1.3B and 14B:

```bash
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root models/fastmetal-1.3b --mlx-checkpoint models/fastmetal-1.3b \
  --height 480 --width 832 --num-frames 81 --seed 0 \
  --prompt "<chosen prompt>" <mode flags> \
  --output-path runs/<run-id>.mp4 --metrics-json runs/<run-id>.json
```

5B — different entrypoint, different resolution, and `--compile` rather than
`--mlx-compile`:

```bash
python examples/inference/basic/mlx_wan22_generate.py \
  --mlx-checkpoint models/fastmetal-5b --text-encoder-root models/fastmetal-5b \
  --vae-root models/fastmetal-5b/vae \
  --height 704 --width 1280 --num-frames 81 --fps 16 --seed 0 \
  --prompt "<chosen prompt>" <mode flags> \
  --output-path runs/<run-id>.mp4 --metrics-json runs/<run-id>.json
```

Take the numbers from `--metrics-json`, not from stdout. Leave ~30s between runs
so throttling does not bleed across them. The first run of a given prompt pays a
cold UMT5 encode (18s at 1.3B/14B, 47s at 5B); say which runs were cold.

## 8. Report and post

Write `runs/REPORT.md`:

- machine profile from step 1
- the commit SHA from step 4
- one row per run: model, mode, prompt #, denoise s, total s, peak GiB,
  cold/warm, pass/fail
- per model, each mode's speedup over its own baseline
- anything odd: swapping, throttling, failures, visible quality problems
- two or three sentences on what this machine is comfortably good for

Then post it:

```bash
gh auth status                      # brew install gh / gh auth login if needed
gh issue comment 31 --repo aryan5v/FastVideo --body-file runs/REPORT.md
```

Print the comment URL. Post even when the results are bad — a machine that
swapped or refused to run a model is a real data point.

The API cannot attach video to a comment. If any output looks wrong, describe it
in the report with the model and mode, and print the file path so it can be
dragged into the issue by hand.
````
