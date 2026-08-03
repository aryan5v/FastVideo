# Agent Prompt — M5 Quantization Survey

Copy everything below the line and hand it to an agent running on the M5
MacBook. It is written for a machine with **no development setup at all** and
leaves nothing behind outside one folder.

## Why these tests

We need to choose a 4-bit quantization format for on-device video generation and
we want the answer without spending GPU-cluster time. The decisive measurement
is **weight reconstruction error per format on real checkpoints** — it ranks
formats in minutes, needs no training and no video generation, and predicts
output quality directly. Everything else is supporting evidence.

The single most valuable comparison: `FastWan-QAD-1.3B` was
quantization-aware-trained against NVFP4 on NVIDIA hardware, and its checkpoint
is stored in bf16 (quantization happens at load). If its NVFP4 reconstruction
error is markedly lower than a non-QAT'd checkpoint's, the training transferred
to Apple's implementation and we can ship it on Mac with no retraining.

---

You are running on a borrowed Apple M5 MacBook with 24 GB of unified memory and
no development tooling installed. Your job is to run a quantization survey and
report numbers back.

## Hard rules

1. **Everything lives in `~/fastvideo-m5-survey/`.** Every tool, cache, model
   download, virtual environment, and output file. Nothing outside that folder
   may be created or modified.
2. **Do not use `sudo`.** Do not install Homebrew. Do not install anything
   system-wide. Do not modify shell profiles (`.zshrc`, `.bash_profile`, etc.).
3. **Do not run `xcode-select --install`** — it opens a GUI dialog on someone
   else's machine. If something genuinely requires it, stop and report that
   instead.
4. When finished, the machine must be restorable with exactly one command:
   `rm -rf ~/fastvideo-m5-survey`. Verify this is true before you finish.
5. Disk: the downloads below total roughly 6 GB. Check free space first with
   `df -h ~` and stop if there is less than 20 GB available.

## Step 1 — Set up an isolated environment

```bash
mkdir -p ~/fastvideo-m5-survey && cd ~/fastvideo-m5-survey

# uv installs into our folder, not the system, and brings its own Python.
export UV_INSTALL_DIR="$HOME/fastvideo-m5-survey/uv"
export UV_CACHE_DIR="$HOME/fastvideo-m5-survey/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/fastvideo-m5-survey/python"
# Critical: keep model downloads inside our folder instead of ~/.cache.
export HF_HOME="$HOME/fastvideo-m5-survey/cache/hf"

curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh
export PATH="$UV_INSTALL_DIR:$PATH"

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install mlx numpy safetensors huggingface_hub
```

Re-export those five environment variables in every new shell you open. If any
model download or cache lands outside `~/fastvideo-m5-survey`, that is a bug —
find it and fix it before continuing.

## Step 2 — Record the machine, and check two gates

```bash
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB unified memory"}'
sw_vers
python -c "import mlx.core as mx; print('mlx', mx.__version__)"
```

Save this output, and check both gates before going further:

**Gate 1 — the chip must actually be M5.** The whole question is about M5
neural accelerators. An M4 invalidates every throughput number.

**Gate 2 — macOS must be 26.2 or later.** MLX only uses the M5 Neural
Accelerators on macOS 26.2+. On an older build the matmuls silently fall back
to the normal GPU path and every throughput reading is meaningless — the
reconstruction-error numbers stay valid, but the speed comparison does not.

If either gate fails, **report it immediately and ask before continuing.** Do
not upgrade the operating system on a borrowed machine under any circumstances.

Also confirm you are on a recent MLX. Releases ship every three to four weeks
and MX/NVFP4 support has been landing progressively, so an old build may report
modes as unsupported that newer ones handle:

```bash
uv pip install --upgrade mlx
python -c "import mlx.core as mx; print('mlx', mx.__version__)"
```

Record the final version — results are only interpretable against it.

## Step 3 — Get the survey script

The script is `fastvideo/benchmarks/mlx_quant_survey.py` in the FastVideo repo,
on branch `claude/minimax-h3-fastvideo-support-t02662`.

```bash
cd ~/fastvideo-m5-survey
git clone --depth 1 --branch claude/minimax-h3-fastvideo-support-t02662 \
    https://github.com/aryan5v/FastVideo.git repo
```

If `git` is unavailable or the clone fails for any reason, **stop and report
that** — the person who gave you this task will send the single file directly.
Do not try to reconstruct the script yourself; the exact quantization calls are
the point of the test.

## Step 4 — Download two checkpoints (transformer weights only)

We only need the transformer subfolder from each. Do **not** download full
pipelines — the text encoders are large and we do not need them.

```bash
cd ~/fastvideo-m5-survey

python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["FastVideo/FastWan2.1-T2V-1.3B-Diffusers"]:
    p = snapshot_download(repo, allow_patterns=["transformer/*"])
    print(repo, "->", p)
PY
```

Then find and download the **QAD** checkpoint. Its exact repo ID is uncertain —
search the `FastVideo` org on Hugging Face for a Wan 2.1 1.3B model whose card
mentions **QAD**, **quantization-aware distillation**, or **NVFP4**:

```bash
python - <<'PY'
from huggingface_hub import HfApi
for m in HfApi().list_models(author="FastVideo", search="Wan"):
    print(m.id)
PY
```

Download the transformer folder of whichever repo matches, the same way.
**Report the exact repo ID you used** — this matters for interpreting results.
If no QAD checkpoint is findable, say so and continue with the other tests.

## Step 5 — Run the survey

```bash
cd ~/fastvideo-m5-survey
mkdir -p results
export PYTHONPATH="$HOME/fastvideo-m5-survey/repo"

# A. Plain distilled checkpoint — the baseline.
python -m fastvideo.benchmarks.mlx_quant_survey \
    --checkpoint <path-to-plain-transformer-dir> \
    --modes int8 int6 int5 int4 mxfp8 mxfp4 nvfp4 \
    --json-out results/survey_plain.json 2>&1 | tee results/survey_plain.txt

# B. QAD checkpoint — the one that may already carry NVFP4 training.
python -m fastvideo.benchmarks.mlx_quant_survey \
    --checkpoint <path-to-qad-transformer-dir> \
    --modes int8 int6 int5 int4 mxfp8 mxfp4 nvfp4 \
    --json-out results/survey_qad.json 2>&1 | tee results/survey_qad.txt
```

The `snapshot_download` calls print the local paths; the transformer directory
is `<that path>/transformer`.

If a mode is reported unsupported, that is a valid result — record it and move
on. Do not try to work around it.

## Step 6 — Throughput at a second shape

The default throughput probe uses one shape. Run two more so we can tell whether
any speedup is real or shape-specific:

```bash
for dim in 1536 5120; do
  python -m fastvideo.benchmarks.mlx_quant_survey \
      --modes int8 int6 mxfp4 nvfp4 --dim $dim --tokens 8192 --iters 30 \
      --json-out results/tput_${dim}.json 2>&1 | tee results/tput_${dim}.txt
done
```

No `--checkpoint` here, so it skips the error survey and only benchmarks.

## Step 7 — Report

Report these, and attach every file in `results/`:

1. Machine info from Step 2, **including whether the chip is genuinely M5**.
2. The exact Hugging Face repo IDs you downloaded.
3. **Which quantization modes the installed MLX supports**, and the MLX version.
4. **The reconstruction-error table for both checkpoints.** This is the key
   result. We specifically want: is `nvfp4` error on the QAD checkpoint lower
   than `nvfp4` error on the plain checkpoint?
5. **The throughput table at both shapes** — particularly whether `mxfp4` and
   `nvfp4` are meaningfully faster than `int8`, or roughly the same.
6. Anything that failed, with the exact error text.

Do not interpret the results or recommend a decision. Report the numbers.

## Step 8 — Clean up

```bash
du -sh ~/fastvideo-m5-survey            # note the size before deleting
ls -la ~/.cache/huggingface 2>/dev/null # must NOT exist, or must predate today
```

Copy `results/` somewhere it will survive (send it back), then:

```bash
rm -rf ~/fastvideo-m5-survey
```

Confirm nothing else was touched: no Homebrew, no system Python packages, no
shell profile edits, no files in `~/Documents`, `~/Downloads`, or `~/.cache`.
State explicitly in your report that cleanup is complete and what you verified.
