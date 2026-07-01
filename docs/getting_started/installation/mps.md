# MPS (Apple Silicon)

Instructions to install FastVideo for Apple Silicon.

## Requirements

- **OS: MacOS**
- **Python: 3.12.4**

## Set up using Python

### Create a new Python environment

#### uv
Recommended default: use [uv](https://docs.astral.sh/uv/) for faster and more stable environment setup.

Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, create a new environment using:

```console
# (Recommended) Create a new uv environment. Use `--seed` to install `pip` and `setuptools`.
uv venv --python 3.12 --seed
source .venv/bin/activate
```

#### Conda (alternative)

You can also create a Python environment using [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

##### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
source ~/.zshrc
```

##### 2. Create and activate a Conda environment for FastVideo

```bash
conda create -n fastvideo python=3.12.4 -y
conda activate fastvideo
```

### Dependencies

```
brew install ffmpeg
```

### Installation

#### With uv (recommended)

```bash
uv pip install fastvideo
```

#### With Conda environment (alternative)

`uv` works inside an active conda env too, so prefer `uv pip` for the actual install:

```bash
uv pip install fastvideo
```

### Installation from Source

#### 1. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 2. Install FastVideo

Basic installation:

```bash
uv pip install -e .
```

Alternative with Conda environment:

```bash
uv pip install -e .
```

#### 3. (Optional) Experimental MLX-native runtime

For the experimental Apple-native FastWan path (`fastvideo/mlx_runtime`), install the
`mlx` extra. MLX runs the DiT denoising loop directly on Metal via
[MLX](https://github.com/ml-explore/mlx); the text encoder and VAE decode currently run
on the PyTorch MPS backend.

```bash
uv pip install -e '.[mlx]'
```

The `mlx` dependency is guarded by an Apple-Silicon environment marker, so the command is
a no-op for the MLX package on non-`arm64` machines. On macOS, the `torch` wheel resolved
by FastVideo already ships the MPS backend on Apple Silicon — no extra index is required.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](../../contributing/overview.md)

## Hardware Requirements

### For Basic Inference

- Mac M1, M2, M3, or M4.
- **16 GB unified memory** is the baseline target for local generation (e.g. FastWan
  T2V-1.3B at 480p / 5s). To fit 16 GB, encode the prompt and free the text encoder
  before loading the DiT (the MLX example supports a subprocess encode-then-free mode).
- **32 GB+** gives more headroom for higher resolution, longer clips, and running the
  full-precision Wan VAE decoder; Mac Studio-class machines (64 GB+) can push to 720p and
  longer generations.

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ) for additional support.
