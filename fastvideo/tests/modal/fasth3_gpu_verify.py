"""FastH3 GPU verification entrypoints on Modal (hao-ai-lab workspace).

Phase-0 verification per docs/design/fasth3_roadmap.md: prove the upstream
MiniMax H3 stack works end to end on GPU, reproduce the bitwise golden-gate
environment, and mint the teacher reference artifacts every later gate is
scored against.

Run from the fork (never upstream):

    modal run fastvideo/tests/modal/fasth3_gpu_verify.py \
        --commit <fasth3-mlx-runtime commit>

    # or per entrypoint:
    modal run fastvideo/tests/modal/fasth3_gpu_verify.py \
        run_h3_golden_gate --commit <commit>
    modal run fastvideo/tests/modal/fasth3_gpu_verify.py \
        run_h3_t2va_smoke --commit <commit>
    modal run fastvideo/tests/modal/fasth3_gpu_verify.py \
        run_h3_ssim --commit <commit>

Requirements: Modal profile `hao-ai-lab`; secret `fasth3-hf-token`
(HF_TOKEN / HF_API_KEY / HUGGINGFACE_HUB_TOKEN / WANDB_API_KEY); volume
`hf-model-weights` mounted at /root/data (HF cache + outputs persist across
runs, so the ~145 GB H3 weight download happens once).

Device keying: the H3 golden gate and SSIM references are minted on
`NVIDIA_GB200`; this Modal workspace has no GB200, so verification here runs
on **4xH100** (the 146 GB of weights do not fit one 80 GB H100). The
GB200-keyed golden gate runs as-is on the training cluster (B200 reports
``NVIDIA GB200`` and the existing golden matches).
"""

from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from modal_image_utils import resolve_image_ref  # noqa: E402
except ModuleNotFoundError:
    def resolve_image_ref(image_ref: str) -> str:  # type: ignore[misc]
        return image_ref

FORK_REPO = "https://github.com/aryan5v/FastVideo.git"
DEFAULT_COMMIT = os.getenv("FAS_THREE_COMMIT", "b7571796")
MODEL_REPO = "MiniMaxAI/MiniMax-H3"

app = modal.App("fasth3-gpu-verify")

image_version = os.getenv("IMAGE_VERSION", "latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
image_ref = resolve_image_ref(image_tag)

image = (
    modal.Image.from_registry(image_ref, add_python="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .env({
        "PATH": "/root/.local/bin:$PATH",
        "IMAGE_VERSION": image_version,
        "FASTVIDEO_FA4": os.environ.get("FASTVIDEO_FA4", "1"),
        # Numerics: match the environment the GB200 goldens were minted under.
        "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
)

model_vol = modal.Volume.from_name("hf-model-weights")
SECRET = modal.Secret.from_name("fasth3-hf-token")

# The golden gate is device-keyed; the B200 training cluster reports
# ``NVIDIA GB200`` and matches the minted golden. This workspace has no GB200,
# so default to H100 (the test then seeds a local golden and fails with mint
# instructions — the documented path for a new device).
GOLDEN_GATE_GPU = os.getenv("FAS3_GOLDEN_GPU", "H100")

COMMON_KWARGS = dict(
    image=image,
    volumes={"/root/data": model_vol},
    secrets=[SECRET],
    timeout=5400,
    retries=0,
)


def _prepare_workspace(commit: str) -> str:
    """Clone the fork at the requested commit; editable-install our code."""
    import subprocess

    repo_root = "/FastVideo"
    command = f"""
    set -euo pipefail
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    export HF_HOME='/root/data/.cache'
    if [ -d {repo_root}/.git ]; then
      cd {repo_root}
      git remote set-url origin {FORK_REPO}
      git fetch --prune origin
    else
      git clone {FORK_REPO} {repo_root}
      cd {repo_root}
    fi
    git checkout {commit}
    uv pip install -e ".[test]"
    hf auth login --token "$HF_API_KEY"
    """
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_API_KEY": os.environ.get("HF_API_KEY", "")},
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Workspace setup failed with exit code {result.returncode}")
    return repo_root


@app.function(gpu=GOLDEN_GATE_GPU, **COMMON_KWARGS)
def run_h3_golden_gate(commit: str = DEFAULT_COMMIT) -> str:
    """Single-block bitwise fingerprint vs the GB200 golden (fast, ~1.2 GB)."""
    import subprocess

    repo_root = _prepare_workspace(commit)
    command = (
        f"set -euo pipefail && source $HOME/.local/bin/env && source /opt/venv/bin/activate && "
        f"cd {repo_root} && export HF_HOME='/root/data/.cache' && "
        f"python -m pytest fastvideo/tests/golden_gate/test_minimax_h3_t2v.py -v -x"
    )
    result = subprocess.run(["/bin/bash", "-lc", command], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Golden gate failed with exit code {result.returncode}")
    return "golden-gate: PASS"


@app.function(gpu="H100:4", **COMMON_KWARGS)
def run_h3_t2va_smoke(
    commit: str = DEFAULT_COMMIT,
    prompt: str = (
        "A lighthouse beam sweeping across a stormy sea at night, rain streaking "
        "the lens, waves crashing against the rocks, dramatic thunder in the distance"
    ),
    steps: int = 50,
    num_frames: int = 124,
    height: int = 768,
    width: int = 1344,
    seed: int = 2026,
) -> str:
    """End-to-end T2VA smoke: real weights, video + native stereo audio out."""
    import shlex
    import subprocess

    repo_root = _prepare_workspace(commit)
    output_dir = "/root/data/fasth3_verify/smoke_t2va"
    command = (
        f"set -euo pipefail && source $HOME/.local/bin/env && source /opt/venv/bin/activate && "
        f"cd {repo_root} && export HF_HOME='/root/data/.cache' && "
        f"python examples/inference/basic/basic_minimax_h3_t2v.py "
        f"--model-path {MODEL_REPO} --prompt {shlex.quote(prompt)} --output {output_dir} "
        f"--steps {steps} --num-frames {num_frames} --height {height} --width {width} "
        f"--seed {seed} --num-gpus 4"
    )
    result = subprocess.run(["/bin/bash", "-lc", command], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"T2VA smoke failed with exit code {result.returncode}")
    model_vol.commit()
    return f"T2VA smoke: PASS -> {output_dir}/minimax_h3_t2v.mp4"


@app.function(gpu=os.getenv("FAS3_SSIM_GPU", "H100:4"), **COMMON_KWARGS)
def run_h3_ssim(commit: str = DEFAULT_COMMIT) -> str:
    """Full H3 SSIM gate (4 GPUs, GB200-keyed references; ~13-min generation)."""
    import subprocess

    repo_root = _prepare_workspace(commit)
    command = (
        f"set -euo pipefail && source $HOME/.local/bin/env && source /opt/venv/bin/activate && "
        f"cd {repo_root} && export HF_HOME='/root/data/.cache' && "
        f"python -m pytest fastvideo/tests/ssim/test_minimax_h3_similarity.py -v -x"
    )
    result = subprocess.run(["/bin/bash", "-lc", command], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"H3 SSIM failed with exit code {result.returncode}")
    return "H3 SSIM: PASS"


@app.function(gpu="H100:1", **COMMON_KWARGS)
def run_h3_parity_suite(commit: str = DEFAULT_COMMIT) -> str:
    """Device-agnostic upstream parity suites (transformer, VAEs, scheduler,
    packing, pipeline smoke) — the correctness gates for the torch side."""
    import subprocess

    repo_root = _prepare_workspace(commit)
    command = (
        f"set -euo pipefail && source $HOME/.local/bin/env && source /opt/venv/bin/activate && "
        f"cd {repo_root} && export HF_HOME='/root/data/.cache' && "
        f"export MINIMAX_H3_RUN_DIT_PARITY=1 MINIMAX_H3_RUN_VAE_PARITY=1 "
        f"MINIMAX_H3_RUN_VIDEO_VAE_PARITY=1 MINIMAX_H3_RUN_AUDIO_VAE_PARITY=1 && "
        f"python -m pytest "
        f"tests/local_tests/transformers/test_minimax_h3_transformer_parity.py "
        f"tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py "
        f"tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py "
        f"tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py "
        f"tests/local_tests/minimax_h3/test_minimax_h3_packing.py "
        f"-v -x"
    )
    result = subprocess.run(["/bin/bash", "-lc", command], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"H3 parity suite failed with exit code {result.returncode}")
    return "H3 parity suite: PASS"


@app.local_entrypoint()
def verify(
    commit: str = DEFAULT_COMMIT,
    run_golden_gate: bool = True,
    run_smoke: bool = True,
    run_ssim: bool = False,
) -> None:
    print(f"FastH3 GPU verification on commit {commit}")
    if run_golden_gate:
        print(run_h3_golden_gate.remote(commit=commit))
    if run_smoke:
        print(run_h3_t2va_smoke.remote(commit=commit))
    if run_ssim:
        print(run_h3_ssim.remote(commit=commit))
