# FastH3 OmniRef on an RTX 5090

Use the consumer profile at 832 by 480, 124 frames, 24 fps, eight PDD forwards,
and seed 1234. `bench.py` creates one generator, runs two warm-ups for each
reference set, and measures the next two complete MP4 writes with generated audio.
It records stage times, sampled NVML GPU use, and total cgroup memory including
file cache. Generator loading has separate memory measurements.

The speaker case uses the RTX 5090 example's prompt and `couple.jpg`. The dance
case uses `80s_dance.jpg` to exercise faces and fast motion. Both references ship
in the repository. These two cases are a limited quality sample.

## Prepare the pod

Rsync the checkout and its initialized `fastvideo-kernel/include/cutlass`
submodule to `/root/FastVideo`. Exclude `.git`, virtual environments, and outputs.
Run `setup.sh` on the pod, using local disk for the environment and model copy.
The network volume only supplies existing models. Offline Hugging Face mode
prevents missing components from triggering downloads.

```bash
bash /root/FastVideo/scripts/benchmarks/fasth3_omniref_5090/setup.sh
```

The setup selects the CUDA 13 PyTorch index explicitly. The task's suggested
CUDA 12.8 index does not provide the repository's pinned PyTorch 2.12.0, and
older uv releases do not accept `UV_TORCH_BACKEND=cu130`. The setup also installs
the matching CUDA 13 compiler and development components into the virtual
environment through NVIDIA's `cuda-toolkit` package. Set `HTTPS_PROXY` if the
pod needs a working package route; no proxy endpoint or credentials are stored
in the setup script.

`prepare_adaln.py` computes tables from the selected checkpoint's own time
embedder and AdaLN projections. It uses the pipeline's PDD plan and scheduler,
preserves FP32 timestep embeddings and BF16 modulation, and checks exact native
modulation and cached lookup results. It covers image/video references with or
without reference audio. A table belongs to its checkpoint and fixed schedule.

## Benchmark

```bash
source /root/fv/bin/activate
cd /root/FastVideo
export CUDA_HOME=$(python - <<'PYCUDA'
from importlib.metadata import distribution
dist = distribution("nvidia-cuda-nvcc")
print(dist.locate_file(next(path for path in dist.files if str(path).endswith("/bin/nvcc"))).parent.parent)
PYCUDA
)
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/benchmarks/fasth3_omniref_5090/bench.py \
  --model /root/models/FastH3-Omni-NVFP4-QAD-v2 \
  --decoder /root/models/FastH3-Omni-NVFP4-QAD-v2/vae_decoder_nvfp4/decoder-nvfp4.safetensors \
  --adaln-table /root/models/omniref-v2-adaln.pt \
  --output /root/omniref-5090-results/v2-36L
```

Repeat with `/root/models/decoder-8L-nvfp4.safetensors` and another output
directory for the eight-layer decoder. `--smoke` writes one full-size untimed
clip. `--quality` writes one untimed clip per reference set. Neither produces
timing medians. Run the previous checkpoint with its own AdaLN table and the
same prompts, references, seed, decoder, and attention settings for comparisons.

The benchmark defaults to the cuDNN NVFP4 GEMM backend and expandable allocator
segments. Explicit environment settings override those defaults. No inference
compilation is enabled by the example's generator configuration.
