#!/usr/bin/env bash
# Run on the assigned pod after rsyncing this checkout, including CUTLASS headers.
set -euo pipefail

repo=${OMNIREF_REPO:-/root/FastVideo}
venv=${OMNIREF_VENV:-/root/fv}
source_model=${OMNIREF_SOURCE_MODEL:-/workspace/models/FastH3-Omni-NVFP4-QAD-v2}
local_model=${OMNIREF_LOCAL_MODEL:-/root/models/FastH3-Omni-NVFP4-QAD-v2}
export UV_LINK_MODE=copy
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python3 - "$source_model" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
component = root / "transformer_ref"
json.loads((component / "config.json").read_text())
index = json.loads((component / "diffusion_pytorch_model.safetensors.index.json").read_text())
for name in {*index["weight_map"].values(), "nvfp4_weights.safetensors"}:
    path = component / name
    if not path.is_file() or path.stat().st_size <= 8:
        raise SystemExit(f"Missing or empty model file: {path}. Restore it before setup.")
PY

if ! pgrep -f '^/bin/bash /root/reclaim.sh$' >/dev/null; then
    cat > /root/reclaim.sh <<'SH'
#!/bin/bash
while true; do
    current=$(cat /sys/fs/cgroup/memory.current)
    [ "$current" -gt 35000000000 ] && echo 25G > /sys/fs/cgroup/memory.reclaim 2>/dev/null
    sleep 5
done
SH
    chmod +x /root/reclaim.sh
    setsid nohup /root/reclaim.sh >/dev/null 2>&1 </dev/null &
fi

if [[ ! -x "$venv/bin/python" ]]; then
    uv venv "$venv" --python python3.12
fi
source "$venv/bin/activate"
cd "$repo"
uv pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match -e ".[dev]"
uv pip install 'cuda-toolkit[nvcc,cccl,crt,nvvm,cublas,cudart,curand,cufft,cusolver,cusparse,nvrtc,nvtx,nvjitlink]==13.0.2'
CUDA_HOME=$(python - <<'PY'
from importlib.metadata import distribution
dist = distribution("nvidia-cuda-nvcc")
nvcc = next(path for path in dist.files if str(path).endswith("/bin/nvcc"))
print(dist.locate_file(nvcc).parent.parent)
PY
)
export CUDA_HOME
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["CUDA_HOME"])
if not (root / "lib64").exists():
    (root / "lib64").symlink_to("lib", target_is_directory=True)
for library in (root / "lib").glob("*.so.*"):
    target = library.with_name(library.name.split(".so.", 1)[0] + ".so")
    if not target.exists():
        target.symlink_to(library.name)
PY
(
    cd fastvideo-kernel
    ./build.sh
)
mkdir -p "$local_model"
for name in transformer text_encoder tokenizer processor vae audio_vae scheduler audio_scheduler \
    vae_decoder_nvfp4 fastvideo_inference.json modular_model_index.json README.md; do
    if [[ ! -e "$local_model/$name" ]]; then
        cp -rL "$source_model/$name" "$local_model/$name"
    fi
done
if [[ ! -e "$local_model/transformer_ref" ]]; then
    ln -s transformer "$local_model/transformer_ref"
fi
if [[ ! -f /root/models/decoder-8L-nvfp4.safetensors ]]; then
    cp /workspace/models/FastH3-NVFP4-VAE-Decoder/fast-8L/decoder-nvfp4.safetensors \
        /root/models/decoder-8L-nvfp4.safetensors
fi
python scripts/benchmarks/fasth3_omniref_5090/prepare_adaln.py \
    "$local_model" /root/models/omniref-v2-adaln.pt
python -c 'import torch, fastvideo, fp4attn_cuda, fp4quant_cuda; print(torch.__version__, torch.cuda.get_device_name())'
