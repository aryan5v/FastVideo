# 5B MXFP4 2-Step QAD Training Recipe

## What Changed

- Added a torch-only MXFP4 QAT twin in `fastvideo/layers/quantization/mlx_mxfp4_qat.py`.
- Extended `fastvideo/train/callbacks/mlx_qat.py` with `mode: affine | mxfp4`; affine remains the default, while MXFP4 uses the fixed MLX grid (group size 32, no `bits`).
- Added the Mac parity gate in `fastvideo/tests/mlx/test_mlx_mxfp4_qat_parity.py`.
- Added the draft 4xB200 config in `examples/train/configs/distribution_matching/wan/dmd2_ti2v_5b_mlx_mxfp4.yaml`.

## Mac Verification

Run from `/Users/aryank/fvtrain`:

```bash
source /Users/aryank/claude-fastvideo/FastVideo/.venv/bin/activate
export PYTHONPATH=$PWD
pytest fastvideo/tests/mlx/test_mlx_mxfp4_qat_parity.py -q
python - <<'PY'
import yaml
path = "examples/train/configs/distribution_matching/wan/dmd2_ti2v_5b_mlx_mxfp4.yaml"
with open(path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["method"]["dmd_denoising_steps"], cfg["pipeline"]["dmd_denoising_steps"])
PY
python - <<'PY'
import importlib.util
import sys
import types
import torch

train_mod = types.ModuleType("fastvideo.train")
train_mod.__path__ = []
callbacks_mod = types.ModuleType("fastvideo.train.callbacks")
callbacks_mod.__path__ = []
callback_mod = types.ModuleType("fastvideo.train.callbacks.callback")
class Callback:
    pass
callback_mod.Callback = Callback
sys.modules["fastvideo.train"] = train_mod
sys.modules["fastvideo.train.callbacks"] = callbacks_mod
sys.modules["fastvideo.train.callbacks.callback"] = callback_mod

spec = importlib.util.spec_from_file_location("mlx_qat_under_test", "fastvideo/train/callbacks/mlx_qat.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

w = torch.tensor([[0.0, 0.1, 0.2, 0.4] * 8], dtype=torch.float32)
fq = mod._fake_quantize_weight(w, mode="mxfp4", group_size=32, bits=8, simulate_dtype=torch.float16)
print(bool(torch.any(fq != w)))
PY
```

Latest local result:

```text
pytest fastvideo/tests/mlx/test_mlx_mxfp4_qat_parity.py -q
...........                                                              [100%]
11 passed in 0.05s

python -m py_compile fastvideo/layers/quantization/mlx_mxfp4_qat.py fastvideo/train/callbacks/mlx_qat.py fastvideo/tests/mlx/test_mlx_mxfp4_qat_parity.py
PASS

yaml.safe_load examples/train/configs/distribution_matching/wan/dmd2_ti2v_5b_mlx_mxfp4.yaml
top ['callbacks', 'method', 'models', 'pipeline', 'training']
student FastVideo/FastWan2.2-TI2V-5B-Diffusers
method_steps [1000, 500]
pipeline_steps [1000, 500]
qat {'mode': 'mxfp4', 'simulate_dtype': 'fp16'}

stubbed mlx_qat.py callback-mode check
changed True
callback mxfp4 32

pytest fastvideo/tests/mlx/test_mlx_affine_qat_parity.py -q
.....................                                                    [100%]
21 passed in 0.06s
```

I attempted `fastvideo.train.utils.config.load_run_config(...)`, but this Mac
venv is missing optional training data dependencies (`pyarrow`, then
`datasets`) and importing `fastvideo.train` pulls those modules through package
side effects. I did not install packages on this Mac.

## Recipe

Best-guess launch command for a single 4-GPU DGX B200 node:

```bash
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
torchrun --nnodes 1 --nproc_per_node 4 --master_port 29500 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/distribution_matching/wan/dmd2_ti2v_5b_mlx_mxfp4.yaml
```

The config uses `Wan-AI/Wan2.2-TI2V-5B-Diffusers` for teacher and critic/fake-score initialization. The student warm-start is `FastVideo/FastWan2.2-TI2V-5B-Diffusers`, because the registry exposes it as the FastWan TI2V 5B distilled path.

Data-free wiring is through `method.rollout_mode: simulate`. In this mode, `DMD2Method.single_train_step` uses `latents_source="zeros"` and `_student_rollout` simulates generator states instead of consuming VAE latents from the dataset. The current modular trainer still needs text-conditioning batches from a dataloader; `training.data.data_path` is therefore a placeholder for a text-conditioning parquet/prompt source and must not point at `Wan-Syn`.

## 2-Step Quality Retention Plan

Moving from 3 steps to 2 steps is a direct speed win because it removes one denoise forward, but it is a quality risk. The run should only ship 2-step if fixed-prompt/fixed-seed validation holds against the 3-step warm-start/reference. Otherwise, fall back to the 3-step schedule.

Intended gate:

- Render the 2-step schedule `[1000, 500]` every 200 training steps on `validation_64.json`.
- Render the 3-step reference `[1000, 757, 522]` on the same prompts/seeds every 200 steps.
- Compare prompt adherence, motion coherence, temporal stability, and objectionable artifacts. If 2-step visibly regresses versus 3-step, stop or continue with 3-step instead of forcing 2-step.
- Warm-starting from `FastVideo/FastWan2.2-TI2V-5B-Diffusers` is part of the risk reduction plan; starting this 2-step MXFP4 run from the base model is not the preferred path.

Important: the current checked-in config can set the 2-step DMD validation
schedule through `pipeline.dmd_denoising_steps`, but `WanDMDPipeline` reads that
single pipeline-level schedule and ignores each validation callback's
`sampling_timesteps`. The 3-step A/B reference therefore needs reviewer action:
either add per-callback DMD schedule support, or run a second validation
config/override with `pipeline.dmd_denoising_steps=[1000,757,522]`.

## Reviewer Checklist

- [ ] Confirm `FastVideo/FastWan2.2-TI2V-5B-Diffusers` is the intended already-distilled 5B student checkpoint. I found it in `fastvideo/registry.py`, but not as an explicit output in `examples/distill/Wan2.2-TI2V-5B-Diffusers/`.
- [ ] Confirm the 2-step schedule. I did not find a canonical 2-step Wan2.2 5B schedule in the repo, so the draft uses `[1000, 500]`.
- [ ] Confirm the data-free text-conditioning source for `training.data.data_path`. The config deliberately avoids Wan-Syn latents because Wan-Syn is Wan2.1/16-channel and incompatible with Wan2.2 TI2V 5B/48-channel latents.
- [ ] Confirm that `fastvideo.train.models.wan.WanModel` is the correct training wrapper for Wan2.2 TI2V 5B. It loads `WanTransformer3DModel` through `load_module_from_path`, and `Wan2_2_TI2V_5B_Config` is selected by the registry for the model IDs above.
- [ ] Confirm I2V validation requirements. The new-stack TI2V LoRA config says validation/inference can use `WanPipeline` with TI2V activated by pipeline config and `image_path`/`video_path`; this QAD draft uses `WanDMDPipeline`. If I2V-specific validation is mandatory, add image paths to `validation_64.json` or split a separate I2V validation pass.
- [ ] Confirm latent dimensions: the legacy data-free 5B script uses `num_latent_t=31`, `num_height=704`, `num_width=1280`, `num_frames=121`; the modular config mirrors those.
- [ ] Confirm validation A/B wiring before launch. Current `WanDMDPipeline` ignores the validation callback's `sampling_timesteps` and reads `pipeline_config.dmd_denoising_steps`; with the task's file-edit constraints, the checked-in config gates the 2-step schedule, but the 3-step reference callback may need a validation callback enhancement or a separate launch override/config to truly render `[1000, 757, 522]`.
- [ ] Confirm `flow_shift: 5`; this is set from the Wan2.2 TI2V 5B config and legacy 5B script.
- [ ] Confirm `num_gpus: 4` and `hsdp_shard_dim: 4` match the target 4xB200 topology.
