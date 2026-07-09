# FastWan-QAD-INT8-1.3B: Local Text-to-Video on Apple Silicon

> Draft — do not publish until the release owner selects raw or EMA from the
> final visual review and the model-card checksums are available.

## TL;DR

FastVideo's Apple-native FastWan runtime generates a five-second 480p-class
text-to-video clip locally on an Apple M4 Max. The release candidate uses an
MLX INT8 DiT, a three-step DMD schedule, and TAEHV decode. The recorded
480x832x81 result is 123.7 seconds end to end, including 117.6 seconds of
denoising, at 5.63 GiB MLX peak memory.

This is a source-install, text-to-video release for the validated M4 Max,
36 GB-class configuration. It is not a general 16 GB Mac claim, and it does
not include image-to-video.

## What ships

- FastWan-QAD-INT8-1.3B in Diffusers form plus a pre-quantized MLX DiT
  checkpoint that avoids requantizing weights at startup.
- One source-tree generation command using MLX for denoising and MPS for text
  encoding and TAEHV decode.
- A reproducible benchmark harness and Metal/MLX-CPU runtime tests.

TAEHV is vendored under its MIT license; FastVideo source is Apache-2.0.

## Results and release decision

The candidate raw and EMA checkpoints measure respectively 0.9360 and 0.9331
mean MS-SSIM when each INT8 result is compared with the same model's FP16
result. That metric measures quantization consistency only. It does not score
absolute visual quality, so it cannot select a release checkpoint.

**Release owner decision: [TODO: raw or EMA after final motion7 visual review].**

The final post must include the selected revision, checksums, model-card link,
fixed generation command, and the reviewed visual examples. It must not claim
bitwise fake-quant parity, use the invalid 0.9860 run-1 EMA number, or claim
that 16 GB Macs have been validated.

## Run it

```console
uv pip install -e '.[mlx]'
python examples/inference/basic/mlx_wan_prompt_to_video.py [release command from the model card]
```

See the Apple Silicon FastWan guide for the full command, hardware statement,
and troubleshooting.
