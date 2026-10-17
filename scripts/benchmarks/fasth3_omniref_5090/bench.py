# SPDX-License-Identifier: Apache-2.0
"""RTX 5090 OmniRef: two warm-ups, then two measured clips per reference set."""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class MemoryPeak:
    """Sample total pod cgroup memory and NVML GPU use every 100 ms."""

    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.host = 0
        self.anon = 0
        self.gpu = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                cgroup = Path("/sys/fs/cgroup")
                self.host = max(self.host, int((cgroup / "memory.current").read_text()))
                stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
                self.anon = max(self.anon, int(stats["anon"]))
                self.gpu = max(self.gpu, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            except Exception as exc:
                self.error = str(exc)
                return
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.nvml.nvmlShutdown()

    def result(self):
        return {"peak_host_bytes": self.host if self.error is None else None,
                "peak_host_anon_bytes": self.anon if self.error is None else None,
                "peak_gpu_used_bytes": self.gpu if self.error is None else None,
                "sampling_error": self.error}


def load_example():
    path = ROOT / "examples/inference/basic/basic_fasth3_omniref_pdd.py"
    spec = importlib.util.spec_from_file_location("omniref_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--adaln-table", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="One untimed full-size clip only")
    parser.add_argument("--quality", action="store_true", help="One untimed clip for each reference set")
    parser.add_argument("--profile", action="store_true", help="Log CUDA stage spans and attention token geometry")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    os.environ.setdefault("FASTVIDEO_NVFP4_MM_BACKEND", "cudnn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")
    if args.profile:
        os.environ["FASTVIDEO_H3_SP_PROFILE"] = "1"

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig
    from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

    if args.profile:
        from fastvideo.models.dits import minimax_h3_vsa_fp4

        original_layout = minimax_h3_vsa_fp4._layout_for
        seen_geometry = set()

        def logged_layout(meta, rotary_emb):
            layout = original_layout(meta, rotary_emb)
            geometry = (int(meta.untile_combined_index.numel()), layout.rows, int(meta.num_prefix_tiles),
                        layout.tile, layout.n_tiles)
            if geometry not in seen_geometry:
                seen_geometry.add(geometry)
                prefix = meta.variable_block_sizes[:meta.num_prefix_tiles]
                print("ATTENTION_GEOMETRY " + json.dumps({
                    "packed_tokens": geometry[0], "padded_rows": layout.rows,
                    "prefix_valid_tokens": int(prefix.sum().item()),
                    "prefix_padded_rows": int(meta.num_prefix_tiles) * layout.tile,
                    "tile_size": layout.tile, "tiles": layout.n_tiles,
                    "sparsity": float(meta.VSA_sparsity),
                }), flush=True)
            return layout

        minimax_h3_vsa_fp4._layout_for = logged_layout

    example = load_example()
    profile = example.DEVICE_PROFILES["rtx-5090"]
    example.apply_profile(profile, args.decoder, args.adaln_table)
    config = example.build_generator_config(args.model, 1, profile=profile)
    contract = example.load_contract(args.model)
    assert contract["num_inference_steps"] == 8
    cases = [
        {"name": "speaker", "image": "assets/images/couple.jpg",
         "prompt": "A person on a sofa, lit by a warm lamp, speaking to camera."},
        {"name": "dance", "image": "assets/images/80s_dance.jpg",
         "prompt": "The man in the kitchen spins and steps quickly to upbeat disco music. His face stays clear "
                   "as he turns toward the camera, with energetic arm movements and natural motion."},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"model": str(args.model), "decoder": args.decoder, "adaln_table": args.adaln_table,
               "torch": torch.__version__, "cuda": torch.version.cuda, "contract": contract,
               "source_commit": os.environ.get("FASTVIDEO_SOURCE_COMMIT"),
               "gpu": subprocess.check_output(
                   ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader"],
                   text=True).strip(),
               "command": [sys.executable, *sys.argv],
               "env": {key: value for key, value in os.environ.items()
                       if key.startswith(("FASTVIDEO_", "PYTORCH_", "CUDNN_"))},
               "cases": cases, "seed": 1234, "shape": [832, 480, 124], "fps": 24, "runs": []}
    destination = args.output / "results.json"

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(destination)

    save()
    generator = None
    try:
        with MemoryPeak() as load_memory:
            started = time.perf_counter()
            generator = VideoGenerator.from_config(config)
            results["load_s"] = time.perf_counter() - started
        results["load_memory"] = load_memory.result()
        save()
        for case in cases[:1] if args.smoke else cases:
            references = [MiniMaxH3Reference(source=str(ROOT / case["image"]), media_type="image")]
            for run in range(1 if args.smoke or args.quality else 4):
                output = args.output / f"{case['name']}_{run}.mp4"
                request = GenerationRequest(
                    prompt=case["prompt"], negative_prompt="", inputs=InputConfig(references=references),
                    sampling=SamplingConfig(seed=1234, height=480, width=832, num_frames=124, fps=24,
                                            num_inference_steps=8, guidance_scale=1.0, batch_cfg=False),
                    output=OutputConfig(output_path=str(output), save_video=True, return_frames=False))
                with MemoryPeak() as memory:
                    started = time.perf_counter()
                    result = generator.generate(request)
                    elapsed = time.perf_counter() - started
                stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
                record = {"case": case["name"], "run": run, "warmup": run < 2, "wall_s": elapsed,
                          "output": str(result.video_path), "memory": memory.result(),
                          "peak_alloc_mb": result.peak_memory_mb,
                          "stages_s": {name: float(metrics["execution_time"]) for name, metrics in stages.items()
                                       if metrics.get("execution_time") is not None}}
                results["runs"].append(record)
                save()
                print("RUN " + json.dumps(record), flush=True)
        results["medians_s"] = {case["name"]: statistics.median(
            run["wall_s"] for run in results["runs"] if run["case"] == case["name"] and not run["warmup"])
            for case in cases if not args.smoke and not args.quality}
        save()
    except BaseException as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        monitor = locals().get("memory", locals().get("load_memory"))
        if monitor is not None:
            results["failure_memory"] = monitor.result()
        save()
        raise
    finally:
        if generator is not None:
            generator.shutdown()


if __name__ == "__main__":
    main()
