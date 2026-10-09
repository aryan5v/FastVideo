# SPDX-License-Identifier: Apache-2.0
"""T3/T4 metrics for the OmniRef NVFP4 Stage A gate, plus labelled side-by-side MP4s.

``score`` decodes, per held-out row, bf16 seed s, bf16 seed s+1 (the noise floor) and NVFP4 seed s,
plus the row's visual references, and records:

- reference fidelity: DINOv2 and CLIP-I similarity of sampled frames to the references
  (``dino_ref``, ``clip_ref``); first/last frame PSNR and LPIPS to the keyframes
  (first_frame / first_last_frame / storyboard: first reference; first_last_frame: last reference;
  continue_*: the last frame of the preceding video, i.e. continuity);
- reference-free: CLIP-T to the caption, LAION aesthetic score (CLIP-L + MLP), VBench-style
  subject consistency (DINOv2 frame-to-first and frame-to-previous);
- pairwise against bf16 seed s (T3): LPIPS / PSNR of the video, log-mel L1 and cosine of the audio.

ArcFace, VisionReward, imaging quality (MUSIQ), motion smoothness and AV sync are not computed.

``report`` aggregates with paired bootstrap CIs and applies Gate A's T4 rule: reference-fidelity
metrics may drop at most 2% relative to bf16 seed s; every other metric's mean shift must stay
within the bf16 seed noise floor (|mean NVFP4 - bf16 s| <= max(|mean bf16 s+1 - bf16 s|, its 95% CI
half-width)).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "distill" / "minimax_h3_nvfp4_decoder"))

from calibrate_omniref_nvfp4 import add_plan_args, heldout_name, heldout_plan  # noqa: E402

VARIANTS = ("bf16_s0", "bf16_s1", "nvfp4_s0")
REF_FIDELITY = ("dino_ref", "clip_ref", "first_psnr", "first_lpips", "last_psnr", "last_lpips")
LOWER_IS_BETTER = {"first_lpips", "last_lpips"}
GATE_REF_DROP = 0.02
SAMPLED_FRAMES = 8


def _embeds(output: Any) -> torch.Tensor:
    """Projected CLIP features: a tensor in transformers 4.x, ``pooler_output`` of a model output in 5.x."""
    return (output if isinstance(output, torch.Tensor) else output.pooler_output).float()


class Scorers:

    def __init__(self, models_dir: Path, device: torch.device) -> None:
        import lpips
        from transformers import AutoModel, CLIPModel, CLIPTokenizer

        self.device = device
        self.dino = AutoModel.from_pretrained(models_dir / "dinov2-base").to(device).eval()
        self.clip = CLIPModel.from_pretrained(models_dir / "clip-vit-large-patch14").to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(models_dir / "clip-vit-large-patch14")
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        head = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128),
                                   torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                                   torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        state = torch.load(models_dir / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth", map_location="cpu",
                           weights_only=True)
        head.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
        self.aesthetic = head.to(device).eval()

    @staticmethod
    def _resize_norm(frames: torch.Tensor, size: int, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
        x = F.interpolate(frames, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
        return (x - torch.tensor(mean, device=x.device).view(1, 3, 1, 1)) / torch.tensor(std, device=x.device).view(
            1, 3, 1, 1)

    @torch.no_grad()
    def dino_embed(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._resize_norm(frames, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return F.normalize(self.dino(pixel_values=x).pooler_output.float(), dim=-1)

    @torch.no_grad()
    def clip_embed(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._resize_norm(frames, 224, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return _embeds(self.clip.get_image_features(pixel_values=x))

    @torch.no_grad()
    def clip_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text], truncation=True, max_length=77, padding=True, return_tensors="pt").to(self.device)
        return F.normalize(_embeds(self.clip.get_text_features(**tokens)), dim=-1)

    @torch.no_grad()
    def lpips_mean(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.cat([self.lpips(a[i:i + 8] * 2 - 1, b[i:i + 8] * 2 - 1).flatten()
                                for i in range(0, a.shape[0], 8)]).mean())


def to_float(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 ``[T, H, W, 3]`` -> float ``[T, 3, H, W]`` in [0, 1]."""
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(255)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b)**2).mean())
    return 99.0 if mse == 0 else 10 * float(np.log10(1.0 / mse))


def match(ref: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return ref if ref.shape[-2:] == like.shape[-2:] else F.interpolate(ref, size=like.shape[-2:], mode="bilinear",
                                                                       antialias=True, align_corners=False)


def log_mel(waveform: torch.Tensor, rate: int) -> torch.Tensor:
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=rate, n_fft=1024, hop_length=256, n_mels=80).to(
        waveform.device)
    return (mel(waveform.float().mean(-1)) + 1e-5).log()


def reference_frames(driver: Any, row: dict[str, Any]) -> list[np.ndarray]:
    """Decoded visual references in order, each uint8 ``[T, H, W, 3]`` (images have T = 1)."""
    from generate_omniref_latents import _tensor, prepared_references

    from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens

    patch = h3_dit_patch_size(driver.fastvideo_args)
    rows = _tensor(row, "reference_video_latent_rows")
    decoded, cursor = [], 0
    for ref in prepared_references(row):
        if ref.media_type == "audio":
            continue
        count = ref.num_latent_frames * (ref.latent_height // patch[1]) * (ref.latent_width // patch[2])
        latent = unpatchify_video_tokens(rows[cursor:cursor + count], ref.num_latent_frames, ref.latent_height,
                                         ref.latent_width, 24, patch)[0]
        cursor += count
        decoded.append(driver.decode(latent))
    return decoded


def score_video(scorers: Scorers, video: torch.Tensor, refs: list[torch.Tensor], case: str,
                text: torch.Tensor | None) -> dict[str, float]:
    count = video.shape[0]
    picks = torch.linspace(0, count - 1, SAMPLED_FRAMES).round().long().tolist()
    sampled = video[picks]
    dino, clip = scorers.dino_embed(sampled), scorers.clip_embed(sampled)
    clip_n = F.normalize(clip, dim=-1)
    metrics: dict[str, float] = {}
    if refs:
        ref_dino = [F.normalize(scorers.dino_embed(r[torch.linspace(0, r.shape[0] - 1, min(4, r.shape[0])).long()])
                                .mean(0, keepdim=True), dim=-1) for r in refs]
        ref_clip = [F.normalize(scorers.clip_embed(r[torch.linspace(0, r.shape[0] - 1, min(4, r.shape[0])).long()])
                                .mean(0, keepdim=True), dim=-1) for r in refs]
        metrics["dino_ref"] = float(np.mean([float((dino @ e.T).max()) for e in ref_dino]))
        metrics["clip_ref"] = float(np.mean([float((clip_n @ e.T).max()) for e in ref_clip]))
        if case.startswith("continue"):
            anchor = refs[0][-1:]
        else:
            anchor = refs[0][:1]
        first = video[:1]
        anchor = match(anchor, first)
        metrics["first_psnr"], metrics["first_lpips"] = psnr(first, anchor), scorers.lpips_mean(first, anchor)
        if case == "first_last_frame" and len(refs) > 1:
            last, anchor = video[-1:], match(refs[-1][:1], video[-1:])
            metrics["last_psnr"], metrics["last_lpips"] = psnr(last, anchor), scorers.lpips_mean(last, anchor)
    if text is not None:
        metrics["clip_t"] = float((clip_n @ text.T).mean())
    metrics["aesthetic"] = float(scorers.aesthetic(clip_n).mean())
    every = video[::2]
    emb = scorers.dino_embed(every)
    metrics["subject_consistency"] = float(((emb[1:] @ emb[0]) + (emb[1:] * emb[:-1]).sum(-1)).mean() / 2)
    return metrics


def side_by_side(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=18)
    count = min(frames.shape[0] for _, frames in panels)
    columns = []
    for label, frames in panels:
        strip = Image.new("RGB", (frames.shape[2], 26), "black")
        ImageDraw.Draw(strip).text((6, 3), label, fill="white", font=font)
        columns.append(np.concatenate((np.asarray(strip)[None].repeat(count, 0), frames[:count]), axis=1))
    imageio.mimsave(path, list(np.concatenate(columns, axis=2)), fps=24, format="mp4", quality=7, macro_block_size=2)


def score(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from generate_omniref_latents import OmniRefLatentGenerator, read_row

    out = Path(args.output_dir)
    media = out / "sbs"
    media.mkdir(parents=True, exist_ok=True)
    path = out / f"t4-shard{args.shard:02d}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    plan = [clip for clip in heldout_plan(args)[args.shard::args.num_shards] if clip["id"] not in done]
    if not plan:
        return
    device = torch.device("cuda")
    scorers = Scorers(Path(args.models_dir), device)
    driver = OmniRefLatentGenerator(args)
    sources = {"bf16_s0": Path(args.bf16_dir), "bf16_s1": Path(args.bf16_dir), "nvfp4_s0": Path(args.nvfp4_dir)}
    sbs_written = 0
    for clip in plan:
        files = {tag: sources[tag] / heldout_name(clip, tag) for tag in VARIANTS}
        if not all(f.exists() for f in files.values()):
            print(json.dumps({"id": clip["id"], "skip": [t for t, f in files.items() if not f.exists()]}), flush=True)
            continue
        row = read_row(clip["parquet"])
        caption = pq.read_table(clip["parquet"], columns=["caption"]).column("caption").to_pylist()[0]
        text = scorers.clip_text(caption) if caption else None
        refs_np = reference_frames(driver, row)
        refs = [to_float(r, device) for r in refs_np]
        decoded, results, mels = {}, {}, {}
        for tag, file in files.items():
            item = torch.load(file, weights_only=False)
            decoded[tag] = driver.decode(item["video"])
            video = to_float(decoded[tag], device)
            results[tag] = score_video(scorers, video, refs, clip["case"], text)
            if "waveform" in item:
                mels[tag] = log_mel(item["waveform"].to(device), item["sample_rate"])
        base = to_float(decoded["bf16_s0"], device)
        for tag in ("bf16_s1", "nvfp4_s0"):
            video = to_float(decoded[tag], device)
            results[tag]["lpips_vs_bf16"] = scorers.lpips_mean(video[::4], base[::4])
            results[tag]["psnr_vs_bf16"] = psnr(video, base)
            if tag in mels and "bf16_s0" in mels:
                a, b = mels[tag], mels["bf16_s0"]
                length = min(a.shape[-1], b.shape[-1])
                a, b = a[..., :length], b[..., :length]
                results[tag]["mel_l1_vs_bf16"] = float((a - b).abs().mean())
                results[tag]["mel_cos_vs_bf16"] = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        record = {k: clip[k] for k in ("id", "case", "resolution", "seed")} | {"metrics": results}
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({"id": clip["id"], **{t: {k: round(v, 4) for k, v in m.items()} for t, m in results.items()}}),
              flush=True)
        if sbs_written < args.sbs_per_shard and clip["resolution"] == "480p":
            panels = [("bf16 seed s", decoded["bf16_s0"]), ("NVFP4 seed s", decoded["nvfp4_s0"]),
                      ("bf16 seed s+1", decoded["bf16_s1"])]
            if refs_np:
                anchor = refs_np[0][-1:] if clip["case"].startswith("continue") else refs_np[0][:1]
                still = match(to_float(anchor, device), base[:1]).mul(255).round().byte().permute(0, 2, 3, 1)
                panels.insert(0, ("reference", np.repeat(still.cpu().numpy(), decoded["bf16_s0"].shape[0], axis=0)))
            side_by_side(media / f"{clip['case']}_{clip['id'][-24:]}.mp4", panels)
            sbs_written += 1
    driver.shutdown()


# --------------------------------------------------------------------------- report
def bootstrap_ci(values: np.ndarray, rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(rounds, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    rows = [json.loads(line) for p in sorted(out.glob("t4-shard*.jsonl")) for line in p.read_text().splitlines()]
    rows = [r for r in rows if r["resolution"] in args.gate_resolutions]
    metrics = sorted({k for r in rows for k in r["metrics"]["bf16_s0"]})
    table: dict[str, Any] = {}
    passed = bool(rows)  # fail closed: no scored rows, no pass
    for metric in metrics:
        paired = [r for r in rows if all(metric in r["metrics"][t] for t in VARIANTS)]
        if len(paired) < 3:
            continue
        base = np.array([r["metrics"]["bf16_s0"][metric] for r in paired])
        quant = np.array([r["metrics"]["nvfp4_s0"][metric] for r in paired])
        seed = np.array([r["metrics"]["bf16_s1"][metric] for r in paired])
        dq, ds = quant - base, seed - base
        entry = {"n": len(paired), "bf16_s0": float(base.mean()), "nvfp4_s0": float(quant.mean()),
                 "bf16_s1": float(seed.mean()), "delta_nvfp4": float(dq.mean()), "delta_nvfp4_ci": bootstrap_ci(dq),
                 "delta_seed": float(ds.mean()), "delta_seed_ci": bootstrap_ci(ds)}
        if metric in REF_FIDELITY:
            sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
            drop = sign * dq.mean() / abs(base.mean())
            entry["relative_drop"] = float(drop)
            entry["relative_drop_seed"] = float(sign * ds.mean() / abs(base.mean()))
            entry["rule"] = f"drop <= {GATE_REF_DROP:.0%}"
            entry["passed"] = bool(drop <= GATE_REF_DROP)
        else:
            low, high = entry["delta_seed_ci"]
            floor = max(abs(ds.mean()), (high - low) / 2)
            entry["noise_floor"] = floor
            entry["rule"] = "|delta_nvfp4| <= noise floor"
            entry["passed"] = bool(abs(dq.mean()) <= floor)
        passed &= entry["passed"]
        table[metric] = entry
    required = {"dino_ref", "clip_ref", "first_psnr", "first_lpips"}
    missing = sorted(required - set(table))
    passed = passed and not missing
    pairwise = {}
    for metric in ("lpips_vs_bf16", "psnr_vs_bf16", "mel_l1_vs_bf16", "mel_cos_vs_bf16"):
        values = {t: [r["metrics"][t][metric] for r in rows if metric in r["metrics"][t]] for t in ("nvfp4_s0",
                                                                                                    "bf16_s1")}
        if values["nvfp4_s0"]:
            pairwise[metric] = {t: float(np.mean(v)) for t, v in values.items() if v}
    by_case = {case: {m: float(np.mean([r["metrics"]["nvfp4_s0"].get(m, np.nan) - r["metrics"]["bf16_s0"].get(m, np.nan)
                                        for r in rows if r["case"] == case])) for m in REF_FIDELITY}
               for case in sorted({r["case"] for r in rows})}
    result = {"rows": len(rows), "missing_required_metrics": missing, "resolutions": args.gate_resolutions, "gate_t4_passed": passed, "metrics": table,
              "t3_pairwise_vs_bf16_s0": pairwise, "ref_fidelity_delta_by_case": by_case,
              "not_measured": ["ArcFace", "VisionReward", "MUSIQ", "motion smoothness", "AV sync",
                               "<Audio 1> log-mel (no full_scene_audio_image_reference rows on NVL)"]}
    t1_path = out / "t1_report.json"
    if t1_path.exists():
        gate_t1 = json.loads(t1_path.read_text()).get("gate_t1", {})
        result["gate_t1"] = gate_t1
        result["gate_a_passed"] = bool(passed and gate_t1.get("passed", False))
    (out / "gate_a_report.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"gate_t4_passed": passed, "gate_a_passed": result.get("gate_a_passed")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("score", "report"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="composed OmniRef model dir (VAE decode)")
    parser.add_argument("--bf16-dir")
    parser.add_argument("--nvfp4-dir")
    parser.add_argument("--models-dir", help="dinov2-base, clip-vit-large-patch14, aesthetic/")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--sbs-per-shard", type=int, default=3)
    parser.add_argument("--gate-resolutions", nargs="+", default=["480p"])
    add_plan_args(parser)
    args = parser.parse_args()
    {"score": score, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
