#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Transcribe a FastH3 speech gate artifact and record an auditable WER result."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import wave
from pathlib import Path

import numpy as np
import torch


def _read_stereo_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if channels != 2 or sample_width != 2:
        raise ValueError(f"Expected stereo int16 WAV, got channels={channels} width={sample_width}")
    stereo = np.frombuffer(frames, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
    return stereo.mean(axis=1), sample_rate


def _resample_for_whisper(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio
    if sample_rate == 32000:
        usable = len(audio) // 2 * 2
        return audio[:usable].reshape(-1, 2).mean(axis=1)
    raise ValueError(f"Unsupported ASR input sample rate: {sample_rate}")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_error_rate(reference: str, hypothesis: str) -> tuple[float, int]:
    ref = _words(reference)
    hyp = _words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for ref_word in ref:
        current = [previous[0] + 1]
        for index, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[index] + 1,
                previous[index - 1] + (ref_word != hyp_word),
            ))
        previous = current
    edits = previous[-1]
    return edits / max(1, len(ref)), edits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="h0-vsa-sync-asr-review")
    parser.add_argument("--wandb-project", default="fasth3-14b-2step-qad-sprint")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--max-wer", type=float, default=0.25)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for the ASR gate")

    from huggingface_hub import HfApi
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import wandb

    audio, source_sample_rate = _read_stereo_wav(args.wav)
    audio_16k = _resample_for_whisper(audio, source_sample_rate)
    revision = str(HfApi().model_info(args.asr_model, token=os.environ.get("HF_TOKEN")).sha)
    dtype = torch.float16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.asr_model, revision=revision)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0,
    )
    started = time.time()
    result = transcriber(
        {"raw": audio_16k, "sampling_rate": 16000},
        return_timestamps=True,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    transcript = str(result["text"]).strip()
    wer, edits = _word_error_rate(args.expected, transcript)
    passed = wer <= args.max_wer
    payload = {
        "wav": str(args.wav.resolve()),
        "source_sample_rate": source_sample_rate,
        "asr_sample_rate": 16000,
        "duration_seconds": len(audio) / source_sample_rate,
        "expected": args.expected,
        "transcript": transcript,
        "word_error_rate": wer,
        "word_edits": edits,
        "max_word_error_rate": args.max_wer,
        "passed": passed,
        "chunks": result.get("chunks", []),
        "asr_model": args.asr_model,
        "asr_revision": revision,
        "completed_at_unix": time.time(),
        "wall_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="evaluation",
        config={
            "asr_model": args.asr_model,
            "asr_revision": revision,
            "expected": args.expected,
            "max_word_error_rate": args.max_wer,
            "persistent_wav": str(args.wav),
        },
    )
    wandb.log({"gate/asr_wer": wer, "gate/asr_pass": int(passed)})
    artifact = wandb.Artifact(f"{args.run_id}-result", type="evaluation-manifest", metadata=payload)
    artifact.add_file(str(args.output))
    run.log_artifact(artifact)
    run.summary.update({
        "transcript": transcript,
        "word_error_rate": wer,
        "passed": passed,
        "persistent_manifest": str(args.output),
    })
    run.finish()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(f"ASR gate failed: WER {wer:.3f} exceeds {args.max_wer:.3f}")


if __name__ == "__main__":
    main()
