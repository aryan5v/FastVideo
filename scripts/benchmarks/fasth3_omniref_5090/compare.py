"""Create native-resolution comparisons with separate original AAC tracks."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path, help="Benchmark output containing v2-{36L,8L} and baseline-{36L,8L}")
root = parser.parse_args().root
destination = root / "comparisons"
destination.mkdir(exist_ok=True)
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
validation = {}


def inspect(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,nb_frames,duration,sample_rate,channels:stream_tags=title,handler_name",
        "-of", "json", str(path)], text=True))
    video = [s for s in data["streams"] if s["codec_type"] == "video"]
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    assert len(video) == 1 and int(video[0]["nb_frames"]) == 124, path
    assert video[0]["height"] == 480 and video[0]["width"] in (832, 1664), path
    assert len(audio) == (2 if video[0]["width"] == 1664 else 1), path
    for stream in audio:
        assert stream["channels"] == 2 and stream["sample_rate"] == "32000", path
    data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(audio) == 1:
        process = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
            "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, check=True)
        data["volume_dbfs"] = dict(re.findall(r"(mean_volume|max_volume): ([-\d.]+) dB", process.stderr))
        assert float(data["volume_dbfs"]["max_volume"]) > -60, path
        pcm = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-acodec",
            "pcm_f32le", "-ac", "2", "-ar", "32000", "-"])
        samples = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2)
        assert np.isfinite(samples).all(), path
        rms = np.sqrt(np.mean(samples.astype(np.float64)**2, axis=0))
        window = 3200
        windows = samples[:len(samples) // window * window].reshape(-1, window, 2)
        data["audio_signal"] = {
            "decoded_seconds": len(samples) / 32000,
            "rms_dbfs_channels": (20 * np.log10(np.maximum(rms, 1e-15))).tolist(),
            "peak_abs": float(np.abs(samples).max()),
            "fraction_samples_at_or_above_full_scale": float(np.mean(np.abs(samples) >= 1.0)),
            "fraction_100ms_windows_below_minus60_dbfs": float(
                np.mean(np.sqrt(np.mean(windows.astype(np.float64)**2, axis=(1, 2))) < 0.001)),
            "dc_channels": np.mean(samples.astype(np.float64), axis=0).tolist(),
        }
        assert abs(len(samples) / 32000 - float(video[0]["duration"])) < 0.1, path
        loudness = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af",
            "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True, check=True)
        data["loudness_summary"] = loudness.stderr.split("Summary:")[-1].strip()
    return data


for decoder in ("36L", "8L"):
    for case in ("speaker", "dance"):
        previous = root / f"baseline-{decoder}" / f"{case}_0.mp4"
        current = root / f"v2-{decoder}" / f"{case}_2.mp4"
        target = destination / f"compare_{case}_{decoder}.mp4"
        filters = (
            f"[0:v]drawtext=fontfile={font}:text='Previous QAD':x=12:y=12:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.6[left];"
            f"[1:v]drawtext=fontfile={font}:text='QAD v2':x=12:y=12:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.6[right];[left][right]hstack=inputs=2[video]"
        )
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(previous), "-i", str(current),
            "-filter_complex", filters, "-map", "[video]", "-map", "0:a:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "4", "-c:a", "copy",
            "-metadata:s:a:0", "handler_name=Previous QAD audio",
            "-metadata:s:a:1", "handler_name=QAD v2 audio",
            "-disposition:a:0", "default", "-disposition:a:1", "0", "-shortest", str(target)], check=True)
        for clip in (previous, current, target):
            validation[str(clip.relative_to(root))] = inspect(clip)
        for frame in (12, 36, 60, 84, 108):
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", str(target),
                "-vf", f"select=eq(n\\,{frame})", "-vsync", "0", "-frames:v", "1",
                str(destination / f"{case}_{decoder}_frame{frame:03d}.png")], check=True)
        print(f"Validated {target.name}: 124 native frames, two original audio streams", flush=True)
(root / "media_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
