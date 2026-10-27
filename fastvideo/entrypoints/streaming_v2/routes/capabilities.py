# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/streamv2/capabilities``: what the served model can generate."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()

# Hardcoded for MiniMax-H3 Ref2VA until FastVideo can derive these facts from
# the loaded model. Frame counts follow H3's causal-VAE grid (17 * n + 5) at
# 24 fps; `frame_sizes` is [width, height] by aspect ratio, then resolution.
CAPABILITIES: dict[str, Any] = {
    "model_id": "h3-ref2va",
    "name": "H3 Ref2AV",
    "min_segment_duration_sec": 5,
    "max_segment_duration_sec": 15,
    "max_reference_images": 9,
    "max_reference_aspect_ratio": 4.0,
    "frame_sizes": {
        "16:9": {
            "720p": [1344, 768]
        }
    },
    "num_frames_by_duration_sec": {
        "5": 124,
        "6": 158,
        "7": 175,
        "8": 192,
        "9": 226,
        "10": 243,
        "11": 277,
        "12": 294,
        "13": 328,
        "14": 345,
        "15": 362,
    },
}


@router.get("/v1/streamv2/capabilities")
async def get_capabilities() -> dict[str, Any]:
    return CAPABILITIES
