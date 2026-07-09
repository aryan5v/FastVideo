# SPDX-License-Identifier: Apache-2.0
"""Preparation-only parity scaffold for the future Wan2.2-TI2V-5B port.

Coverage scope: both. This becomes a real official-versus-FastVideo component
and pipeline parity test only after PORT_STATUS Q001–Q005 identify the source,
weights, component APIs, and conversion route. Until then its dependency skip
prevents accidental claims that the unimplemented port has parity evidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_wan2_2_ti2v_5b_parity_requires_preparation_handoff() -> None:
    """Make missing official inputs a precise skip rather than false evidence."""
    official_ref = os.environ.get("WAN2_2_TI2V_5B_OFFICIAL_REF_DIR")
    weights = os.environ.get("WAN2_2_TI2V_5B_WEIGHTS")
    if not official_ref or not weights:
        pytest.skip("PORT_STATUS Q001/Q005: official reference and weights are not selected")

    missing = [path for path in (official_ref, weights) if not Path(path).exists()]
    if missing:
        pytest.skip(f"official parity inputs are unavailable: {', '.join(missing)}")

    pytest.skip("PORT_STATUS Q002–Q004: official component call paths are not recorded")
