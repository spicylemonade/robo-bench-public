"""Paths to vendored robot assets (downloaded by ``python -m roboticsbench.assets.fetch``)."""

from __future__ import annotations

import os
from pathlib import Path

VENDOR = Path(os.environ.get("RBENCH_ASSETS_DIR") or (Path(__file__).resolve().parent / "vendor"))

G1_DIR = VENDOR / "g1"
G1_XML = G1_DIR / "g1_gear_wbc.xml"
G1_POLICY_BALANCE = G1_DIR / "policy" / "GR00T-WholeBodyControl-Balance.onnx"
G1_POLICY_WALK = G1_DIR / "policy" / "GR00T-WholeBodyControl-Walk.onnx"

PANDA_DIR = VENDOR / "panda"
PANDA_XML = PANDA_DIR / "panda.xml"


def ensure_assets() -> None:
    """Download assets on first use (no-op when already present)."""
    from roboticsbench.assets.fetch import assets_present, fetch

    if not assets_present():
        fetch(quiet=True)
