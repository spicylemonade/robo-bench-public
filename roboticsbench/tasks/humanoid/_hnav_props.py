"""Humanoid-task camera helpers (the ones humanoid/ninja_course uses)."""

from __future__ import annotations

from typing import Sequence

import mujoco

from roboticsbench.core.geometry import lookat_xyaxes


def tracking_camera(scene, name: str, offset: Sequence[float], look_offset: Sequence[float], *,
                    fovy: float = 50.0, body: str = "pelvis") -> None:
    """Camera that follows ``body`` at a fixed world-frame offset without rotating with it
    (MuJoCo 'track' mode). Offsets are relative to the body's initial (qpos0) position."""
    b = scene.spec.body(body)
    cam = b.add_camera(name=name, pos=list(offset), xyaxes=lookat_xyaxes(offset, look_offset), fovy=fovy)
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
