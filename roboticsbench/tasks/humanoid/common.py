"""Shared helpers for G1 humanoid tasks."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from roboticsbench.tasks.base import Task


class HumanoidTask(Task):
    robot: ClassVar[str] = "g1"
    world: ClassVar[dict[str, Any]] = {"floor": "grid", "style": "warehouse"}
    cameras: ClassVar[tuple[str, ...]] = ("head", "chase")
    video_camera: ClassVar[str] = "chase"
    time_limit: ClassVar[float] = 120.0
    max_actions: ClassVar[int] = 40
    success_hold: ClassVar[float] = 0.2
    ends_on_failure: ClassVar[bool] = False  # the agent keeps the full time: it may get up after a fall

    def robot_xy(self, env) -> np.ndarray:
        return env.robot.base_pos[:2]


def robot_touching(env, geom_ids: set[int]) -> bool:
    """True if any robot geom is in contact with any geom in ``geom_ids``."""
    d = env.data
    rg = env.robot.geom_ids
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in rg and c.geom2 in geom_ids) or (c.geom2 in rg and c.geom1 in geom_ids):
            return True
    return False


def feet_touching(env, geom_ids: set[int]) -> bool:
    d = env.data
    fg = env.robot.foot_geoms
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in fg and c.geom2 in geom_ids) or (c.geom2 in fg and c.geom1 in geom_ids):
            return True
    return False


def geoms_with_prefix(model, prefix: str) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        n = model.geom(g).name
        if n and n.startswith(prefix):
            out.add(g)
    return out


def segment_crossed(p_prev, p_now, a, b) -> bool:
    """Did the robot's xy path p_prev->p_now cross the gate segment a->b?"""
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = cross(a, b, p_prev), cross(a, b, p_now)
    d3, d4 = cross(p_prev, p_now, a), cross(p_prev, p_now, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def heading_to(src, dst) -> float:
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))
