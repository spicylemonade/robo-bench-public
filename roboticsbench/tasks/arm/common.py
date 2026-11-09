"""Shared table-top scene for Panda tasks.

World frame: robot base at the origin, table surface at z = 0, +x points away from the robot
(the "front" camera looks back at the robot from +x). The comfortable workspace is roughly
x in [0.3, 0.75], y in [-0.35, 0.35].
"""

from __future__ import annotations

from typing import Any, ClassVar

import mujoco
import numpy as np

from roboticsbench.core.scene import SceneBuilder
from roboticsbench.tasks.base import Task

TABLE_X = (-0.35, 1.05)
TABLE_Y = (-0.65, 0.65)
TABLE_TOP = 0.0
TABLE_H = 0.75


class ArmTask(Task):
    robot: ClassVar[str] = "panda"
    world: ClassVar[dict[str, Any]] = {"floor_z": -TABLE_H, "floor": "grid"}
    cameras: ClassVar[tuple[str, ...]] = ("front", "top")
    video_camera: ClassVar[str] = "front"
    time_limit: ClassVar[float] = 150.0
    max_actions: ClassVar[int] = 50

    def build(self, scene: SceneBuilder, rng: np.random.Generator) -> None:
        build_table(scene)

    def settle_objects(self) -> bool:
        return True


def build_table(scene: SceneBuilder, color: str = "wood") -> None:
    """Workbench with a 10 cm measuring grid (helps agents estimate positions from images), dressed as a lab."""
    from roboticsbench.core import visuals as V
    from roboticsbench.core.environments import lab

    cx, cy = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2
    hx, hy = (TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2
    scene.static_box("table", [cx, cy, TABLE_TOP - 0.02], [hx, hy, 0.02], material=V.material(scene, "worktop_grid"),
                     friction=[0.8, 0.005, 0.0001])
    lab(scene, TABLE_X, TABLE_Y, TABLE_H)
    # world axes glyph at the robot base: red = +x, green = +y
    scene.marker("axis_x", [0.16, -0.18, 0.0008], shape="box", size=(0.06, 0.004, 0.0008), color="red", alpha=0.9)
    scene.marker("axis_y", [0.10, -0.12, 0.0008], shape="box", size=(0.004, 0.06, 0.0008), color="green", alpha=0.9)
    scene.add_camera("front", pos=[1.75, 0.0, 0.78], target=[0.38, 0.0, 0.06], fovy=45)
    scene.add_camera("top", pos=[0.45, 0.0, 1.55], target=[0.45, 0.0, 0.0], fovy=45, up=(1.0, 0.0, 0.0))
    scene.add_camera("side", pos=[0.45, -1.35, 0.55], target=[0.45, 0.0, 0.04], fovy=48)
    scene.add_camera("overview", pos=[1.55, -1.25, 1.05], target=[0.35, 0.0, 0.05], fovy=48)


def on_table(p, margin: float = 0.0) -> bool:
    return (TABLE_X[0] + margin <= p[0] <= TABLE_X[1] - margin and
            TABLE_Y[0] + margin <= p[1] <= TABLE_Y[1] - margin and p[2] > TABLE_TOP - 0.05)


def sample_xy(rng: np.random.Generator, n: int, *, x=(0.35, 0.68), y=(-0.3, 0.3), min_dist: float = 0.1,
              avoid: list[tuple[float, float, float]] | None = None, tries: int = 2000) -> list[np.ndarray]:
    """Rejection-sample ``n`` table positions at least ``min_dist`` apart (and away from ``avoid``
    circles given as (x, y, radius))."""
    pts: list[np.ndarray] = []
    avoid = avoid or []
    for _ in range(tries):
        if len(pts) == n:
            break
        p = np.array([rng.uniform(*x), rng.uniform(*y)])
        if all(np.linalg.norm(p - q) >= min_dist for q in pts) and \
                all(np.hypot(p[0] - a[0], p[1] - a[1]) >= a[2] for a in avoid):
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError("could not place objects")
    return pts
