"""Environment kits: dressed, good-looking surroundings for the task areas.

* :func:`lab` - the Panda's workbench in a robotics lab (grid work surface, steel frame, pegboard wall, shelves).
* :func:`dress_hall` - humanoid venues around a task's footprint: ``warehouse``, ``gym``, ``park``, ``house``.
  Called by :class:`~roboticsbench.core.env.Env` after ``task.build`` so decor never intrudes on the play area.

All decor is visual only (no collision, no mass) except the floor.
"""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

from roboticsbench.core import visuals as V

GEOM = mujoco.mjtGeom
BOX, CYL, SPH, CAP, ELL = (GEOM.mjGEOM_BOX, GEOM.mjGEOM_CYLINDER, GEOM.mjGEOM_SPHERE, GEOM.mjGEOM_CAPSULE,
                           GEOM.mjGEOM_ELLIPSOID)


def _deco(scene, gtype, pos, size, material, quat=None, name=None, body=None):
    return V.visual_geom(scene, body if body is not None else scene.spec.worldbody, gtype=gtype, size=size, pos=pos,
                         quat=quat, material=material, name=name)


def _yawq(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


# ============================================================================== arm lab
def lab(scene, table_x=(-0.35, 1.05), table_y=(-0.65, 0.65), table_h: float = 0.75) -> None:
    """Robotics-lab dressing around the Panda workbench (the table collision box is created by the caller)."""
    V.render_quality(scene)
    V.lights(scene, "studio", center=(0.35, 0.0, 0.0))
    fz = -table_h
    x0, x1 = table_x
    y0, y1 = table_y
    steel, dark = V.material(scene, "steel"), V.material(scene, "steel_dark")
    # table frame: edge banding, legs, rails and a lower shelf
    band = V.material(scene, "walnut")
    for (cx, cy, hx, hy) in (((x0 + x1) / 2, y0, (x1 - x0) / 2 + 0.01, 0.01), ((x0 + x1) / 2, y1, (x1 - x0) / 2 + 0.01, 0.01),
                             (x0, 0.0, 0.01, (y1 - y0) / 2), (x1, 0.0, 0.01, (y1 - y0) / 2)):
        _deco(scene, BOX, [cx, cy, -0.02], [hx, hy, 0.021], band)
    for sx in (x0 + 0.06, x1 - 0.06):
        for sy in (y0 + 0.06, y1 - 0.06):
            _deco(scene, BOX, [sx, sy, fz / 2 - 0.02], [0.025, 0.025, table_h / 2 - 0.02], steel)
            _deco(scene, CYL, [sx, sy, fz + 0.01], [0.035, 0.01], V.material(scene, "rubber_black"))
    for sy in (y0 + 0.06, y1 - 0.06):
        _deco(scene, BOX, [(x0 + x1) / 2, sy, -0.08], [(x1 - x0) / 2 - 0.06, 0.018, 0.035], dark)
    _deco(scene, BOX, [(x0 + x1) / 2, 0.0, fz + 0.18], [(x1 - x0) / 2 - 0.08, (y1 - y0) / 2 - 0.08, 0.012],
          V.material(scene, "plywood"))
    for k, (bx, by) in enumerate(((0.2, -0.3), (0.55, 0.25), (0.8, -0.1))):
        _deco(scene, BOX, [bx, by, fz + 0.26], [0.14, 0.11, 0.07], V.material(scene, "cardboard", seed=k))
    # robot mount plate
    _deco(scene, CYL, [0.0, 0.0, 0.002], [0.13, 0.002], dark)
    # room: floor material, walls
    floor = next((g for g in scene.spec.worldbody.geoms if g.name == "floor"), None)
    if floor is not None:
        floor.material = V.material(scene, "tiles_lab")
    wall = V.material(scene, "paint_white")
    _deco(scene, BOX, [-1.7, 0.0, fz + 1.6], [0.05, 3.2, 1.6], wall)  # back wall (behind the robot)
    _deco(scene, BOX, [0.8, 3.2, fz + 1.6], [2.6, 0.05, 1.6], wall)
    _deco(scene, BOX, [0.8, -3.2, fz + 1.6], [2.6, 0.05, 1.6], wall)
    _deco(scene, BOX, [3.4, 0.0, fz + 1.6], [0.05, 3.2, 1.6], V.material(scene, "paint_grey"))
    for yy in (-3.14, 3.14):  # skirting boards
        _deco(scene, BOX, [0.8, yy, fz + 0.05], [2.6, 0.012, 0.05], dark)
    _deco(scene, BOX, [-1.64, 0.0, fz + 0.05], [0.012, 3.2, 0.05], dark)
    # pegboard with tools on the back wall
    _deco(scene, BOX, [-1.63, -0.9, fz + 1.35], [0.015, 0.8, 0.45], V.material(scene, "pegboard"))
    for i, (ty, tz, col) in enumerate(((-1.4, 1.5, "red"), (-1.15, 1.35, "blue"), (-0.85, 1.5, "yellow"),
                                       (-0.55, 1.3, "orange"))):
        _deco(scene, BOX, [-1.6, ty, fz + tz], [0.012, 0.018, 0.12], V.material(scene, f"plastic_{col}"))
        _deco(scene, CYL, [-1.6, ty, fz + tz - 0.2], [0.006, 0.08], V.material(scene, "chrome"))
    # shelves with parts bins
    for level, z in enumerate((0.9, 1.3, 1.7)):
        _deco(scene, BOX, [-1.5, 1.1, fz + z], [0.18, 0.9, 0.015], V.material(scene, "maple"))
        for k in range(4):
            col = ("blue", "red", "yellow", "green")[(k + level) % 4]
            _deco(scene, BOX, [-1.5, 0.4 + 0.42 * k, fz + z + 0.07], [0.13, 0.17, 0.055], V.material(scene, f"plastic_{col}"))
    for sy in (0.2, 2.0):
        _deco(scene, BOX, [-1.5, sy, fz + 1.0], [0.18, 0.015, 1.0], steel)
    # side desk with a monitor (visible in the side / overview cameras)
    _deco(scene, BOX, [0.5, 2.7, fz + 0.74], [0.7, 0.35, 0.02], V.material(scene, "oak"))
    _deco(scene, BOX, [0.5, 2.95, fz + 1.02], [0.32, 0.02, 0.19], V.material(scene, "black_metal"))
    _deco(scene, BOX, [0.5, 2.93, fz + 1.02], [0.3, 0.004, 0.17], V.material(scene, "glow_dark_blue"))
    for sx in (-0.15, 1.15):
        _deco(scene, BOX, [sx, 2.7, fz + 0.37], [0.03, 0.3, 0.37], dark)
    # ceiling light panels
    for lx in (-0.5, 0.8, 2.1):
        _deco(scene, BOX, [lx, 0.0, fz + 3.15], [0.5, 0.25, 0.02], V.material(scene, "glow_white"))


# ============================================================================== humanoid venues
def content_bbox(scene, margin: float = 1.5) -> tuple[float, float, float, float]:
    """xy bounding box of everything the task placed in the world (static geoms and bodies), plus a margin."""
    xs, ys = [0.0], [0.0]
    for g in scene.spec.worldbody.geoms:
        if g.name == "floor" or g.type == GEOM.mjGEOM_PLANE:
            continue
        r = float(max(g.size[0], g.size[1] if g.type == BOX else g.size[0]))
        xs += [g.pos[0] - r, g.pos[0] + r]
        ys += [g.pos[1] - r, g.pos[1] + r]
    for b in scene.spec.worldbody.bodies:
        if b.name in ("pelvis", "link0"):
            continue
        xs.append(float(b.pos[0]))
        ys.append(float(b.pos[1]))
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


FLOORS = {"warehouse": "concrete_grid", "gym": "rubber_grid", "park": "grass_grid", "house": "floor_wood_grid",
          "arena": "concrete_grid"}


def dress_hall(scene, style: str = "warehouse") -> None:
    """Dress the surroundings of the task footprint (call after the task has built its layout)."""
    V.render_quality(scene, haze=[0.78, 0.82, 0.88, 1.0])
    x0, x1, y0, y1 = content_bbox(scene, margin=2.5)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ext = max(x1 - x0, y1 - y0)
    floor = next((g for g in scene.spec.worldbody.geoms if g.name == "floor"), None)
    if floor is not None:
        floor.material = V.material(scene, FLOORS.get(style, "concrete_grid"))
    if style == "park":
        V.lights(scene, "outdoor", center=(cx, cy, 0.0), extent=ext)
        _park(scene, x0, x1, y0, y1)
        return
    V.lights(scene, "hall", center=(cx, cy, 0.0), extent=ext)
    wall_mat = {"warehouse": "paint_grey", "gym": "paint_blue", "house": "paint_white", "arena": "paint_grey"}[style]
    h = 4.5 if style in ("warehouse", "arena") else 3.0
    wall = V.material(scene, wall_mat)
    for (px, py, hx, hy) in (((x0 + x1) / 2, y0 - 0.1, (x1 - x0) / 2 + 0.2, 0.1),
                             ((x0 + x1) / 2, y1 + 0.1, (x1 - x0) / 2 + 0.2, 0.1),
                             (x0 - 0.1, (y0 + y1) / 2, 0.1, (y1 - y0) / 2 + 0.2),
                             (x1 + 0.1, (y0 + y1) / 2, 0.1, (y1 - y0) / 2 + 0.2)):
        _deco(scene, BOX, [px, py, h / 2], [hx, hy, h / 2], wall)
    if style == "warehouse":
        _warehouse(scene, x0, x1, y0, y1, h)
    elif style == "gym":
        _gym(scene, x0, x1, y0, y1)
    elif style == "house":
        _house(scene, x0, x1, y0, y1)


def _warehouse(scene, x0, x1, y0, y1, h):
    steel = V.material(scene, "steel_dark")
    orange = V.material(scene, "metal_orange")
    boxes = [V.material(scene, "cardboard", seed=k) for k in range(3)]
    haz = V.material(scene, "hazard")
    # pallet racking along the long walls
    for side, yy in ((-1, y0 + 0.6), (1, y1 - 0.6)):
        x = x0 + 1.0
        k = 0
        while x < x1 - 1.0:
            for dx in (-1.2, 1.2):
                _deco(scene, BOX, [x + dx, yy, 1.6], [0.04, 0.45, 1.6], orange)
            for z in (0.15, 1.2, 2.3):
                _deco(scene, BOX, [x, yy, z], [1.25, 0.46, 0.03], steel)
                for j in range(3):
                    if (k + j) % 4 == 3:
                        continue
                    bw = 0.26 + 0.06 * ((k + j) % 3)
                    _deco(scene, BOX, [x - 0.8 + 0.8 * j, yy, z + 0.03 + bw], [0.3, 0.32, bw], boxes[(k + j) % 3])
                k += 1
            x += 2.6
    # pillars with hazard wraps near the corners
    for (px, py) in ((x0 + 0.4, y0 + 0.4), (x1 - 0.4, y0 + 0.4), (x0 + 0.4, y1 - 0.4), (x1 - 0.4, y1 - 0.4)):
        _deco(scene, BOX, [px, py, h / 2], [0.2, 0.2, h / 2], V.material(scene, "concrete"))
        _deco(scene, BOX, [px, py, 0.5], [0.205, 0.205, 0.5], haz)
    # roof trusses + light fittings
    xx = x0
    while xx <= x1:
        _deco(scene, BOX, [xx, (y0 + y1) / 2, h - 0.1], [0.06, (y1 - y0) / 2, 0.08], steel)
        _deco(scene, BOX, [xx, (y0 + y1) / 2, h - 0.3], [0.08, 0.6, 0.03], V.material(scene, "glow_white"))
        xx += 4.0


def _gym(scene, x0, x1, y0, y1):
    pad = V.material(scene, "plastic_dark_blue")
    for side, yy in ((-1, y0 + 0.08), (1, y1 - 0.08)):
        x = x0 + 0.6
        while x < x1 - 0.6:
            _deco(scene, BOX, [x, yy, 1.0], [0.55, 0.06, 0.9], pad)
            x += 1.2
    for k, (px, col) in enumerate(((x0 + 1.5, "red"), (x1 - 1.5, "yellow"))):
        _deco(scene, BOX, [px, y1 - 0.2, 2.4], [0.9, 0.02, 0.35], V.material(scene, f"plastic_{col}"))
    for i in range(3):  # stacked mats in a corner
        _deco(scene, BOX, [x1 - 1.0, y0 + 1.0, 0.06 + 0.12 * i], [0.7, 0.5, 0.06], V.material(scene, "plastic_blue"))


def _house(scene, x0, x1, y0, y1):
    wood = V.material(scene, "walnut")
    fabric = V.material(scene, "felt_blue")
    _deco(scene, BOX, [x0 + 1.0, y1 - 0.7, 0.25], [0.9, 0.4, 0.25], fabric)  # sofa
    _deco(scene, BOX, [x0 + 1.0, y1 - 0.35, 0.55], [0.9, 0.08, 0.3], fabric)
    _deco(scene, BOX, [x1 - 1.0, y1 - 0.5, 0.4], [0.6, 0.3, 0.02], wood)  # side table
    for dx in (-0.5, 0.5):
        _deco(scene, BOX, [x1 - 1.0 + dx, y1 - 0.5, 0.2], [0.03, 0.25, 0.2], wood)
    _deco(scene, CYL, [x1 - 1.0, y1 - 0.5, 0.55], [0.08, 0.13], V.material(scene, "ceramic_white"))
    _deco(scene, ELL, [x1 - 1.0, y1 - 0.5, 0.82], [0.14, 0.14, 0.18], V.material(scene, "plastic_green"))


def _park(scene, x0, x1, y0, y1):
    bark = V.material(scene, "walnut")
    leaves = [V.material(scene, "matte_dark_green"), V.material(scene, "matte_green"),
              V.material(scene, "matte_lime")]
    rng = np.random.default_rng(7)
    # trees scattered around the perimeter
    per = 2 * ((x1 - x0) + (y1 - y0))
    n = int(per / 3.0)
    for i in range(n):
        s = per * i / n
        if s < x1 - x0:
            px, py = x0 + s, y0 - 1.0
        elif s < (x1 - x0) + (y1 - y0):
            px, py = x1 + 1.0, y0 + s - (x1 - x0)
        elif s < 2 * (x1 - x0) + (y1 - y0):
            px, py = x1 - (s - (x1 - x0) - (y1 - y0)), y1 + 1.0
        else:
            px, py = x0 - 1.0, y1 - (s - 2 * (x1 - x0) - (y1 - y0))
        px += rng.uniform(-0.5, 0.5)
        py += rng.uniform(-0.5, 0.5)
        th = rng.uniform(1.6, 2.6)
        _deco(scene, CYL, [px, py, th / 2], [0.09, th / 2], bark)
        r = rng.uniform(0.8, 1.3)
        _deco(scene, ELL, [px, py, th + r * 0.6], [r, r, r * 0.9], leaves[i % 3])
    # a low fence line just outside the footprint
    wood = V.material(scene, "oak")
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1)):
        L = math.hypot(bx - ax, by - ay)
        _deco(scene, BOX, [(ax + bx) / 2, ay, 0.45], [L / 2, 0.02, 0.04], wood)
        k = 0.0
        while k <= L:
            _deco(scene, BOX, [ax + k, ay, 0.3], [0.03, 0.03, 0.3], wood)
            k += 2.0
