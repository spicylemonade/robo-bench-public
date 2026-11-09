"""Humanoid-task structures and helpers (the ones humanoid/ninja_course uses)."""

from __future__ import annotations

import mujoco

from roboticsbench.core import visuals as V


GEOM = mujoco.mjtGeom

BOX, CYL, SPH, CAP = GEOM.mjGEOM_BOX, GEOM.mjGEOM_CYLINDER, GEOM.mjGEOM_SPHERE, GEOM.mjGEOM_CAPSULE


def _vg(scene, body, gtype, pos, size, mat, quat=None, name=None):
    return V.visual_geom(scene, body, gtype=gtype, size=size, pos=pos, quat=quat, material=mat, name=name)


def vent_duct(scene, name: str, x0: float, length: float, yc: float, *, inner_w: float = 1.0, roof: float = 1.21,
              wall_t: float = 0.06, roof_t: float = 0.06) -> list[str]:
    """Low rectangular ventilation duct along +x from ``x0``: two side walls and a roof whose underside is ``roof``
    above the floor (collision boxes), galvanised-metal look with seam bands and a black entrance frame. Returns
    the collision geom names (for contact checks)."""
    names = []
    alu = V.material(scene, "aluminum")
    zc = (roof + roof_t) / 2
    for side, sgn in (("r", -1), ("l", 1)):
        nm = f"{name}_wall_{side}"
        scene.static_box(nm, [x0 + length / 2, yc + sgn * (inner_w / 2 + wall_t / 2), zc],
                         [length / 2, wall_t / 2, zc], material=alu)
        names.append(nm)
    nm = f"{name}_roof"
    scene.static_box(nm, [x0 + length / 2, yc, roof + roof_t / 2], [length / 2, inner_w / 2 + wall_t, roof_t / 2],
                     material=alu)
    names.append(nm)
    wb = scene.spec.worldbody
    seam = V.material(scene, "steel_dark")
    n = max(2, int(length / 0.4))
    for k in range(n + 1):
        xx = x0 + length * k / n
        _vg(scene, wb, BOX, [xx, yc, roof + roof_t + 0.005], [0.012, inner_w / 2 + wall_t + 0.01, 0.006], seam)
        for sgn in (-1, 1):
            _vg(scene, wb, BOX, [xx, yc + sgn * (inner_w / 2 + wall_t + 0.005), zc], [0.012, 0.006, zc], seam)
    blk = V.material(scene, "hazard")
    for xx in (x0, x0 + length):  # hazard-striped entrance / exit frames
        _vg(scene, wb, BOX, [xx, yc, roof + roof_t / 2], [0.02, inner_w / 2 + wall_t + 0.02, roof_t / 2 + 0.02], blk)
        for sgn in (-1, 1):
            _vg(scene, wb, BOX, [xx, yc + sgn * (inner_w / 2 + wall_t / 2), zc], [0.02, wall_t / 2 + 0.02, zc], blk)
    return names


def vwalk(api, x: float, y: float, *, speed: float = 0.6, tol: float = 0.06, heading: float | None = None,
          timeout: float | None = None) -> None:
    api.walk_to(float(x), float(y), speed=speed, tol=tol, heading=heading, timeout=timeout)
