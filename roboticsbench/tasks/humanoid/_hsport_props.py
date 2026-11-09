"""Humanoid-course props and helpers (the ones humanoid/ninja_course uses)."""

from __future__ import annotations

import math
import zlib
from typing import Sequence

import mujoco
import numpy as np

from roboticsbench.core import visuals as V
from roboticsbench.core.geometry import quat_yaw


GEOM = mujoco.mjtGeom

BOX, CYL, SPH, CAP, ELL, MESH = (GEOM.mjGEOM_BOX, GEOM.mjGEOM_CYLINDER, GEOM.mjGEOM_SPHERE, GEOM.mjGEOM_CAPSULE,
                                 GEOM.mjGEOM_ELLIPSOID, GEOM.mjGEOM_MESH)

WATER_TOP = 0.003  # water surfaces (fail on contact) sit just above the floor


def yawq(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def deco(scene, gtype, pos, size, material, *, quat=None, body=None, name=None, mesh=None, fromto=None):
    """Visual-only geom (never collides, no mass)."""
    parent = body if body is not None else scene.spec.worldbody
    if gtype == MESH:
        return V.visual_geom(scene, parent, gtype=gtype, mesh=mesh, pos=pos, quat=quat, material=material, name=name)
    return V.visual_geom(scene, parent, gtype=gtype, size=size, pos=pos, quat=quat, material=material, name=name,
                         fromto=fromto)


def custom_texture(scene, name: str, img: np.ndarray) -> str:
    """Add (once) an RGB texture from an HxWx3 uint8 image."""
    if not any(t.name == name for t in scene.spec.textures):
        h, w = img.shape[:2]
        t = scene.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
        t.data = np.ascontiguousarray(img.astype(np.uint8)).tobytes()
    return name


def _mat(scene, name: str, **kw) -> str:
    if not any(m.name == name for m in scene.spec.materials):
        scene.spec.add_material(name=name, **kw)
    return name


def water_material(scene) -> str:
    """Glossy blue-green water with soft ripples."""
    if not any(m.name == "hs_water" for m in scene.spec.materials):
        rng = np.random.default_rng(zlib.crc32(b"hs_water"))
        n = 256
        t = 0.65 * V.fbm(n, n, 6, 6, 4, rng) + 0.35 * V.fbm(n, n, 24, 3, 3, rng)
        c0, c1 = np.array([18, 74, 104], float), np.array([60, 146, 170], float)
        img = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
        custom_texture(scene, "hs_water_tex", np.clip(img, 0, 255))
        _mat(scene, "hs_water", textures=["", "hs_water_tex"], texuniform=True, texrepeat=[0.5, 0.5],
             rgba=[1, 1, 1, 1], specular=0.9, shininess=0.95, reflectance=0.28)
    return "hs_water"


def sign(scene, name: str, xy: Sequence[float], facing_deg: float, lines, *, width: float = 0.9,
         height: float = 0.42, z: float = 1.25, post: bool = True, bg=(250, 246, 232), fg=(24, 24, 30),
         font_px: int = 60, frame: str = "black_metal") -> None:
    """Visual-only sign board whose readable face looks toward ``facing_deg`` (the direction from the board to
    its reader), on one or two posts. (Box text maps readably onto the local -y face.)"""
    yaw = math.radians(facing_deg + 90.0)
    q = yawq(yaw)
    mat = V.text_material(scene, name, lines, size=(512, int(512 * height / width)), bg=bg, fg=fg, font_px=font_px)
    deco(scene, BOX, [xy[0], xy[1], z], [width / 2, 0.012, height / 2], mat, quat=q, name=f"{name}_board")
    fm = V.material(scene, frame)
    c, s = math.cos(yaw), math.sin(yaw)
    # thin frame + posts
    deco(scene, BOX, [xy[0] - 0.014 * s, xy[1] + 0.014 * c, z], [width / 2 + 0.02, 0.006, height / 2 + 0.02], fm,
         quat=q)
    if post:
        for sgn in (-1, 1):
            px, py = xy[0] + sgn * (width / 2 - 0.05) * c, xy[1] + sgn * (width / 2 - 0.05) * s
            deco(scene, CYL, [px - 0.04 * s, py + 0.04 * c, (z + height / 2) / 2], [0.022, (z + height / 2) / 2], fm)


def floor_disc(scene, name: str, xy, radius: float, color: str = "green", *, ring: str = "white",
               z: float = 0.003) -> None:
    """Opaque painted disc with an outline ring (visual only)."""
    deco(scene, CYL, [xy[0], xy[1], z], [radius + 0.04, 0.0015], V.material(scene, f"matte_{ring}"),
         name=f"{name}_ring")
    deco(scene, CYL, [xy[0], xy[1], z + 0.001], [radius, 0.0015], V.material(scene, f"glow_{color}", emission=0.25),
         name=name)


def finish_arch(scene, name: str, x: float, y0: float, y1: float, *, height: float = 2.3, text="FINISH") -> None:
    """Visual-only finish gantry: two posts and a banner across the course at ``x`` (readable from -x)."""
    fm = V.material(scene, "metal_red")
    for yy in (y0, y1):
        deco(scene, CYL, [x, yy, height / 2], [0.06, height / 2], fm)
        deco(scene, CYL, [x, yy, 0.02], [0.14, 0.02], V.material(scene, "black_metal"))
    w = abs(y1 - y0)
    mat = V.text_material(scene, f"{name}_banner", [text], size=(1024, 128), bg=(200, 30, 30), fg=(250, 250, 250),
                          font_px=96)
    deco(scene, BOX, [x, (y0 + y1) / 2, height - 0.2], [w / 2, 0.012, 0.2], mat, quat=yawq(-math.pi / 2),
         name=f"{name}_banner")


def water_box(scene, name: str, x0: float, x1: float, y0: float, y1: float, *, top: float = WATER_TOP):
    """Static water surface (collides: touching it is the failure condition), rendered with the water material."""
    depth = 0.05
    return scene.static_box(name, [(x0 + x1) / 2, (y0 + y1) / 2, top - depth / 2],
                            [(x1 - x0) / 2, (y1 - y0) / 2, depth / 2], material=water_material(scene))


def ramp_slab(scene, name: str, foot_xy, yaw: float, run: float, width: float, rise: float, *, thickness=0.1,
              material: str | None = None):
    """Static inclined slab: top surface rises from 0 at the foot edge (centre ``foot_xy``) to ``rise`` ``run``
    metres further along ``yaw``. Returns (geom, quat)."""
    from roboticsbench.core.geometry import quat_axis_angle, quat_mul

    a = math.atan2(rise, run)
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    top_mid = np.array([foot_xy[0], foot_xy[1], 0.0]) + u * run / 2 + np.array([0, 0, rise / 2])
    n = -math.sin(a) * u + math.cos(a) * np.array([0.0, 0.0, 1.0])
    c = top_mid - n * thickness / 2
    q = quat_mul(quat_yaw(yaw), quat_axis_angle([0, 1, 0], -a))
    L = math.hypot(run, rise)
    g = scene._add_geom(scene.spec.worldbody, gtype=BOX, size=[L / 2, width / 2, thickness / 2], pos=c.tolist(),
                        quat=q.tolist(), name=name, material=material or V.material(scene, "oak"),
                        friction=[1.0, 0.005, 0.0001])
    return g, q
