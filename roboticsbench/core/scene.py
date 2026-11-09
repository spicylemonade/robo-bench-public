"""Procedural scene construction on top of ``mujoco.MjSpec``.

Tasks receive a :class:`SceneBuilder` wrapping the robot's spec and add props with the helpers
below. Every prop is a real rigid body with mass / friction (or a static geom), so all
interaction goes through MuJoCo contact dynamics -- nothing is kinematically faked.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import mujoco
import numpy as np

from roboticsbench.core.geometry import lookat_xyaxes, quat_yaw

GEOM = mujoco.mjtGeom
Vec = Sequence[float]

BUTTON_HOUSING = 0.024  # push_button housing thickness (m)
BUTTON_CAP = 0.024  # push_button cap thickness (m); the cap face sits BUTTON_HOUSING + BUTTON_CAP out at rest

# Distinct, nameable colours (LLM-facing names).
COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.86, 0.14, 0.14, 1.0),
    "green": (0.16, 0.66, 0.24, 1.0),
    "blue": (0.14, 0.34, 0.86, 1.0),
    "yellow": (0.96, 0.80, 0.10, 1.0),
    "orange": (0.97, 0.50, 0.08, 1.0),
    "purple": (0.55, 0.24, 0.76, 1.0),
    "cyan": (0.10, 0.75, 0.85, 1.0),
    "pink": (0.95, 0.45, 0.70, 1.0),
    "white": (0.93, 0.93, 0.93, 1.0),
    "black": (0.08, 0.08, 0.08, 1.0),
    "gray": (0.50, 0.50, 0.52, 1.0),
    "brown": (0.50, 0.32, 0.18, 1.0),
    "wood": (0.58, 0.45, 0.32, 1.0),
    "darkwood": (0.36, 0.25, 0.16, 1.0),
    "metal": (0.62, 0.64, 0.68, 1.0),
    "gold": (0.90, 0.72, 0.20, 1.0),
    "lava": (1.00, 0.30, 0.02, 1.0),
    "concrete": (0.62, 0.62, 0.60, 1.0),
}


# plain colour names used by tasks -> richer visual presets (roboticsbench.core.visuals)
COLOR_PRESETS = {c: f"plastic_{c}" for c in ("red", "green", "blue", "yellow", "orange", "purple", "cyan", "pink",
                                             "white", "black")}
COLOR_PRESETS.update({"gray": "plastic_grey", "brown": "matte_brown", "wood": "oak", "darkwood": "walnut",
                      "metal": "steel", "gold": "brass", "concrete": "concrete"})


def rgba_of(color: str | Vec) -> list[float]:
    if isinstance(color, str):
        return list(COLORS[color])
    c = list(color)
    return c + [1.0] if len(c) == 3 else c


class SceneBuilder:
    """Thin convenience layer over an MjSpec. ``spec`` is exposed for anything custom."""

    def __init__(self, spec: mujoco.MjSpec):
        self.spec = spec
        self._materials: set[str] = {m.name for m in spec.materials}
        self._uid = 0
        self.free_bodies: list[str] = []  # names of bodies with free joints (objects)

    # ------------------------------------------------------------------ helpers
    def uid(self, prefix: str) -> str:
        self._uid += 1
        return f"{prefix}_{self._uid}"

    def material(self, color: str | Vec, *, reflectance: float = 0.0, specular: float = 0.3,
                 shininess: float = 0.3, emission: float = 0.0) -> str:
        if isinstance(color, str) and reflectance == 0.0 and specular == 0.3 and shininess == 0.3:
            from roboticsbench.core import visuals as V

            preset = COLOR_PRESETS.get(color, color)
            if preset in V.MATERIALS or preset.split("_")[0] in ("plastic", "metal", "rubber", "glow", "matte"):
                return V.material(self, preset, emission=emission)
        rgba = rgba_of(color)
        name = "mat_" + ("_".join(f"{c:.3f}" for c in rgba) +
                         f"_{reflectance:.2f}_{emission:.2f}_{specular:.2f}_{shininess:.2f}").replace(".", "")
        if name not in self._materials:
            self.spec.add_material(name=name, rgba=rgba, reflectance=reflectance, specular=specular,
                                   shininess=shininess, emission=emission)
            self._materials.add(name)
        return name

    # --------------------------------------------------------------- environment
    def setup_world(self, *, floor: str = "grid", floor_size: float = 0, sky: str = "day",
                    ambient: float = 0.16, shadows: bool = True, offscreen: tuple[int, int] = (1280, 960),
                    floor_z: float = 0.0, style: str | None = None) -> None:
        """``style`` (humanoid venues: warehouse / gym / park / house / arena) dresses the surroundings after the task
        has built its layout (see :meth:`finish`); arm tasks get the lab from ``build_table``."""
        self.style = style
        s = self.spec
        s.visual.global_.offwidth = max(s.visual.global_.offwidth, offscreen[0])
        s.visual.global_.offheight = max(s.visual.global_.offheight, offscreen[1])
        s.visual.headlight.ambient = [ambient] * 3
        s.visual.headlight.diffuse = [0.22] * 3
        s.visual.headlight.specular = [0.05] * 3
        s.visual.quality.shadowsize = 4096 if shadows else 0
        s.visual.map.znear = 0.001  # x extent: ~1 cm in the lab, so the wrist camera does not see through the table
        s.visual.map.zfar = 200
        s.stat.extent = max(s.stat.extent, 4.0)
        s.visual.rgba.haze = [0.75, 0.82, 0.9, 1.0]
        if sky == "day":
            s.add_texture(name="rb_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.55, 0.72, 0.92],
                          rgb2=[0.95, 0.97, 1.0], width=512, height=3072)
        else:
            s.add_texture(name="rb_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.25, 0.27, 0.32],
                          rgb2=[0.05, 0.05, 0.08], width=512, height=3072)
        if floor == "grid":
            s.add_texture(name="rb_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.62, 0.64, 0.66],
                          rgb2=[0.54, 0.56, 0.58], mark=mujoco.mjtMark.mjMARK_EDGE,
                          markrgb=[0.42, 0.44, 0.46], width=512, height=512)
            s.add_material(name="rb_floor", textures=["", "rb_floor"], texrepeat=[1, 1],
                           texuniform=True, reflectance=0.05)
        elif floor == "dark":
            s.add_texture(name="rb_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.22, 0.25, 0.30],
                          rgb2=[0.17, 0.20, 0.25], mark=mujoco.mjtMark.mjMARK_EDGE,
                          markrgb=[0.4, 0.45, 0.5], width=512, height=512)
            s.add_material(name="rb_floor", textures=["", "rb_floor"], texrepeat=[1, 1],
                           texuniform=True, reflectance=0.1)
        self._materials.add("rb_floor")
        s.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[floor_size, floor_size, 0.05],
                             pos=[0, 0, floor_z], material="rb_floor", friction=[1.0, 0.005, 0.0001], condim=3)
        s.worldbody.add_light(name="sun", pos=[2, -2, 8], dir=[-0.25, 0.25, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                              castshadow=shadows, diffuse=[0.5, 0.5, 0.47], specular=[0.12, 0.12, 0.12],
                              ambient=[0.05, 0.05, 0.05])
        s.worldbody.add_light(name="fill", pos=[-3, 3, 6], dir=[0.3, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                              castshadow=False, diffuse=[0.18, 0.18, 0.2], specular=[0, 0, 0])

    # ----------------------------------------------------------------- cameras
    def add_camera(self, name: str, pos: Vec, target: Vec, fovy: float = 55.0, parent=None,
                   up: Vec = (0.0, 0.0, 1.0)) -> None:
        body = parent if parent is not None else self.spec.worldbody
        body.add_camera(name=name, pos=list(pos), xyaxes=lookat_xyaxes(pos, target, up=up), fovy=fovy)

    # -------------------------------------------------------------- primitives
    def _add_geom(self, body, *, gtype, size, pos=(0, 0, 0), quat=None, color="gray", name=None,
                  contact=True, friction=None, mass=None, density=None, condim=None, solref=None,
                  solimp=None, group=0, material=None, priority=None, margin=None, fromto=None):
        kw = dict(type=gtype, size=list(size) + [0] * (3 - len(size)), group=group)
        if fromto is not None:
            kw["fromto"] = list(fromto)
        else:
            kw["pos"] = list(pos)
        if name:
            kw["name"] = name
        if quat is not None:
            kw["quat"] = list(quat)
        kw["material"] = material or self.material(color)
        if not contact:
            kw["contype"] = 0
            kw["conaffinity"] = 0
            kw["density"] = 0
        if friction is not None:
            kw["friction"] = list(friction)
        if mass is not None:
            kw["mass"] = mass
        if density is not None:
            kw["density"] = density
        if condim is not None:
            kw["condim"] = condim
        if solref is not None:
            kw["solref"] = list(solref)
        if solimp is not None:
            kw["solimp"] = list(solimp)
        if priority is not None:
            kw["priority"] = priority
        if margin is not None:
            kw["margin"] = margin
        return body.add_geom(**kw)

    def body(self, name: str, pos: Vec, quat=None, *, free: bool = False, parent=None, yaw: float | None = None):
        parent = parent if parent is not None else self.spec.worldbody
        kw = dict(name=name, pos=list(pos))
        if yaw is not None:
            quat = quat_yaw(yaw)
        if quat is not None:
            kw["quat"] = list(quat)
        b = parent.add_body(**kw)
        if free:
            b.add_freejoint(name=f"{name}_free")
            self.free_bodies.append(name)
        return b

    def box(self, name: str, pos: Vec, half: Vec, color="gray", *, free=False, mass=None, yaw=None,
            quat=None, friction=None, contact=True, condim=None, parent=None, solref=None, priority=None):
        b = self.body(name, pos, quat, free=free, parent=parent, yaw=yaw)
        self._add_geom(b, gtype=GEOM.mjGEOM_BOX, size=half, color=color, name=name, contact=contact,
                       friction=friction, mass=mass, condim=condim, solref=solref, priority=priority)
        return b

    def cylinder(self, name: str, pos: Vec, radius: float, half_h: float, color="gray", *, free=False,
                 mass=None, quat=None, friction=None, contact=True, condim=None, parent=None, yaw=None):
        b = self.body(name, pos, quat, free=free, parent=parent, yaw=yaw)
        self._add_geom(b, gtype=GEOM.mjGEOM_CYLINDER, size=[radius, half_h], color=color, name=name,
                       contact=contact, friction=friction, mass=mass, condim=condim)
        return b

    def sphere(self, name: str, pos: Vec, radius: float, color="gray", *, free=False, mass=None,
               friction=None, contact=True, condim=None, parent=None, solref=None):
        b = self.body(name, pos, free=free, parent=parent)
        self._add_geom(b, gtype=GEOM.mjGEOM_SPHERE, size=[radius], color=color, name=name, contact=contact,
                       friction=friction, mass=mass, condim=condim, solref=solref)
        return b

    def static_box(self, name: str, pos: Vec, half: Vec, color="gray", *, yaw: float | None = None,
                   friction=None, contact=True, group=0, material=None):
        quat = quat_yaw(yaw) if yaw is not None else None
        return self._add_geom(self.spec.worldbody, gtype=GEOM.mjGEOM_BOX, size=half, pos=pos, quat=quat,
                              color=color, name=name, contact=contact, friction=friction, group=group,
                              material=material)

    def static_cylinder(self, name: str, pos: Vec, radius: float, half_h: float, color="gray", *, contact=True):
        return self._add_geom(self.spec.worldbody, gtype=GEOM.mjGEOM_CYLINDER, size=[radius, half_h], pos=pos,
                              color=color, name=name, contact=contact)

    def marker(self, name: str, pos: Vec, *, shape: str = "cylinder", size: Vec = (0.25, 0.002),
               color="green", alpha: float = 0.6, yaw: float | None = None):
        """Visual-only floor/table decal (no collision, no mass)."""
        rgba = rgba_of(color)
        rgba[3] = alpha
        gtype = {"cylinder": GEOM.mjGEOM_CYLINDER, "box": GEOM.mjGEOM_BOX, "sphere": GEOM.mjGEOM_SPHERE}[shape]
        quat = quat_yaw(yaw) if yaw is not None else None
        return self._add_geom(self.spec.worldbody, gtype=gtype, size=size, pos=pos, quat=quat,
                              name=name, contact=False, material=self.material(rgba, emission=0.1))

    # ------------------------------------------------------------ compound props
    def wall(self, name: str, p0: Vec, p1: Vec, *, height: float = 1.2, thickness: float = 0.1,
             color="white", z0: float = 0.0):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        mid = (p0 + p1) / 2
        length = float(np.linalg.norm(p1 - p0))
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return self.static_box(name, [mid[0], mid[1], z0 + height / 2],
                               [length / 2 + thickness / 2, thickness / 2, height / 2], color, yaw=yaw)

    def flag(self, name: str, pos: Vec, color="green", height: float = 1.6, z: float = 0.0):
        """Decorative goal flag (pole + banner) standing on height ``z`` -- no collision so it never trips the robot."""
        x, y = pos[0], pos[1]
        self._add_geom(self.spec.worldbody, gtype=GEOM.mjGEOM_CYLINDER, size=[0.02, height / 2],
                       pos=[x, y, z + height / 2], color="white", name=f"{name}_pole", contact=False)
        self._add_geom(self.spec.worldbody, gtype=GEOM.mjGEOM_BOX, size=[0.2, 0.01, 0.13],
                       pos=[x + 0.2, y, z + height - 0.14], color=color, name=f"{name}_banner", contact=False,
                       material=self.material(color, emission=0.25))

    def cone_mesh(self, radius: float, height: float) -> str:
        name = f"rb_cone_{int(radius * 1000)}_{int(height * 1000)}"
        if not any(m.name == name for m in self.spec.meshes):
            n = 20
            verts = [0.0, 0.0, height]
            for i in range(n):
                a = 2 * math.pi * i / n
                verts += [radius * math.cos(a), radius * math.sin(a), 0.0]
            self.spec.add_mesh(name=name, uservert=verts)
        return name

    def traffic_cone(self, name: str, pos: Vec, *, radius: float = 0.14, height: float = 0.45,
                     color="orange", mass: float = 0.8, free: bool = True):
        b = self.body(name, [pos[0], pos[1], 0.0], free=free)
        b.add_geom(name=name, type=GEOM.mjGEOM_MESH, meshname=self.cone_mesh(radius, height),
                   material=self.material(color), mass=mass, pos=[0, 0, 0.024])
        b.add_geom(name=f"{name}_base", type=GEOM.mjGEOM_BOX, size=[radius * 1.25, radius * 1.25, 0.012],
                   pos=[0, 0, 0.012], material=self.material("black"), mass=0.3)
        return b

    def open_box(self, name: str, pos: Vec, inner: Vec, *, wall: float = 0.01, color="blue", free=False,
                 mass: float | None = None, yaw: float | None = None):
        """Tray / bin: floor + four walls, ``inner`` = (half_x, half_y, height)."""
        hx, hy, h = inner
        b = self.body(name, pos, free=free, yaw=yaw)
        mat = self.material(color)
        m_each = None if mass is None else mass / 5
        parts = [
            ([0, 0, wall / 2], [hx + wall, hy + wall, wall / 2]),
            ([hx + wall / 2, 0, h / 2 + wall / 2], [wall / 2, hy + wall, h / 2]),
            ([-hx - wall / 2, 0, h / 2 + wall / 2], [wall / 2, hy + wall, h / 2]),
            ([0, hy + wall / 2, h / 2 + wall / 2], [hx, wall / 2, h / 2]),
            ([0, -hy - wall / 2, h / 2 + wall / 2], [hx, wall / 2, h / 2]),
        ]
        for i, (p, s) in enumerate(parts):
            kw = dict(type=GEOM.mjGEOM_BOX, size=s, pos=p, material=mat, name=f"{name}_p{i}")
            if m_each is not None:
                kw["mass"] = m_each
            b.add_geom(**kw)
        return b

    def push_button(self, name: str, pos: Vec, *, axis: str = "z", radius: float = 0.035, travel: float = 0.012,
                    color="red", housing_color="gray", stiffness: float = 150.0, parent=None, yaw: float = 0.0):
        """Spring-loaded push button mounted ON a surface. ``pos`` is the point on the mounting surface (e.g. the
        table top or a wall face); a 2.4 cm housing sits on it and the cap protrudes another 2.4 cm, so the cap face
        is 4.8 cm out from the surface at rest (``push_button_face``). The plunger is a slide joint named
        ``{name}_joint`` with a hard end stop at ``travel``; pressed depth = -qpos (metres). ``axis`` is the
        direction the button faces: "z" = up, "x" = +x after ``yaw`` (e.g. yaw=pi for a button on a wall that
        faces -x). Body ``{name}`` is the plunger (geom ``{name}_cap``); ``{name}_base`` holds the housing."""
        parent = parent if parent is not None else self.spec.worldbody
        if axis == "z":
            quat = [1, 0, 0, 0]
        else:  # face +x in the yawed frame: rotate local z to +x
            quat = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
        qy = quat_yaw(yaw)
        from roboticsbench.core.geometry import quat_mul

        q = quat_mul(qy, np.array(quat, float))
        base = parent.add_body(name=f"{name}_base", pos=list(pos), quat=list(q))
        base.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[radius * 1.45, BUTTON_HOUSING / 2],
                      pos=[0, 0, BUTTON_HOUSING / 2], material=self.material(housing_color), name=f"{name}_housing")
        plunger = base.add_body(name=name, pos=[0, 0, BUTTON_HOUSING])
        j = plunger.add_joint(name=f"{name}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                              range=[-travel, 0], stiffness=stiffness, damping=2.0, springref=0.0,
                              limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        j.solref_limit = [0.004, 1.0]  # a hard end stop, not a soft spring the arm can push through
        j.solimp_limit = [0.99, 0.999, 0.001, 0.5, 2.0]
        plunger.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[radius, BUTTON_CAP / 2], pos=[0, 0, BUTTON_CAP / 2],
                         material=self.material(color, emission=0.2), name=f"{name}_cap", mass=0.02)
        # the cap sinks into its housing when pressed; MuJoCo's parent/child contact filter does not apply to
        # bodies welded to the world, so exclude the pair explicitly (otherwise the housing blocks the travel)
        self.spec.add_exclude(bodyname1=f"{name}_base", bodyname2=name)
        return plunger

    def hinged_door(self, name: str, hinge_pos: Vec, *, width: float = 0.9, height: float = 2.0,
                    thickness: float = 0.04, yaw: float = 0.0, color="darkwood", mass: float = 6.0,
                    stiffness: float = 2.0, damping: float = 3.0, frictionloss: float = 0.5,
                    swing_range=(-100.0, 100.0)):
        """Vertical door hinged at ``hinge_pos``; closed state spans +x (after ``yaw``)."""
        b = self.body(f"{name}", [hinge_pos[0], hinge_pos[1], 0.0], yaw=yaw)
        b.add_joint(name=f"{name}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                    range=[math.radians(swing_range[0]), math.radians(swing_range[1])],
                    stiffness=stiffness, damping=damping, frictionloss=frictionloss,
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        b.add_geom(name=f"{name}_panel", type=GEOM.mjGEOM_BOX,
                   size=[width / 2 - 0.01, thickness / 2, height / 2], pos=[width / 2, 0, height / 2 + 0.01],
                   material=self.material(color), mass=mass)
        for side in (-1, 1):
            b.add_geom(type=GEOM.mjGEOM_CAPSULE, size=[0.015, 0.06],
                       pos=[width - 0.1, side * (thickness / 2 + 0.03), 1.0], quat=[0.7071, 0, 0.7071, 0],
                       material=self.material("metal"), mass=0.1, contype=0, conaffinity=0)
        return b

    def text_panel(self, *_args, **_kwargs):  # pragma: no cover - MuJoCo has no in-scene text
        raise NotImplementedError

    def finish(self) -> None:
        """Called by the Env after ``task.build``: dress the surroundings for the chosen style."""
        if getattr(self, "style", None):
            from roboticsbench.core.environments import dress_hall

            dress_hall(self, self.style)

    # ------------------------------------------------------------------ build
    def compile(self) -> mujoco.MjModel:
        return self.spec.compile()


# MuJoCo 3.14/3.15's native CCD (GJK/EPA) inverts the normal of 2-16% of shallow cylinder-box, cylinder-mesh and
# cylinder-cylinder contacts (libccd: ~0%; box, mesh, sphere and capsule pairs are fine): a weight set down on a
# balance pan was pushed down through it (2026-10-09). libccd is ~30% slower, so only scenes that can make such a
# contact use it (all humanoid scenes - the G1 has cylinders - and most arm scenes); the slowest arm scene would drop
# from 1.3x to 1.0x real time in a 4-vCPU sandbox.
_CYL = int(GEOM.mjGEOM_CYLINDER)
_CCD_CYLINDER_PARTNERS = {_CYL, int(GEOM.mjGEOM_BOX), int(GEOM.mjGEOM_MESH), int(GEOM.mjGEOM_ELLIPSOID)}


def needs_libccd(m: mujoco.MjModel) -> bool:
    """Whether a collidable cylinder can touch a box / mesh / cylinder / ellipsoid in this model."""
    cols = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    partners = [g for g in cols if int(m.geom_type[g]) in _CCD_CYLINDER_PARTNERS]
    for c in cols:
        if int(m.geom_type[c]) != _CYL:
            continue
        for g in partners:
            if g == c or m.geom_bodyid[g] == m.geom_bodyid[c]:
                continue
            if m.body_weldid[m.geom_bodyid[g]] == 0 and m.body_weldid[m.geom_bodyid[c]] == 0:
                continue  # both static
            if (m.geom_contype[c] & m.geom_conaffinity[g]) or (m.geom_contype[g] & m.geom_conaffinity[c]):
                return True
    return False


def choose_collision_path(m: mujoco.MjModel) -> None:
    if needs_libccd(m):
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD


def ring_positions(n: int, radius: float, center: Iterable[float] = (0, 0), phase: float = 0.0) -> np.ndarray:
    c = np.asarray(list(center), float)
    return np.array([c + radius * np.array([math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n)])
                     for i in range(n)])
