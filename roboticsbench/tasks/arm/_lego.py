"""Interlocking toy bricks ("LEGO-like", 4x scale) for the Panda brick-building task
(arm/lego_manual).

Geometry (all collision is real, no teleporting):

* a brick is a hollow shell - top plate + four walls - with chamfered studs on top and, underneath, the classic
  locating features: diamond tubes between the studs of 2-wide bricks and pins between the studs of 1-wide bricks.
  Studs of the brick below enter the cavity with 0.7 mm clearance per side; the 1.8 mm stud chamfer gives a
  +-2.5 mm capture range when a brick is pressed down slightly misaligned.
* the baseplate is a static stud field with printed column letters / row numbers on its border.

Clutch: a seated brick that is pressed down (contact force from the part below >= ``SNAP_FORCE``) clicks in - a weld
equality is switched on between it and the part it sits on (baseplate or brick), at the exact grid pose, just as
real bricks are held by stud friction. Pulling / twisting a clicked brick harder than ``BREAK_FORCE`` /
``BREAK_TORQUE`` pops it off again (bricks above it stay attached to it).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from roboticsbench.core import visuals as V
from roboticsbench.core.geometry import quat_mul, quat_yaw


GEOM = mujoco.mjtGeom

P = 0.032  # stud pitch (m) - 4x a real brick

H = 0.0384  # brick body height (without studs)

GAP = 0.001  # footprint inset per side (adjacent bricks have 2 mm between them: the stud play lets seated bricks skew up to ~2 deg)

CLEAR = 0.0007  # stud-to-wall / stud-to-tube clearance per side

STUD_R = 0.0096  # stud radius (collision: square of this half-size, chamfered on top)

STUD_H = 0.0068

CHAMFER = 0.0018

TOP_T = 0.004  # top plate thickness

WALL = P / 2 - STUD_R - CLEAR - GAP  # 4.7 mm

PIN = P / 2 - STUD_R - CLEAR  # half-size of the pins inside 1-wide bricks (5.7 mm)

TUBE = (2 * (P / 2 - STUD_R) - CLEAR) / math.sqrt(2)  # half-size of the 45-deg (diamond) tubes in 2-wide bricks

PLATE_T = 0.0048  # baseplate thickness

BORDER = 0.026  # printed border around the baseplate's stud field

SNAP_FORCE = 6.0  # N of downward contact force (beyond nothing) needed to click a seated brick in

BREAK_FORCE = 22.0  # N: a clicked brick pops off when its clutch has to carry more than this

BREAK_TORQUE = 1.6  # N m

SEAT_XY = 0.0028  # m: max grid misalignment for a brick to count as seated

SEAT_Z = 0.0015  # m: max height error

SEAT_YAW = 4.0  # deg

SEAT_TILT = 3.0  # deg

BRICK_FRICTION = [0.45, 0.005, 0.0001]

BRICK_SOLREF = [0.005, 1.0]

BRICK_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]

BRICK_SOLMIX = 100.0  # brick contact parameters dominate the mix (e.g. against the soft finger pads, which would

KINDS: dict[str, tuple[int, int]] = {  # name -> (width, length) in studs; length along the brick's local x
    "1x1": (1, 1), "1x2": (1, 2), "1x3": (1, 3), "1x4": (1, 4), "2x2": (2, 2), "2x3": (2, 3), "2x4": (2, 4),
}

COLORS = ("red", "orange", "yellow", "green", "blue", "white", "black", "tan")

PLATE_RGB = (0.64, 0.66, 0.69)


def brick_mass(kind: str) -> float:
    w, l_ = KINDS[kind]
    return 0.018 * w * l_ + 0.012


def _stud_collision_mesh(scene) -> str:
    """Square stud with a chamfered top (convex hull)."""
    a, c, h = STUD_R, CHAMFER, STUD_H
    pts = []
    for z, s in ((0.0, a), (h - c * 1.1, a), (h, a - c)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append((sx * s, sy * s, z))
    return V._convex_mesh(scene, "lego_stud_col", np.array(pts))


def _stud_visual_mesh(scene) -> str:
    f = 0.0009
    prof = [(STUD_R, 0.0), (STUD_R, STUD_H - f)]
    for i in range(1, 4):
        a = (math.pi / 2) * i / 4
        prof.append((STUD_R - f + f * math.cos(a), STUD_H - f + f * math.sin(a)))
    prof.append((0.0, STUD_H))
    return V.lathe_mesh(scene, "lego_stud_vis", prof, segments=28, cap_bottom=False)


def brick_material(scene, color: str) -> str:
    return V.material(scene, f"plastic_{color}")


def brick_cells_local(kind: str) -> list[tuple[float, float]]:
    """Stud centres in the brick frame (origin = footprint centre, x along the length)."""
    w, l_ = KINDS[kind]
    return [((i + 0.5) * P - l_ * P / 2, (j + 0.5) * P - w * P / 2) for i in range(l_) for j in range(w)]


def add_brick(scene, name: str, kind: str, color: str, pos, yaw: float = 0.0, *, free: bool = True,
              collision: bool = True, parent=None, mocap: bool = False):
    """A brick body (origin at the centre of its bottom face, local x along its length)."""
    w, l_ = KINDS[kind]
    hx, hy = l_ * P / 2 - GAP, w * P / 2 - GAP
    parent = parent if parent is not None else scene.spec.worldbody
    b = parent.add_body(name=name, pos=list(pos), quat=list(quat_yaw(yaw)))
    if mocap:
        b.mocap = True
    if free:
        b.add_freejoint(name=f"{name}_free")
        scene.free_bodies.append(name)
        m = brick_mass(kind)
        b.mass = m
        b.ipos = [0.0, 0.0, 0.58 * H]
        dx, dy, dz = 2 * hx, 2 * hy, H
        b.inertia = [m * (dy * dy + dz * dz) / 12, m * (dx * dx + dz * dz) / 12, m * (dx * dx + dy * dy) / 12]
        b.explicitinertial = True
    mat = brick_material(scene, color)
    if collision:
        ck = dict(friction=BRICK_FRICTION, solref=BRICK_SOLREF, solimp=BRICK_SOLIMP, solmix=BRICK_SOLMIX,
                  material=mat, group=3, density=0.0)
        cav = (H - TOP_T) / 2
        b.add_geom(name=f"{name}_top", type=GEOM.mjGEOM_BOX, size=[hx, hy, TOP_T / 2], pos=[0, 0, H - TOP_T / 2],
                   **ck)
        b.add_geom(name=f"{name}_wall_n", type=GEOM.mjGEOM_BOX, size=[hx, WALL / 2, cav],
                   pos=[0, hy - WALL / 2, cav], **ck)
        b.add_geom(name=f"{name}_wall_s", type=GEOM.mjGEOM_BOX, size=[hx, WALL / 2, cav],
                   pos=[0, -hy + WALL / 2, cav], **ck)
        b.add_geom(name=f"{name}_wall_e", type=GEOM.mjGEOM_BOX, size=[WALL / 2, hy - WALL, cav],
                   pos=[hx - WALL / 2, 0, cav], **ck)
        b.add_geom(name=f"{name}_wall_w", type=GEOM.mjGEOM_BOX, size=[WALL / 2, hy - WALL, cav],
                   pos=[-hx + WALL / 2, 0, cav], **ck)
        for k in range(1, l_):
            x = k * P - l_ * P / 2
            if w == 2:
                b.add_geom(name=f"{name}_tube{k}", type=GEOM.mjGEOM_BOX, size=[TUBE, TUBE, cav], pos=[x, 0, cav],
                           quat=list(quat_yaw(math.pi / 4)), **ck)
            else:
                b.add_geom(name=f"{name}_pin{k}", type=GEOM.mjGEOM_BOX, size=[PIN, PIN, cav], pos=[x, 0, cav], **ck)
        stud = _stud_collision_mesh(scene)
        for i, (x, y) in enumerate(brick_cells_local(kind)):
            b.add_geom(name=f"{name}_stud{i}", type=GEOM.mjGEOM_MESH, meshname=stud, pos=[x, y, H], **ck)
    # looks: rounded shell + round studs
    shell = V.rounded_box_mesh(scene, f"lego_shell_{kind}", (hx, hy, H / 2), 0.0012, segments=3)
    V.visual_mesh(scene, b, shell, material=mat, pos=(0, 0, H / 2))
    sv = _stud_visual_mesh(scene)
    for x, y in brick_cells_local(kind):
        V.visual_mesh(scene, b, sv, material=mat, pos=(x, y, H))
    return b


def plate_texture_image(n: int, px: int = 1024) -> np.ndarray:
    """Top face of the baseplate: stud field + border printed with column letters (plate x) and row numbers
    (plate y) on all four sides. Image top = plate +y, image left = plate -x."""
    from PIL import Image, ImageDraw

    size = n * P + 2 * BORDER
    s = px / size
    base = tuple(int(255 * c) for c in PLATE_RGB)
    border_c = tuple(int(0.8 * c) for c in base)
    im = Image.new("RGB", (px, px), border_c)
    dr = ImageDraw.Draw(im)
    b0 = int(round(BORDER * s))
    dr.rectangle([b0, b0, px - b0, px - b0], fill=base)
    # faint grid lines between cells
    for k in range(n + 1):
        u = int(round((BORDER + k * P) * s))
        dr.line([(u, b0), (u, px - b0)], fill=tuple(int(0.9 * c) for c in base), width=2)
        dr.line([(b0, u), (px - b0, u)], fill=tuple(int(0.9 * c) for c in base), width=2)
    f = V._pil_font(int(BORDER * s * 0.62))
    ink = (250, 250, 248)

    def put(text, cx, cy, rot):
        tw = int(dr.textlength(text, font=f)) + 8
        tile = Image.new("RGBA", (max(tw, 10), int(BORDER * s)), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        bb = td.textbbox((0, 0), text, font=f)
        td.text(((tile.width - (bb[2] - bb[0])) / 2 - bb[0], (tile.height - (bb[3] - bb[1])) / 2 - bb[1]), text,
                font=f, fill=ink)
        tile = tile.rotate(rot, expand=True)
        im.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)

    for i in range(n):
        u = (BORDER + (i + 0.5) * P) * s
        letter = chr(ord("A") + i)
        put(letter, u, px - b0 / 2, 0)  # bottom border (plate -y side), readable from -y
        put(letter, u, b0 / 2, 180)  # top border (plate +y side), readable from +y
        num = str(i + 1)
        v = px - (BORDER + (i + 0.5) * P) * s  # row i+1 at plate y = (i + 0.5) P from the -y edge
        put(num, b0 / 2, v, -90)  # left border (plate -x side), readable from -x
        put(num, px - b0 / 2, v, 90)  # right border, readable from +x
    return np.asarray(im)


def plate_material(scene, n: int) -> str:
    tex = f"lego_plate_tex_{n}"
    if not any(t.name == tex for t in scene.spec.textures):
        img = plate_texture_image(n)
        t = scene.spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                                   height=img.shape[0], nchannel=3)
        t.data = np.ascontiguousarray(img).tobytes()
    mat = f"lego_plate_mat_{n}"
    if not any(m.name == mat for m in scene.spec.materials):
        scene.spec.add_material(name=mat, textures=["", tex], texuniform=False, texrepeat=[1, 1], specular=0.35,
                                shininess=0.5)
    return mat


def add_baseplate(scene, name: str, pos, yaw: float = 0.0, n: int = 8, *, collision: bool = True):
    """Static baseplate; body origin at the plate centre on its bottom face (top face at z = PLATE_T)."""
    b = scene.spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat_yaw(yaw)))
    half = n * P / 2 + BORDER
    b.add_geom(name=f"{name}_slab", type=GEOM.mjGEOM_BOX, size=[half, half, PLATE_T / 2], pos=[0, 0, PLATE_T / 2],
               material=plate_material(scene, n), friction=BRICK_FRICTION, solref=BRICK_SOLREF,
               solimp=BRICK_SOLIMP, solmix=BRICK_SOLMIX, contype=1 if collision else 0,
               conaffinity=1 if collision else 0)
    mat = V.material(scene, "plastic_grey", rgba=(*PLATE_RGB, 1.0))
    stud = _stud_collision_mesh(scene) if collision else None
    sv = _stud_visual_mesh(scene)
    for i in range(n):
        for j in range(n):
            x, y = (i + 0.5) * P - n * P / 2, (j + 0.5) * P - n * P / 2
            if collision:
                b.add_geom(name=f"{name}_stud_{i}_{j}", type=GEOM.mjGEOM_MESH, meshname=stud, pos=[x, y, PLATE_T],
                           friction=BRICK_FRICTION, solref=BRICK_SOLREF, solimp=BRICK_SOLIMP, solmix=BRICK_SOLMIX,
                           material=mat, group=3)
            V.visual_mesh(scene, b, sv, material=mat, pos=(x, y, PLATE_T))
    return b


@dataclass(frozen=True)
class GridPose:
    """A brick's place on the baseplate grid: min-corner cell (col = plate x index, row = plate y index), level
    (0 = on the plate) and the direction of its length ("x" or "y", plate frame)."""
    kind: str
    col: int
    row: int
    level: int
    along: str

    def cells(self) -> set[tuple[int, int]]:
        w, l_ = KINDS[self.kind]
        nx, ny = (l_, w) if self.along == "x" else (w, l_)
        return {(self.col + i, self.row + j) for i in range(nx) for j in range(ny)}

    def canonical(self) -> "GridPose":
        w, l_ = KINDS[self.kind]
        return GridPose(self.kind, self.col, self.row, self.level, "x" if w == l_ else self.along)

    def center_local(self, n: int) -> np.ndarray:
        """Brick origin (bottom-face centre) in the baseplate body frame."""
        w, l_ = KINDS[self.kind]
        nx, ny = (l_, w) if self.along == "x" else (w, l_)
        return np.array([(self.col + nx / 2) * P - n * P / 2, (self.row + ny / 2) * P - n * P / 2,
                         PLATE_T + self.level * H])

    def yaw_local(self) -> float:
        return 0.0 if self.along == "x" else math.pi / 2


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class Clutch:
    """Snap / break logic for the bricks of one baseplate. Construct during ``build`` (declares the weld equalities),
    call :meth:`bind` after compilation, :meth:`reset` on episode reset and :meth:`step` every control tick."""

    def __init__(self, scene, plate: str, n: int, bricks: dict[str, str]):
        self.plate, self.n = plate, n
        self.kinds = dict(bricks)
        self.names = list(bricks)
        self.pairs: dict[tuple[str, str], str] = {}
        parts = [plate] + self.names
        for i, a in enumerate(parts):  # one (inactive) weld per pair of parts that could click together
            for b in parts[i + 1:]:
                en = f"clutch__{a}__{b}"
                scene.spec.add_equality(name=en, type=mujoco.mjtEq.mjEQ_WELD, name1=a, name2=b,
                                        objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.01, 1.0],
                                        solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
                self.pairs[(a, b)] = en
        self.parent: dict[str, str | None] = {}
        self.cooldown: dict[str, float] = {}
        self.events: list[str] = []
        self.debug: list[str] = []

    # ------------------------------------------------------------------ setup
    def bind(self, env) -> None:
        m = env.model
        self.m = m
        self.bid = {nm: m.body(nm).id for nm in [self.plate] + self.names}
        self.eid = {k: m.equality(v).id for k, v in self.pairs.items()}
        self.eq_body2 = {k: k[1] for k in self.pairs}
        self.geom_owner: dict[int, str] = {}
        for nm, b in self.bid.items():
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and m.geom_contype[g]:
                    self.geom_owner[g] = nm

    def reset(self, env) -> None:
        self.parent = {nm: None for nm in self.names}
        self.cooldown = {nm: -1.0 for nm in self.names}
        self.events = []
        self.debug = []
        env.data.eq_active[:] = env.model.eq_active0

    def _eq(self, a: str, b: str) -> tuple[int, str, str]:
        key = (a, b) if (a, b) in self.eid else (b, a)
        return self.eid[key], key[0], key[1]

    # ------------------------------------------------------------ geometry
    def plate_frame(self, env) -> tuple[np.ndarray, np.ndarray]:
        d = env.data
        b = self.bid[self.plate]
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def grid_pose(self, env, name: str, *, tol_xy: float = SEAT_XY, tol_z: float = SEAT_Z) -> GridPose | None:
        """Where the brick sits on the grid, or None when it is not seated on the lattice (tilted, between cells,
        floating, off the plate area...)."""
        d = env.data
        pp, Rp = self.plate_frame(env)
        b = self.bid[name]
        r = Rp.T @ (d.xpos[b] - pp)
        R = Rp.T @ d.xmat[b].reshape(3, 3)
        if math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1)))) > SEAT_TILT:
            return None
        yaw = math.atan2(R[1, 0], R[0, 0])
        k = round(yaw / (math.pi / 2))
        if abs(math.degrees(yaw - k * math.pi / 2)) > SEAT_YAW:
            return None
        zb = r[2] - PLATE_T
        lvl = round(zb / H)
        if lvl < 0 or abs(zb - lvl * H) > tol_z:
            return None
        kind = self.kinds[name]
        w, l_ = KINDS[kind]
        along = "x" if k % 2 == 0 else "y"
        nx, ny = (l_, w) if along == "x" else (w, l_)
        mx = r[0] - nx * P / 2 + self.n * P / 2
        my = r[1] - ny * P / 2 + self.n * P / 2
        col, row = round(mx / P), round(my / P)
        if abs(mx - col * P) > tol_xy or abs(my - row * P) > tol_xy:
            return None
        return GridPose(kind, int(col), int(row), int(lvl), along)

    def snapped(self) -> list[str]:
        return [nm for nm in self.names if self.parent.get(nm)]

    # ------------------------------------------------------------------ runtime
    def _support_force(self, env, name: str, below: set[str]) -> float:
        """Upward (plate-normal) contact force on ``name`` from parts in ``below``."""
        m, d = env.model, env.data
        _, Rp = self.plate_frame(env)
        up = Rp[:, 2]
        f6 = np.zeros(6)
        total = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            o1, o2 = self.geom_owner.get(c.geom1), self.geom_owner.get(c.geom2)
            if o1 == name and o2 in below:
                sign = -1.0
            elif o2 == name and o1 in below:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            n = c.frame[:3]
            total += sign * f6[0] * float(n @ up)
        return total

    def _weld_loads(self, env) -> dict[int, np.ndarray]:
        d = env.data
        nefc = d.nefc
        if nefc == 0:
            return {}
        t = d.efc_type[:nefc]
        rows = np.nonzero(t == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
        out: dict[int, list[int]] = {}
        ids = d.efc_id[:nefc]
        for r in rows:
            out.setdefault(int(ids[r]), []).append(int(r))
        return {e: d.efc_force[np.array(rs)] for e, rs in out.items() if len(rs) == 6}

    def step(self, env) -> None:
        d = env.data
        t = env.t
        # --- break overloaded clutches: a pull on a stack pops the topmost overloaded joint only (the load on the
        # joints below it vanishes once it lets go)
        loads = self._weld_loads(env)
        pp, Rp = self.plate_frame(env)
        over: list[tuple[float, str, int, str]] = []
        for nm in self.names:
            par = self.parent.get(nm)
            if not par:
                continue
            e, _, body2 = self._eq(par, nm)
            f = loads.get(e)
            if f is None:
                continue
            f_on_child = (-f[:3]) if body2 == nm else f[:3]  # force the weld applies to the brick (world frame)
            fl = Rp.T @ f_on_child
            tension = -fl[2]  # weld pulling the brick down onto its support = someone pulls it off
            shear = math.hypot(fl[0], fl[1])
            torque = float(np.linalg.norm(f[3:]))
            if tension > BREAK_FORCE or shear > 2.5 * BREAK_FORCE or torque > BREAK_TORQUE:
                z = float((Rp.T @ (d.xpos[self.bid[nm]] - pp))[2])
                over.append((z, nm, e, par, f"{tension:.0f}/{shear:.0f}/{torque:.2f}"))
        if over:
            z, nm, e, par, why = max(over)
            d.eq_active[e] = 0
            self.parent[nm] = None
            self.cooldown[nm] = t + 0.4
            self.events.append(f"t={t:.1f}s {nm} popped off {par}")
            self.debug.append(f"t={t:.2f} {nm} off {par}: tension/shear/torque {why}")
        # --- click in seated, pressed bricks
        snapped_pose = {nm: self.grid_pose(env, nm) for nm in self.names if self.parent.get(nm)}
        for nm in self.names:
            if self.parent.get(nm) or t < self.cooldown.get(nm, -1.0):
                continue
            b = self.bid[nm]
            if np.linalg.norm(d.cvel[b][3:]) > 0.08:
                continue
            gp = self.grid_pose(env, nm)
            if gp is None:
                continue
            cells = gp.cells()
            if gp.level == 0:
                if not any(0 <= c < self.n and 0 <= r_ < self.n for c, r_ in cells):
                    continue
                support, below = self.plate, {self.plate}
            else:
                best, overlap = None, 0
                below = set()
                for other, op in snapped_pose.items():
                    if op is None or op.level != gp.level - 1:
                        continue
                    ov = len(cells & op.cells())
                    if ov:
                        below.add(other)
                    if ov > overlap:
                        best, overlap = other, ov
                if best is None:
                    continue
                support = best
            if self._support_force(env, nm, below) < SNAP_FORCE:
                continue
            self._weld(env, nm, support, gp)
            snapped_pose[nm] = gp

    def _weld(self, env, name: str, support: str, gp: GridPose) -> None:
        """Lock the brick to its support where it sits now (seated bricks are within the 0.7 mm stud play of the
        grid; snapping to the exact grid pose would fight the gripper that is still holding the brick)."""
        d, m = env.data, env.model
        bb, sb = self.bid[name], self.bid[support]
        e, body1, _ = self._eq(support, name)
        a, b = (sb, bb) if body1 == support else (bb, sb)
        R1 = d.xmat[a].reshape(3, 3)
        relpos = R1.T @ (d.xpos[b] - d.xpos[a])
        relq = quat_mul(_quat_conj(d.xquat[a]), d.xquat[b])
        m.eq_data[e, :] = 0.0
        m.eq_data[e, 3:6] = relpos
        m.eq_data[e, 6:10] = relq
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        self.parent[name] = support
        self.events.append(f"t={env.t:.1f}s {name} clicked onto {support}")


@dataclass(frozen=True)
class DesignBrick:
    color: str
    pose: GridPose

    @property
    def kind(self) -> str:
        return self.pose.kind


# the gripper can close across a brick side of at most this many studs (8 cm opening, 6.4 cm = 2 studs)
MAX_GRIP_STUDS = 2


def grasp_axes(b: GridPose, placed: Iterable[GridPose]) -> list[str]:
    """Plate axes ("x" / "y") along which the fingers can close on brick ``b`` given the other bricks already on its
    level: the cells next to the brick's two faces, around the middle of those faces, must be free (a finger pad
    is 1.7 cm wide and needs up to ~2.5 cm outside the brick)."""
    occ = set()
    for o in placed:
        if o.level == b.level:
            occ |= o.cells()
    cells = b.cells()
    cols = sorted({c for c, _ in cells})
    rows = sorted({r for _, r in cells})
    out = []
    for axis in ("x", "y"):
        span = cols if axis == "x" else rows  # closing direction
        other = rows if axis == "x" else cols
        if len(span) > MAX_GRIP_STUDS:
            continue
        mid = (other[0] + other[-1]) / 2.0
        centre = [k for k in other if abs(k - mid) <= 0.5 + 1e-9]
        if axis == "x":
            side = [(cols[0] - 1, r) for r in centre] + [(cols[-1] + 1, r) for r in centre]
        else:
            side = [(c, rows[0] - 1) for c in centre] + [(c, rows[-1] + 1) for c in centre]
        if not any(s in occ for s in side):
            out.append(axis)
    return out


def _supported_by(b: GridPose, placed: list[GridPose]) -> list[int]:
    return [i for i, o in enumerate(placed) if o.level == b.level - 1 and b.cells() & o.cells()]


def generate_design(rng: np.random.Generator, *, n_plate: int = 8, n_bricks=(9, 12), levels=(3, 5),
                    kinds=("1x2", "1x3", "1x4", "2x2", "2x3", "2x4"), colors=COLORS, attempts: int = 60
                    ) -> list[DesignBrick]:
    """A random but structured model: a connected base layer, upper layers that sit on (and preferably bridge)
    the bricks below, every brick placeable by a parallel gripper in build order. Returns bricks in build order
    (level by level)."""
    best, best_score = None, -1e9
    lo, hi = 1, n_plate - 1  # keep a free cell ring next to the border
    for _ in range(attempts):
        target = int(rng.integers(n_bricks[0], n_bricks[1] + 1))
        n_lv = int(rng.integers(levels[0], levels[1] + 1))
        # bricks per level: bottom-heavy
        per = np.ones(n_lv, int)
        for _k in range(target - n_lv):
            wts = np.array([n_lv - i + 0.5 for i in range(n_lv)], float)
            per[rng.choice(n_lv, p=wts / wts.sum())] += 1
        placed: list[GridPose] = []
        solid: set = set()  # cells of the level below that stand on a column of bricks down to the plate
        ok = True
        for lv in range(n_lv):
            below = {c for o in placed if o.level == lv - 1 for c in o.cells()}
            count = 0
            for _try in range(400):
                if count >= per[lv]:
                    break
                kind = str(rng.choice(kinds))
                along = "x" if rng.random() < 0.5 else "y"
                w, l_ = KINDS[kind]
                nx, ny = (l_, w) if along == "x" else (w, l_)
                col = int(rng.integers(lo, hi - nx + 1))
                row = int(rng.integers(lo, hi - ny + 1))
                g = GridPose(kind, col, row, lv, along).canonical()
                cells = g.cells()
                same = [o for o in placed if o.level == lv]
                if any(cells & o.cells() for o in same):
                    continue
                if lv == 0:
                    if same and not any(_touch(cells, o.cells()) for o in same):
                        continue
                else:
                    sup = cells & below
                    if len(sup) < max(1, math.ceil(len(cells) / 3)):
                        continue
                    # an upper brick must not float over empty space for more than 4 cells
                    if len(cells - below) > 4:
                        continue
                    # nor rest on an overhang: pressing it in would lever the brick below off its studs
                    if not sup <= solid:
                        continue
                if not grasp_axes(g, same):
                    continue
                placed.append(g)
                count += 1
            if count < per[lv]:
                ok = False
                break
            level_cells = {c for o in placed if o.level == lv for c in o.cells()}
            solid = level_cells if lv == 0 else level_cells & solid
        if not ok:
            continue
        bridges = sum(1 for g in placed if g.level > 0 and len(_supported_by(g, placed)) >= 2)
        cells0 = [c for g in placed for c in g.cells()]
        xs, ys = [c for c, _ in cells0], [r for _, r in cells0]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        score = 3.0 * bridges + n_lv - 0.08 * area + 0.3 * len({g.kind for g in placed})
        if score > best_score:
            best, best_score = placed, score
    if best is None:
        raise RuntimeError("could not generate a brick design")
    return _colour(rng, best, colors)


def _touch(a: set, b: set) -> bool:
    return any((c + dc, r + dr) in b for c, r in a for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def _colour(rng: np.random.Generator, placed: list[GridPose], colors) -> list[DesignBrick]:
    """Colours so that bricks touching each other (side by side or stacked) differ where possible."""
    out: list[DesignBrick] = []
    palette = list(colors)
    for g in placed:
        cells = g.cells()
        nb = set()
        for o, db in zip(placed, out):
            if (o.level == g.level and _touch(cells, o.cells())) or (abs(o.level - g.level) == 1 and cells & o.cells()):
                nb.add(db.color)
        choices = [c for c in palette if c not in nb] or palette
        out.append(DesignBrick(str(rng.choice(choices)), g))
    return out


def build_steps(design: list[DesignBrick], max_per_step: int = 2) -> list[list[int]]:
    """Group the build order into manual steps (same level, up to ``max_per_step`` bricks that do not block each
    other's grasp in either order)."""
    steps: list[list[int]] = []
    placed: list[GridPose] = []
    i = 0
    while i < len(design):
        step = [i]
        cur = design[i].pose
        if i + 1 < len(design) and len(step) < max_per_step:
            nxt = design[i + 1].pose
            if nxt.level == cur.level and grasp_axes(cur, placed + [nxt]) and grasp_axes(nxt, placed + [cur]):
                step.append(i + 1)
        for k in step:
            placed.append(design[k].pose)
        steps.append(step)
        i += len(step)
    return steps


def pick_spares(rng: np.random.Generator, design: list[DesignBrick], n: int = 2,
                kinds=("1x2", "1x3", "1x4", "2x2", "2x3", "2x4"), colors=COLORS) -> list[tuple[str, str]]:
    """Decoy parts that look like design parts (same colour, other size - or same size, other colour)."""
    used = {(b.kind, b.color) for b in design}
    out = []
    for _ in range(200):
        if len(out) >= n:
            break
        ref = design[int(rng.integers(len(design)))]
        if rng.random() < 0.5:
            cand = (str(rng.choice([k for k in kinds if k != ref.kind])), ref.color)
        else:
            cand = (ref.kind, str(rng.choice([c for c in colors if c != ref.color])))
        if cand not in used and cand not in out:
            out.append(cand)
    return out


MANUAL_BG = (226, 236, 246)

HIGHLIGHT = (225, 0, 185)

VIEW_MAIN = (60.0, -34.0)  # (azimuth, elevation) deg of the manual's main view: from the A1 corner side

VIEW_BACK = (240.0, -34.0)

VIEW_TOP = "plan"  # orthographic plan view: plate +x to the right, +y up


class ManualRenderer:
    """Renders illustrated building instructions for a design (plate frame == world frame here)."""

    def __init__(self, design: list[DesignBrick], n_plate: int, *, width: int = 1200, height: int = 900):
        from roboticsbench.core.scene import SceneBuilder

        self.design, self.n = design, n_plate
        self.W, self.Hh = width, height
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth = width
        spec.visual.global_.offheight = height
        spec.visual.quality.offsamples = 8
        spec.visual.headlight.ambient = [0.45] * 3
        spec.visual.headlight.diffuse = [0.3] * 3
        spec.visual.headlight.specular = [0.06] * 3
        spec.visual.map.znear = 0.01
        spec.visual.map.zfar = 30.0
        spec.stat.extent = 1.0  # hidden parts are parked far away; keep near/far clipping tuned to the model
        spec.stat.center = [0.0, 0.0, 0.05]
        wb = spec.worldbody
        wb.add_light(name="key", pos=[-1, -2, 3], dir=[0.3, 0.6, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     castshadow=False, diffuse=[0.42, 0.42, 0.4], specular=[0.2, 0.2, 0.2])
        wb.add_light(name="fill", pos=[2, 1, 2], dir=[-0.6, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     castshadow=False, diffuse=[0.16, 0.16, 0.18], specular=[0, 0, 0])
        plan = wb.add_camera(name="plan", pos=[0, 0, 1.0], xyaxes=[1, 0, 0, 0, 1, 0])
        plan.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
        plan.fovy = n_plate * P + 2 * BORDER + 0.012  # orthographic: view height in metres
        sc = SceneBuilder(spec)
        add_baseplate(sc, "plate", [0, 0, 0], 0.0, n_plate, collision=False)
        for i, b in enumerate(design):
            add_brick(sc, f"m{i}", b.kind, b.color, [0, 0, -5 - 0.1 * i], 0.0, free=False, collision=False,
                      mocap=True)
        self.part_keys = sorted({(b.kind, b.color) for b in design}, key=lambda kc: (KINDS[kc[0]], kc[1]))
        for j, (k, c) in enumerate(self.part_keys):
            add_brick(sc, f"part{j}", k, c, [10 + 1.0 * j, 0, 0], 0.0, free=False, collision=False, mocap=True)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.mocap = {i: m.body_mocapid[m.body(f"m{i}").id] for i in range(len(design))}
        self.geoms = {i: {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"m{i}").id}
                      for i in range(len(design))}
        self._r: dict[tuple[int, int], mujoco.Renderer] = {}
        mujoco.mj_forward(self.model, self.data)
        cells = [c for b in design for c in b.pose.cells()]
        xs = [(c + 0.5) * P - n_plate * P / 2 for c, _ in cells]
        ys = [(r + 0.5) * P - n_plate * P / 2 for _, r in cells]
        top = max(b.pose.level for b in design) + 1
        self.center = np.array([(min(xs) + max(xs)) / 2 * 0.5, (min(ys) + max(ys)) / 2 * 0.5,
                                PLATE_T + 0.35 * top * H])

    def close(self) -> None:
        for r in self._r.values():
            r.close()
        self._r.clear()

    # ------------------------------------------------------------------ rendering
    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._r:
            self._r[(w, h)] = mujoco.Renderer(self.model, h, w)
        return self._r[(w, h)]

    def _show(self, upto: set[int]) -> None:
        for i, b in enumerate(self.design):
            mid = self.mocap[i]
            if i in upto:
                self.data.mocap_pos[mid] = b.pose.center_local(self.n)
                self.data.mocap_quat[mid] = quat_yaw(b.pose.yaw_local())
            else:
                self.data.mocap_pos[mid] = [0, 0, -5 - 0.1 * i]
        mujoco.mj_forward(self.model, self.data)

    def _cam(self, view, lookat=None, distance: float = 0.6) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.center if lookat is None else lookat
        cam.distance = distance
        cam.azimuth, cam.elevation = view
        return cam

    def render(self, view, w: int, h: int, *, highlight: set[int] = frozenset(), lookat=None,
               distance: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """RGB image (background painted flat) + mask of highlighted-brick pixels."""
        r = self._renderer(w, h)
        cam = view if isinstance(view, str) else self._cam(view, lookat, distance)
        r.update_scene(self.data, camera=cam)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=cam)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        rgb[seg < 0] = MANUAL_BG
        hl = set().union(*[self.geoms[i] for i in highlight]) if highlight else set()
        mask = np.isin(seg, list(hl)) if hl else np.zeros(seg.shape, bool)
        return rgb, mask

    def visible_fraction(self, i: int, upto: set[int], view) -> float:
        self._show(upto)
        _, m_all = self.render(view, 400, 300, highlight={i})
        self._show({i})
        _, m_one = self.render(view, 400, 300, highlight={i})
        self._show(upto)
        return float(m_all.sum()) / max(1.0, float(m_one.sum()))

    def part_image(self, j: int, w: int = 220, h: int = 165) -> np.ndarray:
        k, _ = self.part_keys[j]
        wd, ln = KINDS[k]
        pos = np.array([10 + 1.0 * j, 0, H * 0.6])
        dist = 0.10 + 0.03 * max(wd, ln)
        rgb, _ = self.render(VIEW_MAIN, w, h, lookat=pos, distance=dist)
        return rgb

    # ------------------------------------------------------------------ pages
    def pages(self) -> dict[str, bytes]:
        from PIL import Image, ImageDraw, ImageFilter

        W, Hh = self.W, self.Hh
        steps = build_steps(self.design)
        font_big, font_mid, font_small = V._pil_font(64), V._pil_font(30), V._pil_font(22)
        out: dict[str, bytes] = {}

        def png(im) -> bytes:
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=True)
            return buf.getvalue()

        def outline(rgb, mask, width=3):
            if not mask.any():
                return rgb
            mimg = Image.fromarray((mask * 255).astype(np.uint8))
            dil = np.asarray(mimg.filter(ImageFilter.MaxFilter(2 * width + 1))) > 0
            ring = dil & ~mask
            rgb = rgb.copy()
            rgb[ring] = HIGHLIGHT
            return rgb

        def callout(page, dr, items, x0=24, y0=24):
            tw, th = 200, 150
            box_w = 24 + len(items) * (tw + 16)
            box_h = th + 70
            dr.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=18, fill=(255, 255, 255),
                                 outline=(120, 130, 150), width=3)
            for n_, (j, cnt) in enumerate(items):
                im = Image.fromarray(self.part_image(j, tw, th))
                px = x0 + 16 + n_ * (tw + 16)
                page.paste(im, (px, y0 + 12))
                dr.text((px + 6, y0 + th + 18), f"{cnt}x", font=font_mid, fill=(20, 20, 30))
                dr.text((px + 80, y0 + th + 24), self.part_keys[j][0], font=font_small, fill=(90, 90, 110))

        # --- cover: parts list
        page = Image.new("RGB", (W, Hh), MANUAL_BG)
        dr = ImageDraw.Draw(page)
        dr.text((40, 30), "PARTS", font=font_big, fill=(20, 20, 30))
        counts: dict[int, int] = {}
        for b in self.design:
            j = self.part_keys.index((b.kind, b.color))
            counts[j] = counts.get(j, 0) + 1
        cols = 4
        tw, th = 250, 190
        for n_, j in enumerate(sorted(counts)):
            cx, cy = 40 + (n_ % cols) * (tw + 30), 130 + (n_ // cols) * (th + 70)
            dr.rounded_rectangle([cx, cy, cx + tw, cy + th + 50], radius=14, fill=(255, 255, 255),
                                 outline=(150, 160, 175), width=2)
            page.paste(Image.fromarray(self.part_image(j, tw - 10, th - 10)), (cx + 5, cy + 5))
            dr.text((cx + 12, cy + th + 6), f"{counts[j]}x", font=font_mid, fill=(20, 20, 30))
            dr.text((cx + 90, cy + th + 12), self.part_keys[j][0], font=font_small, fill=(90, 90, 110))
        out["manual/page_00_parts.png"] = png(page)

        # --- steps
        upto: set[int] = set()
        for s_i, step in enumerate(steps):
            upto |= set(step)
            self._show(upto)
            rgb, mask = self.render(VIEW_MAIN, W, Hh - 60, highlight=set(step))
            page = Image.new("RGB", (W, Hh), MANUAL_BG)
            page.paste(Image.fromarray(outline(rgb, mask)), (0, 60))
            dr = ImageDraw.Draw(page)
            label = f"STEP {s_i + 1} / {len(steps)}"
            dr.text((W - 30 - dr.textlength(label, font=font_big), 14), label, font=font_big, fill=(20, 20, 30))
            items: dict[int, int] = {}
            for i in step:
                j = self.part_keys.index((self.design[i].kind, self.design[i].color))
                items[j] = items.get(j, 0) + 1
            callout(page, dr, sorted(items.items()), 24, 24)
            # plan view (from above) - bottom left
            self._show(upto)
            tw_, th_ = 330, 330
            rgb3, mask3 = self.render(VIEW_TOP, tw_, th_, highlight=set(step))
            x0, y0 = 24, Hh - th_ - 24
            dr.rounded_rectangle([x0 - 8, y0 - 44, x0 + tw_ + 8, y0 + th_ + 8], radius=14, fill=(255, 255, 255),
                                 outline=(120, 130, 150), width=3)
            page.paste(Image.fromarray(outline(rgb3, mask3, 2)), (x0, y0))
            dr.text((x0 + 8, y0 - 38), "view from above", font=font_small, fill=(60, 60, 80))
            hidden = [i for i in step if self.visible_fraction(i, upto, VIEW_MAIN) < 0.45]
            if hidden:
                self._show(upto)
                iw, ih = 460, 345
                rgb2, mask2 = self.render(VIEW_BACK, iw, ih, highlight=set(step))
                x0, y0 = W - iw - 24, Hh - ih - 24
                dr.rounded_rectangle([x0 - 8, y0 - 44, x0 + iw + 8, y0 + ih + 8], radius=14, fill=(255, 255, 255),
                                     outline=(120, 130, 150), width=3)
                page.paste(Image.fromarray(outline(rgb2, mask2, 2)), (x0, y0))
                dr.text((x0 + 8, y0 - 38), "view from the opposite corner", font=font_small, fill=(60, 60, 80))
            out[f"manual/page_{s_i + 1:02d}_step{s_i + 1}.png"] = png(page)

        # --- finished model, two views
        self._show(set(range(len(self.design))))
        page = Image.new("RGB", (W, Hh), MANUAL_BG)
        dr = ImageDraw.Draw(page)
        dr.text((40, 24), "FINISHED MODEL", font=font_big, fill=(20, 20, 30))
        hw = W // 2 - 30
        for k, view in enumerate((VIEW_MAIN, VIEW_BACK)):
            rgb, _ = self.render(view, hw, int(hw * 0.9), distance=0.53)
            page.paste(Image.fromarray(rgb), (20 + k * (hw + 20), 130))
        dr.text((40, 130 + int(hw * 0.9) + 16), "left: from the A1 corner    right: from the opposite corner",
                font=font_small, fill=(60, 60, 80))
        out[f"manual/page_{len(steps) + 1:02d}_finished.png"] = png(page)
        return out


def _rect_corners(x, y, yaw, hx, hy) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = np.array([c, s]), np.array([-s, c])
    ctr = np.array([x, y])
    return np.array([ctr + sx * hx * ax + sy * hy * ay for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def _rects_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def neutral_yaw_deg(x: float, y: float) -> float:
    """Gripper yaw (deg) that leaves the Panda's last wrist joint mid-range at (x, y)."""
    return math.degrees(math.atan2(y, x)) + 45.0


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def best_yaw(base_deg: float, period: float, x: float, y: float, near: float | None = None) -> float:
    """Among base + k*period, the gripper yaw closest to the wrist-neutral yaw at (x, y) (and to ``near``)."""
    nz = neutral_yaw_deg(x, y)
    cands = [base_deg + k * period for k in range(-8, 9)]
    cands = [c for c in cands if abs(c - nz) <= 100.0] or cands

    def cost(c):
        return abs(c - nz) + (0.5 * abs(c - near) if near is not None else 0.0)

    return float(min(cands, key=cost))
