"""Arm-task props and helpers: materials, textures, labels and free bodies (the ones arm/balance_scale uses)."""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

from roboticsbench.core import visuals as V


GEOM = mujoco.mjtGeom

BOX, CYL, SPH, CAP, MESH = (GEOM.mjGEOM_BOX, GEOM.mjGEOM_CYLINDER, GEOM.mjGEOM_SPHERE, GEOM.mjGEOM_CAPSULE,
                            GEOM.mjGEOM_MESH)


def quat_yaw(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def quat_axis(axis: Sequence[float], a: float) -> np.ndarray:
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return np.array([math.cos(a / 2), *(math.sin(a / 2) * ax)])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def joint_adr(model, joint: str) -> tuple[int, int]:
    """(qpos address, dof address) of a 1-dof joint."""
    j = model.joint(joint).id
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def custom_material(scene, name: str, rgba, *, emission: float = 0.0, specular: float = 0.5,
                    shininess: float = 0.6, reflectance: float = 0.0, texture: str | None = None,
                    texrepeat=(1, 1), texuniform: bool = True) -> str:
    """A named material with full control over its parameters (added once)."""
    if not any(m.name == name for m in scene.spec.materials):
        kw = {}
        if texture:
            kw = dict(textures=["", texture], texuniform=texuniform, texrepeat=list(texrepeat))
        scene.spec.add_material(name=name, rgba=list(rgba), emission=emission, specular=specular,
                                shininess=shininess, reflectance=reflectance, **kw)
    return name


def image_texture(scene, name: str, img: np.ndarray) -> str:
    """Add an RGB uint8 image (h, w, 3) as a 2D texture (once)."""
    if not any(t.name == name for t in scene.spec.textures):
        img = np.ascontiguousarray(np.asarray(img, np.uint8)[..., :3])
        t = scene.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                                   height=img.shape[0], nchannel=3)
        t.data = img.tobytes()
    return name


def _font(px: int, prefer: Sequence[str] = ()):
    from PIL import ImageFont

    cands = list(prefer) + ["/System/Library/Fonts/Menlo.ttc",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, px)
        except OSError:
            continue
    try:  # Pillow >= 10.1 ships a scalable default font
        return ImageFont.load_default(size=px)
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def label_image(lines, size=(256, 128), bg=(22, 22, 24), fg=(236, 236, 230), font_px: int = 80,
                border: int = 0, prefer_fonts: Sequence[str] = ()) -> np.ndarray:
    from PIL import Image, ImageDraw

    lines = [lines] if isinstance(lines, str) else list(lines)
    im = Image.new("RGB", size, tuple(bg))
    dr = ImageDraw.Draw(im)
    if border:
        dr.rectangle([border // 2, border // 2, size[0] - 1 - border // 2, size[1] - 1 - border // 2],
                     outline=tuple(fg), width=border)
    f = _font(font_px, prefer_fonts)
    boxes = [dr.textbbox((0, 0), ln, font=f) for ln in lines]
    heights = [b[3] - b[1] for b in boxes]
    gap = int(0.25 * font_px)
    total = sum(heights) + gap * (len(lines) - 1)
    y = (size[1] - total) / 2
    for ln, b, h in zip(lines, boxes, heights):
        w = b[2] - b[0]
        dr.text(((size[0] - w) / 2 - b[0], y - b[1]), ln, font=f, fill=tuple(fg))
        y += h + gap
    return np.asarray(im)


def label_material(scene, name: str, lines, *, size=(256, 128), bg=(22, 22, 24), fg=(236, 236, 230),
                   font_px: int = 80, border: int = 0, specular: float = 0.2, shininess: float = 0.3,
                   emission: float = 0.0, prefer_fonts: Sequence[str] = ()) -> str:
    tex = f"txl_{name}"
    image_texture(scene, tex, label_image(lines, size, bg, fg, font_px, border, prefer_fonts))
    return custom_material(scene, f"txlm_{name}", (1, 1, 1, 1), specular=specular, shininess=shininess,
                           emission=emission, texture=tex, texuniform=False)


def label_geom(scene, body, material: str, *, pos, half_w: float, half_h: float, thick: float = 0.0005,
               up_axis: str = "x", name: str | None = None):
    """Thin visual box whose +z face shows ``material``'s image upright. ``half_w`` x ``half_h`` = the face's half
    width / height as read; ``up_axis`` = the body-frame axis ("x", "-x", "y", "-y") pointing to the top of the
    text (the text normal is the body's +z)."""
    # the +z face maps the image with its width along the box's local x and its up along local y (checked by
    # rendering); rotate the box about z so that its local y points along ``up_axis``
    yaw = {"y": 0.0, "x": -math.pi / 2, "-y": math.pi, "-x": math.pi / 2}[up_axis]
    return V.visual_geom(scene, body, gtype=BOX, size=[half_w, half_h, thick], pos=pos, quat=quat_yaw(yaw),
                         material=material, name=name)


def ring_mesh(scene, name: str, r_in: float, r_out: float, h: float, segments: int = 32) -> str:
    """Flat ring (washer / bezel) of height h standing on z = 0 (visual)."""
    prof = [(r_in, 0.0), (r_out, 0.0), (r_out, h * 0.6), (r_out - 0.25 * (r_out - r_in), h), (r_in, h),
            (r_in, 0.0)]
    return V.lathe_mesh(scene, name, prof, segments=segments, cap_bottom=False)


def robot_touching(env, geoms: set[int]) -> set[int]:
    """Subset of ``geoms`` in contact with any robot geom."""
    d = env.data
    rg = env.robot.geom_ids
    out: set[int] = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if a in rg and b in geoms:
            out.add(b)
        elif b in rg and a in geoms:
            out.add(a)
    return out


def free_speed(env, body: str) -> tuple[float, float]:
    m = env.model
    j = m.body(body).jntadr[0]
    a = m.jnt_dofadr[j]
    v = env.data.qvel[a:a + 6]
    return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))


def set_free(env, body: str, pos, quat=(1, 0, 0, 0)) -> None:
    m, d = env.model, env.data
    j = m.body(body).jntadr[0]
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = pos
    d.qpos[a + 3:a + 7] = quat
    d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0


def task_status(sk) -> dict:
    """The public task status (``observation['task']``) agents see after every call."""
    return sk.env.observe(images=False)["task"]
