"""Procedural look-development for RoboticsBench scenes: textures, materials, meshes and lighting.

Everything is generated in memory (no asset files): seamless noise-based textures (wood, concrete, metal, rubber,
grass, tiles, ...), material presets, mesh generators (lathe / surfaces of revolution, extruded polygons, rounded
boxes) and lighting rigs. Visual-only geometry never collides (``contype = conaffinity = 0``); physical props keep
simple collision primitives and get a detailed visual mesh on top where it matters.

    from roboticsbench.core import visuals as V
    mat = V.material(scene, "oak")                         # textured wood
    cup = V.lathe_mesh(scene, "cup", V.CUP_PROFILE)       # mesh name
    V.visual_mesh(scene, body, cup, material=V.material(scene, "ceramic_white"))
    V.lights(scene, "studio")
"""

from __future__ import annotations

import math
import zlib
from functools import lru_cache
from typing import Any, Sequence

import mujoco
import numpy as np

GEOM = mujoco.mjtGeom


# ============================================================================== noise & texture generators
def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def value_noise(h: int, w: int, cy: int, cx: int, rng: np.random.Generator) -> np.ndarray:
    """Seamlessly tiling value noise with ``cy`` x ``cx`` lattice cells over an h x w image, in [0, 1]."""
    g = rng.random((cy, cx))
    yy = np.arange(h) * cy / h
    xx = np.arange(w) * cx / w
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    fy = _smooth(yy - y0)[:, None]
    fx = _smooth(xx - x0)[None, :]
    y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
    y0, x0 = y0 % cy, x0 % cx
    a, b = g[y0][:, x0], g[y0][:, x1]
    c, d = g[y1][:, x0], g[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(h: int, w: int, cy: int, cx: int, octaves: int, rng: np.random.Generator, gain: float = 0.5) -> np.ndarray:
    """Fractal (multi-octave) tileable noise normalised to [0, 1]."""
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(h, w, cy * 2 ** o, cx * 2 ** o, rng)
        total += amp
        amp *= gain
    out /= total
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-9)


def _ramp(t: np.ndarray, c0: Sequence[float], c1: Sequence[float]) -> np.ndarray:
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    return c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def tex_wood(size: int, rng: np.random.Generator, light=(196, 150, 104), dark=(128, 84, 50), planks: int = 4):
    """Planks running along x with long grain, ring figure and per-plank tone variation."""
    h = w = size
    grain = fbm(h, w, 48, 2, 4, rng)  # long streaks along x
    warp = fbm(h, w, 6, 3, 3, rng)
    yy = np.arange(h)[:, None] / h
    rings = 0.5 + 0.5 * np.sin((yy * 22 + warp * 3.0) * 2 * np.pi)
    t = 0.55 * grain + 0.30 * rings ** 4 + 0.15 * fbm(h, w, 64, 64, 2, rng)
    img = _ramp(np.clip(t, 0, 1), light, dark)
    band = h // planks
    for k in range(planks):
        img[k * band:(k + 1) * band] *= 0.88 + 0.24 * rng.random()
        img[k * band:k * band + 2] *= 0.55  # seam between planks
    return _u8(img)


def tex_concrete(size: int, rng: np.random.Generator, base=(150, 148, 142)):
    h = w = size
    t = 0.6 * fbm(h, w, 6, 6, 5, rng) + 0.4 * fbm(h, w, 64, 64, 2, rng)
    img = np.asarray(base, float)[None, None] * (0.78 + 0.35 * t[..., None])
    pores = rng.random((h, w)) > 0.9965
    img[pores] *= 0.55
    return _u8(img)


def tex_brushed(size: int, rng: np.random.Generator, base=(170, 173, 178)):
    h = w = size
    streak = fbm(h, w, 256, 4, 3, rng)
    t = 0.7 + 0.3 * streak
    return _u8(np.asarray(base, float)[None, None] * t[..., None])


def tex_rubber(size: int, rng: np.random.Generator, base=(48, 50, 54), dots: int = 32):
    h = w = size
    img = np.asarray(base, float)[None, None] * (0.85 + 0.25 * fbm(h, w, 32, 32, 3, rng)[..., None])
    yy, xx = np.mgrid[0:h, 0:w]
    p = size / dots
    d = np.hypot((yy % p) - p / 2, (xx % p) - p / 2)
    img[d < p * 0.22] *= 1.25
    return _u8(img)


def tex_grass(size: int, rng: np.random.Generator):
    h = w = size
    low = fbm(h, w, 4, 4, 4, rng)
    blades = fbm(h, w, 160, 90, 2, rng)
    t = 0.55 * blades + 0.45 * low
    return _u8(_ramp(t, (54, 118, 44), (112, 168, 70)))


def tex_tiles(size: int, rng: np.random.Generator, n: int = 4, base=(226, 224, 218), grout=(160, 158, 152),
              jitter: float = 0.06):
    h = w = size
    img = np.zeros((h, w, 3))
    cell = size // n
    for i in range(n):
        for j in range(n):
            c = np.asarray(base, float) * (1 - jitter + 2 * jitter * rng.random())
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = c
    img *= (0.94 + 0.08 * fbm(h, w, 16, 16, 3, rng))[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    g = max(2, size // 128)
    mask = ((yy % cell) < g) | ((xx % cell) < g)
    img[mask] = grout
    return _u8(img)


def tex_paint(size: int, rng: np.random.Generator, base=(222, 218, 208)):
    h = w = size
    t = fbm(h, w, 8, 8, 4, rng)
    return _u8(np.asarray(base, float)[None, None] * (0.95 + 0.07 * t[..., None]))


def tex_pegboard(size: int, rng: np.random.Generator, base=(196, 166, 124), holes: int = 16):
    img = tex_paint(size, rng, base).astype(float)
    yy, xx = np.mgrid[0:size, 0:size]
    p = size / holes
    d = np.hypot((yy % p) - p / 2, (xx % p) - p / 2)
    img[d < p * 0.16] = (58, 46, 36)
    return _u8(img)


def tex_cardboard(size: int, rng: np.random.Generator):
    h = w = size
    fibers = fbm(h, w, 90, 6, 3, rng)
    t = 0.7 * fibers + 0.3 * fbm(h, w, 8, 8, 3, rng)
    img = _ramp(t, (196, 158, 108), (160, 120, 76))
    yy = np.arange(h)[:, None]
    img *= (0.97 + 0.03 * np.sin(yy / h * 2 * np.pi * 40))[..., None]
    return _u8(img)


def tex_felt(size: int, rng: np.random.Generator, base=(40, 110, 66)):
    t = fbm(size, size, 96, 96, 2, rng)
    return _u8(np.asarray(base, float)[None, None] * (0.88 + 0.2 * t[..., None]))


def tex_asphalt(size: int, rng: np.random.Generator):
    img = tex_concrete(size, rng, base=(78, 80, 84)).astype(float)
    spk = rng.random((size, size)) > 0.985
    img[spk] *= 1.6
    return _u8(img)


def tex_stripes(size: int, rng: np.random.Generator, c0=(242, 196, 24), c1=(28, 28, 30), n: int = 6):
    yy, xx = np.mgrid[0:size, 0:size]
    band = ((xx + yy) * n // size) % 2
    img = np.where(band[..., None] == 0, np.asarray(c0, float), np.asarray(c1, float))
    img *= (0.94 + 0.08 * fbm(size, size, 16, 16, 2, rng))[..., None]
    return _u8(img)


def tex_marble(size: int, rng: np.random.Generator):
    h = w = size
    warp = fbm(h, w, 4, 4, 5, rng)
    xx = np.arange(w)[None, :] / w
    veins = np.abs(np.sin((xx * 3 + warp * 4) * np.pi))
    t = veins ** 0.35
    return _u8(_ramp(t, (120, 118, 124), (236, 234, 230)))


def tex_checker(size: int, rng: np.random.Generator, c0=(236, 236, 232), c1=(30, 30, 32), n: int = 8):
    yy, xx = np.mgrid[0:size, 0:size]
    m = ((yy * n // size) + (xx * n // size)) % 2
    return _u8(np.where(m[..., None] == 0, np.asarray(c0, float), np.asarray(c1, float)))


def with_grid(img: np.ndarray, cells: int, line=(70, 70, 72), width: int = 2, alpha: float = 0.55,
              offset: float = 0.0) -> np.ndarray:
    """Overlay thin grid lines (``cells`` per texture repeat, shifted by ``offset`` cells) on a texture image; lines
    are centred on their positions."""
    out = img.astype(float).copy()
    size = img.shape[0]
    idx = (((np.arange(cells) + offset) * size / cells) - width / 2).round().astype(int)
    for k in idx:
        for d in range(width):
            out[(k + d) % size, :] = out[(k + d) % size, :] * (1 - alpha) + np.asarray(line) * alpha
            out[:, (k + d) % size] = out[:, (k + d) % size] * (1 - alpha) + np.asarray(line) * alpha
    return _u8(out)


def tex_worktop(size: int, rng: np.random.Generator):
    """Light laminate work surface with a 10 cm grid (the texture spans 1 m with texrepeat 2)."""
    base = _u8(np.asarray((214, 210, 200), float)[None, None] * (0.95 + 0.06 * fbm(size, size, 24, 24, 3, rng)[..., None]))
    return with_grid(base, 10, line=(120, 116, 108), width=2, alpha=0.6, offset=0.5)


TEXTURE_KINDS = {
    "worktop_grid": lambda s, r: tex_worktop(s, r),
    # grid lines sit at whole multiples of the cell size in world coordinates (MuJoCo's uniform mapping puts the
    # texture seams half a repeat off the origin, hence offset=0.5): worktop every 10 cm, floors every 1 m
    "concrete_grid": lambda s, r: with_grid(tex_concrete(s, r), 1, line=(236, 214, 120), width=3, alpha=0.7,
                                            offset=0.5),
    "rubber_grid": lambda s, r: with_grid(tex_rubber(s, r), 1, line=(210, 210, 200), width=3, alpha=0.6, offset=0.5),
    "floor_wood_grid": lambda s, r: with_grid(tex_wood(s, r, light=(176, 128, 84), dark=(112, 72, 42), planks=6), 1,
                                              line=(240, 236, 220), width=3, alpha=0.5, offset=0.5),
    "grass_grid": lambda s, r: with_grid(tex_grass(s, r), 1, line=(236, 240, 228), width=2, alpha=0.35, offset=0.5),
    "oak": lambda s, r: tex_wood(s, r),
    "walnut": lambda s, r: tex_wood(s, r, light=(132, 92, 62), dark=(74, 46, 28)),
    "maple": lambda s, r: tex_wood(s, r, light=(228, 196, 150), dark=(186, 146, 100), planks=3),
    "plywood": lambda s, r: tex_wood(s, r, light=(214, 182, 136), dark=(170, 132, 88), planks=1),
    "floor_wood": lambda s, r: tex_wood(s, r, light=(176, 128, 84), dark=(112, 72, 42), planks=6),
    "concrete": lambda s, r: tex_concrete(s, r),
    "concrete_dark": lambda s, r: tex_concrete(s, r, base=(108, 108, 106)),
    "brushed_steel": lambda s, r: tex_brushed(s, r),
    "brushed_dark": lambda s, r: tex_brushed(s, r, base=(84, 88, 94)),
    "rubber": lambda s, r: tex_rubber(s, r),
    "rubber_blue": lambda s, r: tex_rubber(s, r, base=(42, 72, 120), dots=24),
    "grass": lambda s, r: tex_grass(s, r),
    "tiles_white": lambda s, r: tex_tiles(s, r),
    "tiles_grey": lambda s, r: tex_tiles(s, r, n=4, base=(150, 152, 156), grout=(90, 92, 96)),
    "tiles_lab": lambda s, r: tex_tiles(s, r, n=2, base=(200, 202, 204), grout=(150, 152, 156), jitter=0.03),
    "paint_white": lambda s, r: tex_paint(s, r),
    "paint_grey": lambda s, r: tex_paint(s, r, base=(150, 156, 164)),
    "paint_blue": lambda s, r: tex_paint(s, r, base=(76, 110, 150)),
    "paint_green": lambda s, r: tex_paint(s, r, base=(96, 132, 104)),
    "pegboard": lambda s, r: tex_pegboard(s, r),
    "cardboard": lambda s, r: tex_cardboard(s, r),
    "felt_green": lambda s, r: tex_felt(s, r),
    "felt_blue": lambda s, r: tex_felt(s, r, base=(40, 70, 130)),
    "asphalt": lambda s, r: tex_asphalt(s, r),
    "hazard": lambda s, r: tex_stripes(s, r),
    "marble": lambda s, r: tex_marble(s, r),
    "checker": lambda s, r: tex_checker(s, r),
}


@lru_cache(maxsize=64)
def texture_image(kind: str, seed: int = 0, size: int = 512) -> bytes:
    """RGB bytes of a procedural texture (cached across scenes)."""
    if kind not in TEXTURE_KINDS:
        raise KeyError(f"unknown texture kind {kind!r}; known: {sorted(TEXTURE_KINDS)}")
    img = TEXTURE_KINDS[kind](size, np.random.default_rng([seed, zlib.crc32(kind.encode())]))
    return np.ascontiguousarray(img).tobytes()


def texture(scene, kind: str, seed: int = 0, size: int = 512) -> str:
    """Add (once) a 2D texture to the scene's spec and return its name."""
    name = f"tex_{kind}_{seed}_{size}"
    if not any(t.name == name for t in scene.spec.textures):
        t = scene.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=size, height=size,
                                   nchannel=3)
        t.data = texture_image(kind, seed, size)
    return name


# ============================================================================== materials
# preset -> (texture kind or None, rgba, texrepeat (per metre when uniform), specular, shininess, reflectance)
MATERIALS: dict[str, dict[str, Any]] = {
    # functional surfaces with measuring grids (10 cm on work surfaces, 1 m on floors)
    # (with texuniform, one texture repeat spans 2 m / texrepeat: repeat 2 -> the texture covers exactly 1 m)
    "worktop_grid": dict(tex="worktop_grid", repeat=(2, 2), specular=0.25, shininess=0.4, reflectance=0.04),
    "concrete_grid": dict(tex="concrete_grid", repeat=(2, 2), specular=0.12, shininess=0.2, reflectance=0.03),
    "rubber_grid": dict(tex="rubber_grid", repeat=(2, 2), specular=0.1, shininess=0.2),
    "floor_wood_grid": dict(tex="floor_wood_grid", repeat=(2, 2), specular=0.3, shininess=0.45, reflectance=0.06),
    "grass_grid": dict(tex="grass_grid", repeat=(2, 2), specular=0.02, shininess=0.05),
    # woods & boards
    "oak": dict(tex="oak", repeat=(1.5, 1.5), specular=0.25, shininess=0.35, reflectance=0.03),
    "walnut": dict(tex="walnut", repeat=(1.5, 1.5), specular=0.3, shininess=0.4, reflectance=0.04),
    "maple": dict(tex="maple", repeat=(1.5, 1.5), specular=0.25, shininess=0.35, reflectance=0.03),
    "plywood": dict(tex="plywood", repeat=(2, 2), specular=0.15, shininess=0.2),
    "floor_wood": dict(tex="floor_wood", repeat=(0.5, 0.5), specular=0.3, shininess=0.45, reflectance=0.06),
    "cardboard": dict(tex="cardboard", repeat=(4, 4), specular=0.05, shininess=0.1),
    # stone & floors
    "concrete": dict(tex="concrete", repeat=(0.4, 0.4), specular=0.1, shininess=0.15, reflectance=0.02),
    "concrete_dark": dict(tex="concrete_dark", repeat=(0.4, 0.4), specular=0.1, shininess=0.15, reflectance=0.02),
    "tiles_white": dict(tex="tiles_white", repeat=(0.6, 0.6), specular=0.4, shininess=0.6, reflectance=0.05),
    "tiles_grey": dict(tex="tiles_grey", repeat=(0.6, 0.6), specular=0.35, shininess=0.5, reflectance=0.05),
    "tiles_lab": dict(tex="tiles_lab", repeat=(0.8, 0.8), specular=0.45, shininess=0.6, reflectance=0.08),
    "asphalt": dict(tex="asphalt", repeat=(0.5, 0.5), specular=0.05, shininess=0.1),
    "marble": dict(tex="marble", repeat=(1, 1), specular=0.6, shininess=0.8, reflectance=0.1),
    "grass": dict(tex="grass", repeat=(0.35, 0.35), specular=0.02, shininess=0.05),
    "rubber_mat": dict(tex="rubber", repeat=(1, 1), specular=0.1, shininess=0.2),
    "rubber_blue": dict(tex="rubber_blue", repeat=(1, 1), specular=0.1, shininess=0.2),
    # walls
    "paint_white": dict(tex="paint_white", repeat=(0.5, 0.5), specular=0.05, shininess=0.1),
    "paint_grey": dict(tex="paint_grey", repeat=(0.5, 0.5), specular=0.05, shininess=0.1),
    "paint_blue": dict(tex="paint_blue", repeat=(0.5, 0.5), specular=0.05, shininess=0.1),
    "paint_green": dict(tex="paint_green", repeat=(0.5, 0.5), specular=0.05, shininess=0.1),
    "pegboard": dict(tex="pegboard", repeat=(1, 1), specular=0.05, shininess=0.1),
    "hazard": dict(tex="hazard", repeat=(2, 2), specular=0.2, shininess=0.3),
    "checker": dict(tex="checker", repeat=(1, 1), specular=0.2, shininess=0.3),
    "felt_green": dict(tex="felt_green", repeat=(3, 3), specular=0.02, shininess=0.05),
    "felt_blue": dict(tex="felt_blue", repeat=(3, 3), specular=0.02, shininess=0.05),
    # metals
    "steel": dict(tex="brushed_steel", repeat=(4, 4), specular=0.9, shininess=0.85, reflectance=0.25),
    "steel_dark": dict(tex="brushed_dark", repeat=(4, 4), specular=0.8, shininess=0.8, reflectance=0.15),
    "aluminum": dict(rgba=(0.80, 0.82, 0.85, 1), specular=0.9, shininess=0.9, reflectance=0.3),
    "chrome": dict(rgba=(0.86, 0.88, 0.9, 1), specular=1.0, shininess=1.0, reflectance=0.6),
    "brass": dict(rgba=(0.78, 0.62, 0.30, 1), specular=0.9, shininess=0.85, reflectance=0.25),
    "copper": dict(rgba=(0.72, 0.42, 0.26, 1), specular=0.9, shininess=0.85, reflectance=0.2),
    "black_metal": dict(rgba=(0.10, 0.10, 0.11, 1), specular=0.6, shininess=0.7, reflectance=0.1),
    # others
    "ceramic_white": dict(rgba=(0.94, 0.93, 0.90, 1), specular=0.8, shininess=0.9, reflectance=0.08),
    "glass": dict(rgba=(0.75, 0.88, 0.95, 0.35), specular=1.0, shininess=1.0, reflectance=0.3),
    "rubber_black": dict(rgba=(0.08, 0.08, 0.09, 1), specular=0.15, shininess=0.2),
}

PLASTIC_COLORS = {
    "dark_grey": (0.22, 0.23, 0.25),
    "red": (0.80, 0.08, 0.07), "green": (0.12, 0.58, 0.20), "blue": (0.07, 0.30, 0.78), "yellow": (0.98, 0.78, 0.05),
    "orange": (0.97, 0.45, 0.05), "purple": (0.45, 0.18, 0.66), "white": (0.93, 0.93, 0.91),
    "black": (0.07, 0.07, 0.08), "grey": (0.52, 0.53, 0.55), "gray": (0.52, 0.53, 0.55), "cyan": (0.05, 0.62, 0.78),
    "pink": (0.95, 0.42, 0.66), "lime": (0.60, 0.80, 0.10), "tan": (0.85, 0.72, 0.52), "brown": (0.42, 0.25, 0.12),
    "dark_blue": (0.05, 0.12, 0.40), "dark_green": (0.05, 0.30, 0.12), "dark_red": (0.45, 0.05, 0.05),
}


def material(scene, preset: str, *, rgba: Sequence[float] | None = None, tint: Sequence[float] | None = None,
             repeat: Sequence[float] | None = None, emission: float = 0.0, seed: int = 0) -> str:
    """Material name for a preset (see ``MATERIALS``), or ``plastic_<colour>`` / ``metal_<colour>`` /
    ``rubber_<colour>`` / ``glow_<colour>`` for coloured finishes. Added to the spec once."""
    key = f"vm_{preset}_{seed}" + (f"_{'_'.join(f'{c:.2f}' for c in rgba)}" if rgba else "") + \
          (f"_t{'_'.join(f'{c:.2f}' for c in tint)}" if tint else "") + \
          (f"_r{'_'.join(f'{c:.2f}' for c in repeat)}" if repeat else "") + (f"_e{emission:.2f}" if emission else "")
    key = key.replace(".", "")
    if any(m.name == key for m in scene.spec.materials):
        return key
    kw: dict[str, Any] = {}
    if preset in MATERIALS:
        p = dict(MATERIALS[preset])
        if p.get("tex"):
            kw["textures"] = ["", texture(scene, p["tex"], seed)]
            kw["texuniform"] = True
            kw["texrepeat"] = list(repeat or p.get("repeat", (1, 1)))
        col = list(rgba or p.get("rgba", (1, 1, 1, 1)))
        spec_, shin, refl = p.get("specular", 0.3), p.get("shininess", 0.3), p.get("reflectance", 0.0)
    else:
        kind, _, colour = preset.partition("_")
        if kind not in ("plastic", "metal", "rubber", "glow", "matte") or colour not in PLASTIC_COLORS:
            raise KeyError(f"unknown material preset {preset!r}")
        col = [*PLASTIC_COLORS[colour], 1.0] if rgba is None else list(rgba)
        spec_, shin, refl = {"plastic": (0.55, 0.75, 0.06), "metal": (0.9, 0.85, 0.25), "rubber": (0.15, 0.2, 0.0),
                             "glow": (0.3, 0.3, 0.0), "matte": (0.1, 0.1, 0.0)}[kind]
        if kind == "glow":
            emission = max(emission, 0.6)
    if tint:
        col = [col[i] * tint[i] for i in range(3)] + [col[3]]
    if len(col) == 3:
        col = col + [1.0]
    scene.spec.add_material(name=key, rgba=col, specular=spec_, shininess=shin, reflectance=refl,
                            emission=emission, **kw)
    return key


# ============================================================================== meshes
def _mesh(scene, name: str, verts: np.ndarray, faces: np.ndarray) -> str:
    if not any(m.name == name for m in scene.spec.meshes):
        m = scene.spec.add_mesh(name=name)
        m.uservert = np.asarray(verts, float).reshape(-1).tolist()
        m.userface = np.asarray(faces, int).reshape(-1).tolist()
    return name


def lathe_mesh(scene, name: str, profile: Sequence[tuple[float, float]], segments: int = 48,
               cap_top: bool = False, cap_bottom: bool = True) -> str:
    """Surface of revolution about +z from a profile of (radius, z) points listed bottom to top along the surface
    (walk the outside up and, for vessels, back down the inside). Visual meshes may be non-convex."""
    prof = np.asarray(profile, float)
    n = len(prof)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = []
    for r, z in prof:
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    faces = []
    for i in range(n - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + j
            d = (i + 1) * segments + (j + 1) % segments
            faces += [(a, b, d), (a, d, c)]
    if cap_bottom and prof[0, 0] > 1e-6:
        ci = len(verts)
        verts.append((0.0, 0.0, prof[0, 1]))
        for j in range(segments):
            faces.append((ci, (j + 1) % segments, j))
    if cap_top and prof[-1, 0] > 1e-6:
        ci = len(verts)
        verts.append((0.0, 0.0, prof[-1, 1]))
        base = (n - 1) * segments
        for j in range(segments):
            faces.append((ci, base + j, base + (j + 1) % segments))
    return _mesh(scene, name, np.array(verts), np.array(faces))


def _triangulate(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple counter-clockwise polygon."""
    idx = list(range(len(poly)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def extrude_mesh(scene, name: str, polygon: Sequence[tuple[float, float]], height: float, z0: float = 0.0) -> str:
    """Prism from a simple polygon (counter-clockwise, x/y in metres) from z0 to z0 + height."""
    poly = np.asarray(polygon, float)
    if sum(poly[i - 1, 0] * poly[i, 1] - poly[i, 0] * poly[i - 1, 1] for i in range(len(poly))) < 0:
        poly = poly[::-1]
    n = len(poly)
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z0 + height) for x, y in poly]
    faces = []
    for a, b, c in _triangulate(poly):
        faces.append((a + n, b + n, c + n))
        faces.append((c, b, a))
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, j + n), (i, j + n, i + n)]
    return _mesh(scene, name, np.array(verts), np.array(faces))


def rounded_box_mesh(scene, name: str, half: Sequence[float], radius: float, segments: int = 6) -> str:
    """Box with rounded edges and corners (radius in metres) - a convex mesh, usable for collision too."""
    hx, hy, hz = half
    r = min(radius, hx * 0.99, hy * 0.99, hz * 0.99)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                cx, cy, cz = sx * (hx - r), sy * (hy - r), sz * (hz - r)
                for i in range(segments + 1):
                    th = (math.pi / 2) * i / segments
                    for j in range(segments + 1):
                        ph = (math.pi / 2) * j / segments
                        pts.append((cx + sx * r * math.cos(th) * math.cos(ph), cy + sy * r * math.sin(th) *
                                    math.cos(ph), cz + sz * r * math.sin(ph)))
    return _convex_mesh(scene, name, np.array(pts))


def _convex_mesh(scene, name: str, pts: np.ndarray) -> str:
    """Mesh from a point cloud; MuJoCo builds the convex hull (no faces needed)."""
    if not any(m.name == name for m in scene.spec.meshes):
        m = scene.spec.add_mesh(name=name)
        m.uservert = np.asarray(pts, float).reshape(-1).tolist()
    return name


def gear_polygon(teeth: int, r_root: float, r_tip: float, samples: int = 4) -> list[tuple[float, float]]:
    pts = []
    for k in range(teeth):
        a0 = 2 * math.pi * k / teeth
        for i in range(samples):  # root arc
            a = a0 + (2 * math.pi / teeth) * 0.5 * i / samples
            pts.append((r_root * math.cos(a), r_root * math.sin(a)))
        for i in range(samples):  # tooth top
            a = a0 + (2 * math.pi / teeth) * (0.55 + 0.35 * i / max(1, samples - 1))
            pts.append((r_tip * math.cos(a), r_tip * math.sin(a)))
    return pts


# common lathe profiles (metres): (radius, z) bottom -> top (outside), then back down the inside for vessels
def cup_profile(r: float = 0.035, h: float = 0.09, wall: float = 0.004, flare: float = 1.12) -> list:
    return [(0.0, 0.0), (r * 0.82, 0.0), (r * 0.9, 0.004), (r, h * 0.35), (r * flare, h), (r * flare - wall, h),
            (r - wall, h * 0.35), (r * 0.9 - wall, 0.008), (0.0, 0.008)]


def bowl_profile(r: float = 0.09, h: float = 0.05, wall: float = 0.005) -> list:
    out = [(0.0, 0.0), (r * 0.45, 0.0)]
    for i in range(1, 9):
        a = (math.pi / 2) * i / 8
        out.append((r * 0.45 + (r - r * 0.45) * math.sin(a), h * (1 - math.cos(a))))
    inner = [(rr - wall, max(z, wall)) for rr, z in reversed(out[2:])]
    return out + inner + [(0.0, wall)]


def egg_profile(r: float = 0.022, h: float = 0.058, n: int = 16) -> list:
    pts = []
    for i in range(n + 1):
        t = i / n
        z = h * t
        rr = r * math.sin(math.pi * t) ** 0.8 * (1.0 + 0.12 * (0.5 - t))
        pts.append((max(rr, 1e-4 if 0 < i < n else 0.0), z))
    return pts


# ============================================================================== geoms
def visual_geom(scene, body, *, gtype, size=(0.01, 0.01, 0.01), pos=(0, 0, 0), quat=None, material: str,
                mesh: str | None = None, name: str | None = None, fromto=None, group: int = 1):
    """Decorative geometry: never collides, never adds mass."""
    kw: dict[str, Any] = dict(type=gtype, material=material, contype=0, conaffinity=0, density=0, group=group)
    if gtype == GEOM.mjGEOM_MESH:
        kw["meshname"] = mesh
    else:
        kw["size"] = list(size) + [0] * (3 - len(size))
    if fromto is not None:
        kw["fromto"] = list(fromto)
    else:
        kw["pos"] = list(pos)
    if quat is not None:
        kw["quat"] = list(quat)
    if name:
        kw["name"] = name
    return body.add_geom(**kw)


def visual_mesh(scene, body, mesh: str, *, material: str, pos=(0, 0, 0), quat=None, name: str | None = None):
    return visual_geom(scene, body, gtype=GEOM.mjGEOM_MESH, mesh=mesh, material=material, pos=pos, quat=quat,
                       name=name)


def hide_collision(geom) -> None:
    """Keep a collision geom but do not render it (group 3 is hidden in agent/camera renders)."""
    geom.group = 3


# ============================================================================== lighting & rendering
def render_quality(scene, *, samples: int = 8, shadow: int = 4096, haze=None) -> None:
    v = scene.spec.visual
    v.quality.offsamples = samples
    v.quality.shadowsize = shadow
    v.map.shadowclip = 2.0
    v.map.shadowscale = 0.6
    if haze is not None:
        v.rgba.haze = list(haze)


def lights(scene, rig: str = "studio", *, center=(0.0, 0.0, 0.0), extent: float = 3.0) -> None:
    """Replace the default lights with a rig. ``studio`` (arm tables), ``hall`` (large indoor spaces with soft
    overheads), ``outdoor`` (sun + sky fill)."""
    wb = scene.spec.worldbody
    for lt in list(scene.spec.lights):
        scene.spec.delete(lt)
    cx, cy, cz = center
    L = mujoco.mjtLightType
    hv = scene.spec.visual.headlight
    if rig == "studio":
        hv.ambient, hv.diffuse, hv.specular = [0.18] * 3, [0.14] * 3, [0.04] * 3
        wb.add_light(name="key", pos=[cx + 1.2, cy - 1.4, cz + 2.6], dir=[-0.45, 0.5, -1],
                     type=L.mjLIGHT_DIRECTIONAL, castshadow=False, diffuse=[0.34, 0.33, 0.31],
                     specular=[0.2, 0.2, 0.2])
        # crisp shadows over the work area from a spot above the table
        sp = wb.add_light(name="shadow_spot", pos=[cx + 0.9, cy - 0.9, cz + 2.4], dir=[-0.35, 0.35, -1],
                          type=L.mjLIGHT_SPOT, cutoff=38, exponent=2, castshadow=True, diffuse=[0.36, 0.35, 0.33],
                          specular=[0.15, 0.15, 0.15], attenuation=[1, 0, 0])
        sp.softness = 0.4
        wb.add_light(name="fill", pos=[cx - 1.5, cy + 1.5, cz + 2.2], dir=[0.5, -0.4, -1],
                     type=L.mjLIGHT_DIRECTIONAL, castshadow=False, diffuse=[0.22, 0.23, 0.26], specular=[0, 0, 0])
        wb.add_light(name="rim", pos=[cx - 2.0, cy, cz + 1.5], dir=[1, 0, -0.4], type=L.mjLIGHT_DIRECTIONAL,
                     castshadow=False, diffuse=[0.14, 0.14, 0.15], specular=[0.1, 0.1, 0.1])
    elif rig == "hall":
        hv.ambient, hv.diffuse, hv.specular = [0.24] * 3, [0.16] * 3, [0.04] * 3
        wb.add_light(name="key", pos=[cx + 3, cy - 4, cz + 9], dir=[-0.3, 0.4, -1], type=L.mjLIGHT_DIRECTIONAL,
                     castshadow=False, diffuse=[0.36, 0.35, 0.33], specular=[0.12, 0.12, 0.12])
        _tracking_shadow(scene, diffuse=(0.36, 0.35, 0.33))
        for i, (dx, dy) in enumerate(((-1, -1), (1, 1), (-1, 1), (1, -1))):
            wb.add_light(name=f"overhead{i}", pos=[cx + dx * extent * 0.4, cy + dy * extent * 0.4, cz + 6],
                         dir=[0, 0, -1], type=L.mjLIGHT_SPOT, cutoff=60, exponent=4, castshadow=False,
                         diffuse=[0.14, 0.14, 0.13], specular=[0, 0, 0], attenuation=[1, 0, 0])
    elif rig == "outdoor":
        hv.ambient, hv.diffuse, hv.specular = [0.28, 0.29, 0.31], [0.14] * 3, [0.05] * 3
        wb.add_light(name="sun", pos=[cx + 4, cy - 3, cz + 10], dir=[-0.4, 0.3, -1],
                     type=L.mjLIGHT_DIRECTIONAL, castshadow=False, diffuse=[0.46, 0.43, 0.38],
                     specular=[0.3, 0.3, 0.28])
        _tracking_shadow(scene, diffuse=(0.34, 0.32, 0.28))
        wb.add_light(name="skyfill", pos=[cx - 4, cy + 3, cz + 8], dir=[0.4, -0.3, -1], type=L.mjLIGHT_DIRECTIONAL,
                     castshadow=False, diffuse=[0.18, 0.2, 0.26], specular=[0, 0, 0])
    else:
        raise KeyError(f"unknown light rig {rig!r}")


def _tracking_shadow(scene, diffuse=(0.35, 0.35, 0.33), body: str = "pelvis") -> None:
    """A shadow-casting spot that follows the robot (MuJoCo 'trackcom' light): crisp shadows wherever the robot is,
    independent of how large the scene is. Falls back to a fixed spot when the robot body is absent."""
    L = mujoco.mjtLightType
    try:
        parent = scene.spec.body(body)
    except Exception:  # noqa: BLE001
        parent = None
    host = parent if parent is not None else scene.spec.worldbody
    sp = host.add_light(name="shadow_spot", pos=[1.6, -2.0, 5.5], dir=[-0.27, 0.34, -1], type=L.mjLIGHT_SPOT,
                        cutoff=32, exponent=1, castshadow=True, diffuse=list(diffuse), specular=[0.1, 0.1, 0.1],
                        attenuation=[1, 0, 0])
    sp.softness = 0.35
    if parent is not None:
        sp.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM


# ============================================================================== text (labels, signs, notes)
def _pil_font(px: int):
    from PIL import ImageFont

    for cand in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, px)
        except OSError:
            continue
    try:  # Pillow >= 10.1: scalable built-in font (the legacy bitmap default is tiny)
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def text_image(lines: Sequence[str] | str, size=(512, 256), bg=(246, 242, 226), fg=(22, 22, 30),
               font_px: int = 72, border: int = 10) -> np.ndarray:
    from PIL import Image, ImageDraw

    lines = [lines] if isinstance(lines, str) else list(lines)
    im = Image.new("RGB", size, tuple(bg))
    dr = ImageDraw.Draw(im)
    if border:
        dr.rectangle([border // 2, border // 2, size[0] - border // 2, size[1] - border // 2], outline=tuple(fg),
                     width=max(2, border // 3))
    f = _pil_font(font_px)
    heights = [dr.textbbox((0, 0), ln, font=f)[3] for ln in lines]
    total = sum(heights) + 12 * (len(lines) - 1)
    y = (size[1] - total) / 2
    for ln, h in zip(lines, heights):
        w = dr.textlength(ln, font=f)
        dr.text(((size[0] - w) / 2, y), ln, font=f, fill=tuple(fg))
        y += h + 12
    return np.asarray(im)


def text_material(scene, name: str, lines: Sequence[str] | str, *, size=(512, 256), bg=(246, 242, 226),
                  fg=(22, 22, 30), font_px: int = 72) -> str:
    """Material showing text, mapped once onto each face of a box (make the box's face aspect match ``size``).
    For signs, labels, notes (e.g. a combination code the agent must read from a camera image)."""
    tex = f"txt_{name}"
    if not any(t.name == tex for t in scene.spec.textures):
        img = text_image(lines, size, bg, fg, font_px)
        t = scene.spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D, width=size[0], height=size[1],
                                   nchannel=3)
        t.data = np.ascontiguousarray(img).tobytes()
    mat = f"txtmat_{name}"
    if not any(m.name == mat for m in scene.spec.materials):
        scene.spec.add_material(name=mat, textures=["", tex], texuniform=False, texrepeat=[1, 1], specular=0.1,
                                shininess=0.1)
    return mat
