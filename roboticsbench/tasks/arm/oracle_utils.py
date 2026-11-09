"""Reusable scripted manipulation routines for oracles (public skill API only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboticsbench.skills.base import SkillResult


class OracleFailure(RuntimeError):
    pass


class EpisodeOver(Exception):
    """Raised to unwind an oracle once the episode has ended (e.g. it already succeeded)."""


def need(res: SkillResult, what: str = "", sk=None) -> SkillResult:
    if not res.ok:
        if "[EPISODE ENDED" in res.message or "episode already ended" in res.message:
            raise EpisodeOver(res.message)
        raise OracleFailure(f"{what}: {res.message}")
    return res


def find(sk, name: str) -> dict[str, Any]:
    for o in sk.list_objects().data["objects"]:
        if o["name"] == name:
            return o
    raise OracleFailure(f"object {name} not found")


def grasp_yaw(obj_yaw_deg: float, current_deg: float = 0.0) -> float:
    """Pick the symmetric (mod 90) yaw closest to the current gripper yaw."""
    cands = [obj_yaw_deg + k * 90.0 for k in range(-3, 4)]
    cands = [c for c in cands if -135.0 <= c <= 135.0]
    return float(min(cands, key=lambda c: abs(c - current_deg)))


def pick(sk, name: str, *, grasp_z: float | None = None, approach: float = 0.1, lift: float = 0.12,
         force: float = 40.0, yaw_deg: float | None = None) -> None:
    o = find(sk, name)
    p = np.array(o["position"], float)
    yaw = yaw_deg if yaw_deg is not None else grasp_yaw(o.get("yaw_deg", 0.0))
    z = p[2] if grasp_z is None else grasp_z
    need(sk.open_gripper(), "open")
    need(sk.move_to(p[0], p[1], z + approach, yaw_deg=yaw), "approach")
    need(sk.move_to(p[0], p[1], z, yaw_deg=yaw, speed="slow"), "descend")
    need(sk.close_gripper(force), "close")
    need(sk.move_by(dz=lift), "lift")


def place(sk, xyz, *, approach: float = 0.1, yaw_deg: float | None = None, release_up: float = 0.1) -> None:
    x, y, z = xyz
    kw = {} if yaw_deg is None else {"yaw_deg": yaw_deg}
    need(sk.move_to(x, y, z + approach, **kw), "carry")
    need(sk.move_to(x, y, z, speed="slow", **kw), "lower")
    need(sk.open_gripper(), "release")
    need(sk.move_by(dz=release_up), "retreat")
