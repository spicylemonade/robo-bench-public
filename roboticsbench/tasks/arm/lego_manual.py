"""Brick assembly from an illustrated building manual.

A grey 8x8-stud baseplate (printed column letters A-H / row numbers 1-8 on its border) sits on the table among
loose bricks - the parts of a small model plus two decoys. The model exists only in the manual: PNG pages
(parts list, numbered steps with the new bricks outlined and a plan view, the finished model) handed to the agent
as documents. Bricks snap together through real stud / cavity geometry and a clutch that clicks when a seated
brick is pressed down (see ``_lego``). The reference solution knows the design (it stands in for reading the manual
perfectly) and builds it through the public skill API with closed-loop placement.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import mujoco
import numpy as np

from roboticsbench.tasks.arm import _lego as L
from roboticsbench.tasks.arm.common import ArmTask, build_table
from roboticsbench.tasks.arm.oracle_utils import OracleFailure, need
from roboticsbench.tasks.base import Evaluation, Milestones, register, set_free_pose

N_PLATE = 8
GRASP_DZ = 0.022  # TCP height above a brick's bottom face when gripping it (fingertips 1.35 cm above the bottom)
FALLEN_Z = -0.2


def cell_name(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def span_name(gp: L.GridPose) -> str:
    cells = gp.cells()
    c0, c1 = min(c for c, _ in cells), max(c for c, _ in cells)
    r0, r1 = min(r for _, r in cells), max(r for _, r in cells)
    return f"{cell_name(c0, r0)}-{cell_name(c1, r1)}"


class LegoBuildTask(ArmTask):
    """Shared scene / physics / scoring for brick-building tasks. Subclasses decide how the design is shown."""

    category = "assembly"
    difficulty = "expert"
    controls = ("skills", "motor")
    cameras = ("front", "top", "build_cam")
    video_camera = "build_cam"
    success_hold = 1.0
    n_bricks: tuple[int, int] = (9, 12)
    levels: tuple[int, int] = (3, 5)
    n_spares = 2

    # ------------------------------------------------------------------ build
    def make_design(self, rng) -> list[L.DesignBrick]:
        return L.generate_design(rng, n_plate=N_PLATE, n_bricks=self.n_bricks, levels=self.levels)

    def build(self, scene, rng):
        build_table(scene)
        # real time on one CPU core: noslip over ~150 stud contacts cost ~2/3 of the physics; without it (and with
        # softer friction cones) the reference build still passes 5/5 seeds at ~2.5x the speed (2026-10-09)
        scene.spec.option.noslip_iterations = 0
        scene.spec.option.impratio = 3
        self.design = self.make_design(rng)
        self.spares = L.pick_spares(rng, self.design, self.n_spares)
        self.plate_yaw = math.radians(float(rng.choice([0.0, 90.0, 180.0, 270.0])) + float(rng.uniform(-25, 25)))
        self.plate_pos = np.array([0.47 + rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.03), 0.0])
        L.add_baseplate(scene, "baseplate", self.plate_pos, self.plate_yaw, N_PLATE)
        parts = [(b.kind, b.color) for b in self.design] + list(self.spares)
        parts = [parts[i] for i in rng.permutation(len(parts))]
        seen: Counter = Counter()
        self.bricks: dict[str, tuple[str, str]] = {}
        for kind, color in parts:
            seen[(kind, color)] += 1
            k = seen[(kind, color)]
            self.bricks[f"{color}_{kind}" + (f"_{k}" if k > 1 else "")] = (kind, color)
        self.start = self._layout(rng)
        for nm, (kind, color) in self.bricks.items():
            x, y, yaw = self.start[nm]
            L.add_brick(scene, nm, kind, color, [x, y, 0.0], yaw)
        self.clutch = L.Clutch(scene, "baseplate", N_PLATE, {nm: kc[0] for nm, kc in self.bricks.items()})
        c = self.plate_pos
        scene.add_camera("build_cam", pos=[c[0] + 0.62, c[1] - 0.55, 0.52], target=[c[0] - 0.02, c[1], 0.05], fovy=42)
        self.instruction = self.make_instruction()
        ladder = [(f"bricks_{k}", round(0.9 * k / len(self.design), 3)) for k in range(1, len(self.design) + 1)]
        self.milestones = Milestones(ladder)

    def _layout(self, rng) -> dict[str, tuple[float, float, float]]:
        plate = L._rect_corners(self.plate_pos[0], self.plate_pos[1], self.plate_yaw,
                                N_PLATE * L.P / 2 + L.BORDER + 0.035, N_PLATE * L.P / 2 + L.BORDER + 0.035)
        for _attempt in range(50):
            out: dict[str, tuple[float, float, float]] = {}
            rects = [plate] + list(getattr(self, "keepout", []))
            # big bricks first (hardest to fit)
            names = sorted(self.bricks, key=lambda n: -np.prod(L.KINDS[self.bricks[n][0]]))
            ok = True
            for nm in names:
                w, l_ = L.KINDS[self.bricks[nm][0]]
                for _ in range(600):
                    x, y = rng.uniform(0.2, 0.66), rng.uniform(-0.5, 0.5)
                    r = math.hypot(x, y)
                    if r < 0.36 or r > 0.72:
                        continue
                    yaw = math.radians(rng.uniform(-90, 90))
                    rc = L._rect_corners(x, y, yaw, l_ * L.P / 2 + 0.024, w * L.P / 2 + 0.024)
                    if any(L._rects_overlap(rc, o) for o in rects):
                        continue
                    rects.append(L._rect_corners(x, y, yaw, l_ * L.P / 2 + 0.004, w * L.P / 2 + 0.004))
                    out[nm] = (float(x), float(y), float(yaw))
                    break
                else:
                    ok = False
                    break
            if ok:
                return out
        raise RuntimeError("could not lay out the loose bricks")

    def bind(self, env):
        self.clutch.bind(env)

    def reset(self, env, rng):
        self.clutch.reset(env)
        for nm, (x, y, yaw) in self.start.items():
            yaw += math.radians(rng.uniform(-3, 3))
            set_free_pose(env, nm, [x + rng.uniform(-0.002, 0.002), y + rng.uniform(-0.002, 0.002), 0.0005],
                          [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        self.best = 0

    # ------------------------------------------------------------------ instruction
    def design_text(self) -> str:
        raise NotImplementedError

    def make_instruction(self) -> str:
        sizes = ", ".join(f"{k} ({w * 3.2:.1f} x {l_ * 3.2:.1f} cm)" for k, (w, l_) in L.KINDS.items()
                          if k in {kc[0] for kc in self.bricks.values()})
        return (
            f"{self.design_text()} Build it on the grey baseplate: 8 x 8 studs, its border printed with column "
            f"letters A-H and row numbers 1-8 (list_objects() gives the plate's cell_A1_center and the column_step / "
            f"row_step vectors, so the footprint centre of a brick covering columns c0..c1 and rows r0..r1 is "
            f"cell_A1_center + (c0+c1)/2 * column_step + (r0+r1)/2 * row_step, with A = 0 and row 1 = 0). "
            f"Loose bricks lie on the table around the plate - there are more than you need; bricks of the same "
            f"size and colour are interchangeable. Brick sizes (width x length): {sizes}; stud pitch 3.2 cm, brick "
            f"height 3.84 cm plus 0.68 cm studs, masses 50-160 g. Brick positions from list_objects() are body "
            f"centres; yaw_deg is the direction of a brick's length. Each level of the model is 3.84 cm higher: a "
            f"brick on level L has its bottom face at the plate top + L * 3.84 cm (the plate is 4.8 mm thick and "
            f"lies flat on the table; its printed border is 2.6 cm wide around the 25.6 cm stud field). "
            f"CLICKING: a brick only locks onto the studs below it when it is seated flat, exactly on the stud grid "
            f"(the studs enter its hollow underside with 0.7 mm play per side; a 1.8 mm chamfer on the studs guides "
            f"it in if it is off by up to ~2.5 mm) and then pressed down with at least 6 N (e.g. keep holding it and "
            f"push 3-4 mm further down once it touches). A clicked brick holds on to the part below it (overhangs are "
            f"fine) and pops off again if pulled or twisted hard (~20 N) - open the gripper before moving away. "
            f"Unclicked bricks only rest loosely and do not count. list_objects() reports for every brick whether it "
            f"is clicked and, when it sits on the grid, its cells and level. "
            f"GRIPPING: the gripper opens to 8 cm, so grip bricks across their 3.2 or 6.4 cm sides (a 1x2 or 2x2 "
            f"brick also across its 6.4 cm length). Keep the TCP about 2.2 cm above a brick's bottom face so the "
            f"finger tips clear the studs below, and grip on sides whose neighbouring cells are still free - the "
            f"fingers need ~2.5 cm of room next to the brick. "
            f"SUCCESS: every brick of the model clicked in its place (size, colour, cells, level and orientation as "
            f"shown) and no other brick clicked onto the plate or the model; held for 1 s. Partial credit per "
            f"correctly placed brick.")

    # ------------------------------------------------------------------ monitoring
    def step(self, env):
        self.clutch.step(env)

    def _placements(self, env) -> tuple[int, int, dict[str, L.GridPose | None]]:
        poses = {nm: self.clutch.grid_pose(env, nm, tol_xy=0.004, tol_z=0.003) for nm in self.clutch.snapped()}
        need_c = Counter((b.kind, b.color, b.pose.canonical()) for b in self.design)
        have_c = Counter((self.bricks[nm][0], self.bricks[nm][1], gp.canonical()) for nm, gp in poses.items() if gp)
        matched = sum(min(n, have_c[k]) for k, n in need_c.items())
        extra = len(poses) - matched
        return matched, extra, poses

    def evaluate(self, env):
        matched, extra, poses = self._placements(env)
        for k in range(1, matched + 1):
            self.milestones.hit(f"bricks_{k}", env.t)
        ev = Evaluation(progress=self.milestones.progress,
                        metrics={"placed_correctly": matched, "model_bricks": len(self.design),
                                 "extra_clicked": extra, "clicked": len(poses)})
        # a part the model needs fell off the table and no spare of that size & colour is left
        avail = Counter(self.bricks[nm] for nm in self.bricks if env.data.body(nm).xpos[2] > FALLEN_Z)
        needed = Counter((b.kind, b.color) for b in self.design)
        for kc, n in needed.items():
            if avail[kc] < n:
                ev.failed = True
                ev.reason = f"a {kc[1]} {kc[0]} brick needed for the model fell off the table"
                return ev
        if matched == len(self.design) and extra == 0:
            ev.success = True
            ev.reason = "model complete"
        else:
            ev.reason = f"{matched}/{len(self.design)} bricks placed correctly" + \
                        (f", {extra} brick(s) clicked where the model has none" if extra else "")
        return ev

    # ------------------------------------------------------------------ agent-facing
    def objects(self, env):
        pp, Rp = self.clutch.plate_frame(env)
        top = pp + Rp[:, 2] * L.PLATE_T
        a1 = pp + Rp @ np.array([-(N_PLATE - 1) * L.P / 2, -(N_PLATE - 1) * L.P / 2, L.PLATE_T])
        out: list[dict[str, Any]] = [{
            "name": "baseplate", "shape": "8 x 8 stud baseplate (fixed)", "color": "grey",
            "position": [round(float(v), 4) for v in top], "size_m": [0.308, 0.308, L.PLATE_T],
            "yaw_deg": round(math.degrees(math.atan2(Rp[1, 0], Rp[0, 0])), 2),
            "cell_A1_center": [round(float(v), 4) for v in a1],
            "column_step": [round(float(v), 5) for v in Rp[:, 0] * L.P],
            "row_step": [round(float(v), 5) for v in Rp[:, 1] * L.P],
            "note": "position = centre of the top face (studs stand on it); cell_A1_center is on the top face"}]
        d = env.data
        for nm, (kind, color) in self.bricks.items():
            b = d.body(nm)
            R = b.xmat.reshape(3, 3)
            c = b.xpos + R[:, 2] * (L.H / 2)
            w, l_ = L.KINDS[kind]
            o = {"name": nm, "shape": f"{kind} brick", "color": color, "studs": kind,
                 "position": [round(float(v), 4) for v in c], "size_m": [l_ * L.P, w * L.P, L.H],
                 "yaw_deg": round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 1),
                 "tilt_deg": round(math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1)))), 1),
                 "clicked": bool(self.clutch.parent.get(nm))}
            gp = self.clutch.grid_pose(env, nm, tol_xy=0.004, tol_z=0.003)
            if gp is not None:
                o["on_grid"] = {"cells": span_name(gp), "level": gp.level}
            out.append(o)
        return out

    def status(self, env):
        if getattr(env, "vision_only", False):
            # camera-only perception: the robot feels / hears a click or a brick popping off, nothing more
            ev = [e.split(" ")[0] + (" a brick clicked in" if "clicked" in e else " a brick popped off")
                  for e in self.clutch.events[-3:]]
            return {"recent_events": ev}
        clicked = {}
        for nm in self.clutch.snapped():
            gp = self.clutch.grid_pose(env, nm, tol_xy=0.004, tol_z=0.003)
            clicked[nm] = f"{span_name(gp)} level {gp.level}" if gp else "off-grid"
        return {"clicked": clicked, "recent_events": self.clutch.events[-3:]}

    # ------------------------------------------------------------------ reference solution
    def oracle(self, sk):
        """Build ``self.design`` in order (knowing the design = reading the manual perfectly) with the public skill
        API: pick each brick with a side grip, carry it over its target, align it closed-loop from list_objects(),
        lower, press to click, release."""
        placed: list[L.GridPose] = []
        used: set[str] = set()
        for b in self.design:
            self._oracle_place(sk, b, placed, used)
            placed.append(b.pose)
        sk.wait(2.0)

    @staticmethod
    def _reach(sk, x, y, z, yaws, what) -> float:
        """move_to trying equivalent gripper yaws (the local IK sometimes cannot reach a far wrist rotation)."""
        msg = ""
        for yaw in yaws:
            res = sk.move_to(x, y, z, yaw_deg=yaw, speed="normal")
            if res.ok:
                return float(yaw)
            msg = res.message
            if "out of reach" not in msg:
                break
        raise OracleFailure(f"{what}: {msg}")

    def _objs(self, sk) -> dict[str, dict]:
        return {o["name"]: o for o in sk.list_objects().data["objects"]}

    def _oracle_place(self, sk, b: L.DesignBrick, placed: list[L.GridPose], used: set[str]) -> None:
        objs = self._objs(sk)
        plate = objs["baseplate"]
        a1 = np.array(plate["cell_A1_center"])
        cs, rs = np.array(plate["column_step"]), np.array(plate["row_step"])
        plate_yaw = plate["yaw_deg"]
        cells = b.pose.cells()
        c0, c1 = min(c for c, _ in cells), max(c for c, _ in cells)
        r0, r1 = min(r for _, r in cells), max(r for _, r in cells)
        foot = a1 + (c0 + c1) / 2 * cs + (r0 + r1) / 2 * rs
        seat_z = float(a1[2]) + b.pose.level * L.H  # bottom face of the placed brick
        tgt_yaw = plate_yaw + (0.0 if b.pose.along == "x" else 90.0)
        # the plate axis the fingers must close along at the target (its neighbour cells must be free)
        axes = L.grasp_axes(b.pose, [p for p in placed if p.level == b.pose.level])
        w, l_ = L.KINDS[b.kind]
        square = w == l_
        across = "y" if b.pose.along == "x" else "x"
        if square and axes:
            close_axis, width_grip = (across if across in axes else axes[0]), True
        elif across in axes:
            close_axis, width_grip = across, True  # fingers on the long sides
        elif l_ <= L.MAX_GRIP_STUDS and b.pose.along in axes:
            close_axis, width_grip = b.pose.along, False  # fingers on the ends (1x2)
        else:
            raise OracleFailure(f"no free grip for {b}")
        # choose the closest free brick of this size & colour
        cands = [n for n, o in objs.items() if n != "baseplate" and n not in used and o["studs"] == b.kind
                 and o["color"] == b.color and not o["clicked"] and o["position"][2] > -0.1]
        if not cands:
            raise OracleFailure(f"no {b.color} {b.kind} left")
        name = min(cands, key=lambda n: np.linalg.norm(np.array(objs[n]["position"][:2]) - foot[:2]))
        used.add(name)
        o = objs[name]
        p = np.array(o["position"])
        by = o["yaw_deg"]
        period = 90.0 if square else 180.0  # brick symmetry
        # gripper yaw g closes the fingers along (-sin g, cos g): across the brick's width -> g = brick yaw
        g_pick = L.best_yaw(by if width_grip else by + 90.0, period, p[0], p[1])
        model_top = float(a1[2]) + (max([q.level for q in placed], default=-1) + 1) * L.H + L.STUD_H
        clear = max(0.16, model_top + 0.1)
        open_cm = 8.0 if (w if width_grip else l_) == 2 else 6.0
        bottom = p[2] - L.H / 2
        need(sk.open_gripper(open_cm), "open")
        g_pick = self._reach(sk, p[0], p[1], clear, [g_pick, g_pick + period, g_pick - period], "above brick")
        need(sk.move_to(p[0], p[1], bottom + GRASP_DZ + 0.03, yaw_deg=g_pick), "pre-grasp")
        need(sk.move_to(p[0], p[1], bottom + GRASP_DZ, yaw_deg=g_pick, speed="slow"), "descend")
        res = need(sk.close_gripper(35.0), "grip")
        if "holding something" not in res.message:
            raise OracleFailure(f"missed {name}: {res.message}")
        need(sk.move_to(p[0], p[1], clear, yaw_deg=g_pick, speed="slow"), "lift")
        # where the brick sits in the hand (from list_objects + proprioception), in the gripper frame
        o = self._objs(sk)[name]
        base = plate_yaw if close_axis == "y" else plate_yaw - 90.0  # fingers closing along that plate axis
        g_place = L.best_yaw(base, 180.0, foot[0], foot[1], near=g_pick)
        tcp = np.array(sk.robot.proprio()["gripper_position"])
        dz = tcp[2] - (np.array(o["position"])[2] - L.H / 2)  # TCP height above the brick bottom
        off_w = np.array(o["position"][:2]) - tcp[:2]
        a = math.radians(g_place - g_pick)
        off_p = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ off_w
        x, y = foot[0] - off_p[0], foot[1] - off_p[1]
        g_place = self._reach(sk, x, y, clear, [g_place, g_place + 180.0, g_place - 180.0], "carry")
        # closed-loop alignment just above the tallest neighbour this brick could touch (same-level bricks stand
        # 2 mm away), then straight down; a gripped brick lags fast wrist turns by a few degrees
        hover = seat_z + L.H + L.STUD_H + 0.006
        yaw_cmd = g_place
        for _ in range(7):
            need(sk.move_to(x, y, hover + dz, yaw_deg=yaw_cmd, speed="slow"), "hover")
            o = self._objs(sk)[name]
            err = foot[:2] - np.array(o["position"][:2])
            yerr = L.wrap180(((tgt_yaw - o["yaw_deg"]) + period / 2) % period - period / 2)
            if np.linalg.norm(err) < 0.0004 and abs(yerr) < 0.25:
                break
            x, y, yaw_cmd = x + err[0], y + err[1], yaw_cmd + yerr
        need(sk.move_to(x, y, seat_z + L.STUD_H + 0.003 + dz, yaw_deg=yaw_cmd, speed="slow"), "descend")
        o = self._objs(sk)[name]
        err = foot[:2] - np.array(o["position"][:2])
        if np.linalg.norm(err) > 0.0006:
            x, y = x + err[0], y + err[1]
        need(sk.move_to(x, y, seat_z + dz + 0.002, yaw_deg=yaw_cmd, speed="slow"), "lower")
        for push in (0.004, 0.007):
            sk.move_to(x, y, seat_z + dz - push, yaw_deg=yaw_cmd, speed="slow")  # press (stops on contact)
            if self._objs(sk)[name]["clicked"]:
                break
        need(sk.open_gripper(open_cm), "release")
        need(sk.move_to(x, y, clear, yaw_deg=yaw_cmd, speed="normal"), "retreat")
        if not self._objs(sk)[name]["clicked"]:
            raise OracleFailure(f"{name} did not click in at {span_name(b.pose)} level {b.pose.level}")


@register
class LegoManual(LegoBuildTask):
    name = "arm/lego_manual"
    summary = ("Build a 5-6 brick model on a studded baseplate following an illustrated step-by-step manual (PNG "
               "pages); bricks click together through real stud geometry and must be pressed in place.")
    time_limit = 450.0
    max_actions = 210
    # 9-12 bricks on 3-5 levels until 2026-10: the reference solution needs ~14 s per brick, and the model was cut so
    # that it needs about half of the 3-minute real-time budget
    n_bricks = (5, 6)
    levels = (2, 3)

    def design_text(self) -> str:
        n_steps = len(L.build_steps(self.design))
        last = n_steps + 1
        return (f"Build the brick model shown in the building instructions: manual/page_00_parts.png (parts list), "
                f"manual/page_01_step1.png ... page_{n_steps:02d}_step{n_steps}.png (one page per step: the parts "
                f"needed, a 3D view from the A1 corner of the plate with the newly added bricks outlined in magenta, "
                f"and a plan view from above) and manual/page_{last:02d}_finished.png (the finished model from two "
                f"sides). The model is {len(self.design)} bricks on "
                f"{max(b.pose.level for b in self.design) + 1} levels.")

    def documents(self, env) -> dict[str, bytes]:
        if not hasattr(self, "_pages"):
            mr = L.ManualRenderer(self.design, N_PLATE)
            try:
                self._pages = mr.pages()
            finally:
                mr.close()
        return self._pages
