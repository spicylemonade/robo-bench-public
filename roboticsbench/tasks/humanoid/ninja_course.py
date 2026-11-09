"""humanoid/ninja_course -- a ninja-warrior course: stairs up, a swinging platform over a pit, a ramp down, a crawl
tunnel, the finish.

Sections, in order along +x (the robot starts at (0, 0) facing +x; side walls at y = +-1.6):
1. STAIRS: 5 steps (rise 0.14 m, tread 0.32 m, 1.2 m wide, handrails) up to deck A at 0.70 m.
2. SWING: a motor-driven swinging platform (0.9 x 0.9 m, kept level by a parallelogram linkage, handrails) crosses a
   2.16 m water pit between deck A and deck B. It swings like a free pendulum (rods 2.4 m, +-15 deg, period 3.11 s) on
   a fixed schedule; at each end of its swing it is level with a deck and its edge within 1 cm of the deck edge (within
   15 cm for ~0.36 s either side). Riding it tilts the apparent gravity by up to 15 deg along x.
3. RAMP: deck B ends in a 3.0 m ramp (13 deg) down to the ground.
4. CRAWL SPACE: a tunnel 2.4 m long, 1.0 m wide, roof 0.80 m - lower than any squat (the deepest crouch is ~1.06 m
   tall), so the robot must crawl (hands and knees ~0.66 m tall, forearms and knees ~0.54 m). On the crawl mat around
   it a low posture (kneeling, crawling, lying down) does not count as a fall.
5. FINISH: the green circle (radius 0.5 m) behind the finish arch.
Failure: touching the water (falls do not fail - the episode runs on). Sections only count in order.

Feasibility (docs/progress/ninja.md has the numbers): every element is within the G1's motor limits - a 0.14 m step
needs 51-84 N m at the knee (limit 139); riding the +-15 deg swing needs a lean that follows the apparent gravity
(within the ankles' -50..+30 deg range, ~11 N m per ankle to track it, limit 50); the tunnel needs the deep crouch
(pelvis ~0.6 m, head top ~1.15 m) and the crawl on hands and knees: with straight arms on the fists the ~75 N per
hand runs along the arm, through the 5 N m wrists without bending them; getting down and up again stays within the knee
and hip limits. The hanging obstacles of a real ninja course (monkey bars, rope swings) were left
out on purpose: the Dex3 fingers (1.4 N m) cannot hold the robot's 337 N weight (a one-hand hang needs ~3.9 N m per
finger). The reference solution's walking policy cannot climb steps or ride the swing (measured: it falls on 5 cm
steps and on swings above ~4 deg) and cannot crawl, so it covers the ramp and the finish and skips the stairs, the
swing and the crawl space (reported in the metrics) - those are feasible on the numbers above but unproven by a
controller.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roboticsbench.core import visuals as V
from roboticsbench.tasks.base import Evaluation, Milestones, obj, register
from roboticsbench.tasks.humanoid import _hsport_props as P
from roboticsbench.tasks.humanoid._hnav_props import tracking_camera
from roboticsbench.tasks.humanoid._hpuz_props import vent_duct, vwalk
from roboticsbench.tasks.humanoid.common import HumanoidTask

WALL_Y = 1.6
WIDTH = 1.2  # stairs, decks and ramp
# stairs
N_STEPS, RISE, TREAD, X_STAIRS = 5, 0.14, 0.32, 1.2
DECK_Z = N_STEPS * RISE  # 0.70
X_DECK_A0 = X_STAIRS + N_STEPS * TREAD  # 2.8
X_DECK_A1 = 4.2
# swing
ROD = 2.4
THETA0 = math.radians(15.0)
PERIOD = 2 * math.pi * math.sqrt(ROD / 9.81)  # 3.11 s: it moves like a free pendulum
PLAT_HALF = 0.45
X_PIVOT = X_DECK_A1 + 0.01 + PLAT_HALF + ROD * math.sin(THETA0)
Z_PIVOT = DECK_Z + ROD * math.cos(THETA0)
X_DECK_B0 = X_PIVOT + ROD * math.sin(THETA0) + PLAT_HALF + 0.01
X_DECK_B1 = X_DECK_B0 + 1.15
# ramp, tunnel, finish
RAMP_RUN = 3.0
X_RAMP_FOOT = X_DECK_B1 + RAMP_RUN
X_TUNNEL0, TUNNEL_LEN, TUNNEL_W, TUNNEL_ROOF = X_RAMP_FOOT + 2.5, 2.4, 1.0, 0.80
MAT_X0, MAT_X1, MAT_HY = X_TUNNEL0 - 1.3, X_TUNNEL0 + TUNNEL_LEN + 1.3, 0.8  # crawl mat: a low posture is not a fall here
X_FINISH = MAT_X1 + 1.0
FINISH_R = 0.5
DRIVE_KP, DRIVE_KV = 40000.0, 6000.0
SECTIONS = ("stairs", "swing", "ramp", "crawl", "finish")


@register
class NinjaCourse(HumanoidTask):
    name = "humanoid/ninja_course"
    category = "course"
    difficulty = "expert"
    summary = ("Ninja-warrior course: climb 5 stairs, time a step onto a motor-driven swinging platform, ride it over "
               "a water pit (leaning with the swing) and step off at the far deck, walk down a ramp, crawl on hands "
               "and knees through a 0.8 m high crawl space, get up and reach the finish.")
    time_limit = 300.0
    max_actions = 60
    success_hold = 0.5
    world = {"floor": "grid", "style": "arena"}
    cameras = ("head", "chase", "side")
    video_camera = "side"
    controls = ("motor", "velocity")
    oracle_control = "velocity"

    # ------------------------------------------------------------------ build
    def build(self, scene, rng):
        self.phase = float(rng.uniform(0.0, PERIOD))  # where in its swing the platform is at t = 0
        wood = V.material(scene, "oak")
        steel = V.material(scene, "steel")
        hy = WIDTH / 2
        # side walls along the whole course
        for s in (-1, 1):
            scene.static_box(f"wall_{s}", [(X_FINISH + 1.5 - 0.8) / 2, s * (WALL_Y + 0.05), 0.4],
                             [(X_FINISH + 1.5 + 0.8) / 2, 0.05, 0.4], material=V.material(scene, "paint_blue"))
        scene.static_box("wall_end", [X_FINISH + 1.55, 0.0, 0.4], [0.05, WALL_Y + 0.1, 0.4],
                         material=V.material(scene, "paint_blue"))
        # 1. stairs and deck A (solid blocks), handrails
        self.step_geoms = []
        for k in range(N_STEPS):
            x0 = X_STAIRS + k * TREAD
            top = RISE * (k + 1)
            g = f"step_{k}"
            scene.static_box(g, [x0 + TREAD / 2, 0.0, top / 2], [TREAD / 2, hy, top / 2], material=wood)
            P.deco(scene, P.BOX, [x0 + 0.03, 0.0, top + 0.001], [0.025, hy - 0.02, 0.0012],
                   V.material(scene, "matte_yellow"))  # nosing stripe
            self.step_geoms.append(g)
        scene.static_box("deck_a", [(X_DECK_A0 + X_DECK_A1) / 2, 0.0, DECK_Z / 2],
                         [(X_DECK_A1 - X_DECK_A0) / 2, hy, DECK_Z / 2], material=wood)
        for s in (-1, 1):  # handrails: up the stairs and along deck A
            y = s * (hy + 0.04)
            pts = [(X_STAIRS, 0.0), (X_DECK_A0, DECK_Z), (X_DECK_A1 - 0.05, DECK_Z)]
            for (xa, za), (xb, zb) in zip(pts[:-1], pts[1:]):
                scene._add_geom(scene.spec.worldbody, gtype=P.CAP, size=[0.025],
                                fromto=[xa, y, za + 0.9, xb, y, zb + 0.9], material=steel, name=f"rail_{s}_{xa:.1f}")
            for xa, za in pts:
                P.deco(scene, P.CYL, None, [0.02], steel, fromto=[xa, y, za, xa, y, za + 0.9])
        # 2. the pit (water: touching it fails), the swing frame and the platform
        self.water = P.water_box(scene, "pit_water", X_DECK_A1, X_DECK_B0, -WALL_Y, WALL_Y)
        for s in (-1, 1):  # A-frame posts carrying the pivot beam
            for dx in (-0.5, 0.5):
                P.deco(scene, P.CYL, None, [0.05], steel,
                       fromto=[X_PIVOT + dx, s * 1.35, 0.0, X_PIVOT, s * 1.35, Z_PIVOT + 0.05])
        P.deco(scene, P.CYL, None, [0.06], steel, fromto=[X_PIVOT, -1.4, Z_PIVOT, X_PIVOT, 1.4, Z_PIVOT])
        arm = scene.spec.worldbody.add_body(name="swing_arm", pos=[X_PIVOT, 0.0, Z_PIVOT])
        jp = arm.add_joint(name="swing_pivot", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        jp.armature = 5.0
        rod = V.material(scene, "black_metal")
        for y in (-0.42, 0.42):
            arm.add_geom(type=P.CAP, fromto=[0, y, 0, 0, y, -ROD], size=[0.02, 0, 0], mass=1.5, contype=0,
                         conaffinity=0, material=rod)
        plat = arm.add_body(name="swing_platform", pos=[0.0, 0.0, -ROD])
        jl = plat.add_joint(name="swing_level", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        jl.armature = 5.0
        plat.add_geom(name="platform_deck", type=P.BOX, pos=[0, 0, -0.03], size=[PLAT_HALF, PLAT_HALF, 0.03],
                      mass=15.0, material=wood, friction=[1.0, 0.01, 0.001])
        for y in (-0.42, 0.42):  # the platform's handrails (the rods hang from their tops)
            plat.add_geom(type=P.CAP, fromto=[-PLAT_HALF + 0.03, y, 0.9, PLAT_HALF - 0.03, y, 0.9], size=[0.02, 0, 0],
                          mass=0.5, material=steel)
            for x in (-PLAT_HALF + 0.03, PLAT_HALF - 0.03):
                plat.add_geom(type=P.CAP, fromto=[x, y, 0.0, x, y, 0.9], size=[0.018, 0, 0], mass=0.3, contype=0,
                              conaffinity=0, material=steel)
        eq = scene.spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="swing_level", name2="swing_pivot")
        eq.data[0:5] = [0.0, -1.0, 0.0, 0.0, 0.0]  # level = -pivot: the platform stays horizontal
        # the drive: PD on the scheduled angle AND velocity, evaluated every physics step (force = ctrl - kp q - kv qd
        # with ctrl = kp q_d + kv qd_d set each tick), far stronger than anything riding on it
        act = scene.spec.add_actuator(name="swing_motor", target="swing_pivot", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -DRIVE_KP, -DRIVE_KV]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        # 3. deck B and the ramp down
        scene.static_box("deck_b", [(X_DECK_B0 + X_DECK_B1) / 2, 0.0, DECK_Z / 2],
                         [(X_DECK_B1 - X_DECK_B0) / 2, hy, DECK_Z / 2], material=wood)
        P.ramp_slab(scene, "ramp", (X_RAMP_FOOT, 0.0), math.pi, RAMP_RUN, WIDTH, DECK_Z, material=wood)
        scene.static_box("ramp_fill", [X_DECK_B1 - 0.02, 0.0, DECK_Z / 2], [0.02, hy, DECK_Z / 2], material=wood)
        # 4. the crawl space on its mat, 5. the finish
        P.deco(scene, P.BOX, [(MAT_X0 + MAT_X1) / 2, 0.0, 0.006], [(MAT_X1 - MAT_X0) / 2, MAT_HY, 0.006],
               V.material(scene, "rubber_blue"), name="crawl_mat")
        self.tunnel_geoms = vent_duct(scene, "tunnel", X_TUNNEL0, TUNNEL_LEN, 0.0, inner_w=TUNNEL_W, roof=TUNNEL_ROOF)
        P.floor_disc(scene, "finish", (X_FINISH, 0.0), FINISH_R, "green")
        P.finish_arch(scene, "finish_arch", X_FINISH + 0.8, -1.2, 1.2)
        for k, (x, txt) in enumerate(((X_STAIRS - 0.5, "1 STAIRS"), (X_DECK_A1 - 0.6, "2 SWING"),
                                      (X_DECK_B1 + 0.4, "3 RAMP"), (MAT_X0 - 0.3, "4 CRAWL"))):
            P.sign(scene, f"sign_{k}", (x, WALL_Y - 0.25), 180.0 + 90.0, [txt], width=0.9, height=0.32, z=1.6,
                   font_px=64)
        tracking_camera(scene, "side", offset=(0.0, -5.2, 2.2), look_offset=(0.0, 0.0, 0.7), fovy=55)
        self.milestones = Milestones([("top_of_stairs", 0.2), ("boarded_swing", 0.35), ("across_the_pit", 0.5),
                                      ("down_the_ramp", 0.6), ("through_the_crawl_space", 0.8)])
        self._write_instruction()

    def _write_instruction(self):
        self.instruction = (
            "Ninja course: complete five sections IN ORDER and stop in the green finish circle. Coordinates: world "
            "frame in metres, you start at (0, 0) facing +x (the course direction), +y is to your left; side walls at "
            f"y = +-{WALL_Y}. (1) STAIRS: {N_STEPS} steps, each {RISE} m high and {TREAD} m deep, {WIDTH} m wide "
            f"(y = +-{WIDTH / 2}), the first riser at x = {X_STAIRS}; handrails 0.9 m above the treads on both sides "
            f"(you may touch them, but your fingers cannot carry your weight). They lead up to deck A (top at z = "
            f"{DECK_Z:.2f}, x = {X_DECK_A0:.2f}..{X_DECK_A1:.2f}). (2) SWING: a water pit (touching the water fails) "
            f"separates deck A from deck B (x = {X_DECK_B0:.2f}..{X_DECK_B1:.2f}, top {DECK_Z:.2f}). A motor-driven "
            f"swinging platform ({2 * PLAT_HALF} x {2 * PLAT_HALF} m, top surface kept level, handrails 0.9 m high "
            f"along its sides at y = +-0.42) hangs from a pivot at x = {X_PIVOT:.3f}, z = {Z_PIVOT:.3f} on {ROD} m "
            "rods and swings like a free pendulum on a fixed schedule: rod angle theta(t) = -15 deg x cos(2 pi (t + "
            f"{self.phase:.3f}) / {PERIOD:.3f}) (t = simulated time in s; negative = toward deck A), so the platform "
            f"centre is at x = {X_PIVOT:.3f} + {ROD} sin(theta), its top at z = {Z_PIVOT:.3f} - {ROD} cos(theta). At "
            "each end of its swing it is level with a deck, its edge within 1 cm of the deck edge (within 15 cm for "
            "~0.36 s either side). Step on at deck A when it arrives, ride it, and step off onto deck B at the far end "
            "of the swing. The ride tilts the apparent gravity by up to 15 deg along x (it points along the rods): "
            "lean with it, or the swing throws you off. Its drive is far stronger than you - you cannot change its "
            f"schedule. (3) RAMP: deck B ends in a {RAMP_RUN} m long ramp ({math.degrees(math.atan2(DECK_Z, RAMP_RUN)):.0f} "
            f"deg) down to the ground at x = {X_RAMP_FOOT:.2f}. (4) CRAWL SPACE: a tunnel from x = {X_TUNNEL0:.2f} "
            f"to {X_TUNNEL0 + TUNNEL_LEN:.2f}, inner width {TUNNEL_W} m (y = +-{TUNNEL_W / 2}), roof underside "
            f"{TUNNEL_ROOF} m above the floor - lower than any squat (your deepest crouch is ~1.06 m tall), so crawl: on "
            "hands and knees (arms straight, fists down, back level) you are ~0.66 m tall, on forearms and knees "
            "~0.54 m. You may touch the tunnel. A blue crawl mat covers x = "
            f"{MAT_X0:.2f}..{MAT_X1:.2f}, y = +-{MAT_HY}: kneel, crawl or lie down on it as you like, and get back up on "
            "the mat. (5) FINISH: the green circle "
            f"(radius {FINISH_R} m) at ({X_FINISH:.2f}, 0) behind the finish arch: success = your pelvis inside it for "
            "0.5 s after sections 1-4. The task status shows the next section. You see through your "
            "head camera only.")

    # ------------------------------------------------------------------ episode
    def theta(self, t: float) -> float:
        """The documented rod angle (positive toward deck B, +x); the pivot joint's angle is its negative (a positive
        rotation about +y swings the platform toward -x)."""
        return -THETA0 * math.cos(2 * math.pi * (t + self.phase) / PERIOD)

    def theta_dot(self, t: float) -> float:
        w = 2 * math.pi / PERIOD
        return THETA0 * w * math.sin(w * (t + self.phase))

    def _drive(self, env, t: float) -> None:
        tm = t + env.control_dt / 2  # the coming tick's midpoint
        env.data.ctrl[self.motor] = -(DRIVE_KP * self.theta(tm) + DRIVE_KV * self.theta_dot(tm))

    def bind(self, env):
        m = env.model
        self.jp = int(m.jnt_qposadr[m.joint("swing_pivot").id])
        self.jl = int(m.jnt_qposadr[m.joint("swing_level").id])
        self.vp = int(m.jnt_dofadr[m.joint("swing_pivot").id])
        self.vl = int(m.jnt_dofadr[m.joint("swing_level").id])
        self.motor = m.actuator("swing_motor").id
        self.platform = m.geom("platform_deck").id
        self.water_id = m.geom("pit_water").id
        self.tunnel_ids = {m.geom(n).id for n in self.tunnel_geoms}
        self.deck_a = {m.geom(n).id for n in ["deck_a"]}
        self.deck_b = {m.geom(n).id for n in ["deck_b"]}
        sides = {}
        for side in ("left", "right"):
            b = m.body(f"{side}_ankle_roll_link").id
            sides[side] = {g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]}
        self.feet = sides

    def reset(self, env, rng):
        d = env.data
        th, w = -self.theta(0.0), -self.theta_dot(0.0)  # joint angle / rate
        d.qpos[self.jp], d.qpos[self.jl] = th, -th
        d.qvel[self.vp], d.qvel[self.vl] = w, -w
        self._drive(env, 0.0)
        self.done: list[str] = []
        self.fail_reason = ""
        self.skipped: list[str] = []

    def allows_low_posture(self, env) -> bool:
        x, y = env.robot.base_pos[0], env.robot.base_pos[1]
        return MAT_X0 <= x <= MAT_X1 and abs(y) <= MAT_HY

    def _feet_on(self, env, geoms: set[int]) -> bool:
        d = env.data
        hit = {"left": False, "right": False}
        for i in range(d.ncon):
            c = d.contact[i]
            for side, fg in self.feet.items():
                if (c.geom1 in fg and c.geom2 in geoms) or (c.geom2 in fg and c.geom1 in geoms):
                    hit[side] = True
        return hit["left"] and hit["right"]

    def _touching(self, env, geoms: set[int]) -> bool:
        d, rg = env.data, env.robot.geom_ids
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in rg and c.geom2 in geoms) or (c.geom2 in rg and c.geom1 in geoms):
                return True
        return False

    def next_section(self) -> str:
        return SECTIONS[len(self.done)] if len(self.done) < len(SECTIONS) else "done"

    def step(self, env):
        self._drive(env, env.t)
        if self.fail_reason:
            return
        if self._touching(env, {self.water_id}):
            self.fail_reason = "fell into the water pit"
            return
        x = float(env.robot.base_pos[0])
        nxt = self.next_section()
        if nxt == "stairs" and self._feet_on(env, self.deck_a):
            self.done.append("stairs")
            self.milestones.hit("top_of_stairs", env.t)
        elif nxt == "swing":
            if self._feet_on(env, {self.platform}):
                self.milestones.hit("boarded_swing", env.t)
            if self.milestones.has("boarded_swing") and self._feet_on(env, self.deck_b):
                self.done.append("swing")
                self.milestones.hit("across_the_pit", env.t)
        elif nxt == "ramp" and x > X_RAMP_FOOT + 0.2 and env.robot.base_pos[2] < 1.0:
            self.done.append("ramp")
            self.milestones.hit("down_the_ramp", env.t)
        elif nxt == "crawl" and x > X_TUNNEL0 + TUNNEL_LEN + 0.3:
            self.done.append("crawl")
            self.milestones.hit("through_the_crawl_space", env.t)

    def evaluate(self, env):
        p = env.robot.base_pos
        in_finish = math.hypot(p[0] - X_FINISH, p[1]) < FINISH_R
        ok = len(self.done) == 4 and in_finish and not self.fail_reason
        return Evaluation(success=ok, progress=min(0.95, self.milestones.progress), failed=bool(self.fail_reason),
                          reason=self.fail_reason,
                          metrics={"sections_done": list(self.done), "next": self.next_section(),
                                   "reference_skipped": list(self.skipped)})

    # ------------------------------------------------------------------ agent-facing
    def status(self, env):
        return {"sections_done": list(self.done), "next": self.next_section(),
                "swing_angle_deg": round(math.degrees(self.theta(env.t)), 2),
                "on_crawl_mat": self.allows_low_posture(env)}

    def objects(self, env):
        d = env.data
        pid = env.model.body("swing_platform").id
        return [obj("deck_a", "deck", "wood", [(X_DECK_A0 + X_DECK_A1) / 2, 0.0, DECK_Z], [X_DECK_A1 - X_DECK_A0,
                                                                                        WIDTH, DECK_Z]),
                obj("swing_platform", "platform", "wood", d.xpos[pid], [0.9, 0.9, 0.06],
                    rod_angle_deg=round(-math.degrees(float(d.qpos[self.jp])), 2)),
                obj("deck_b", "deck", "wood", [(X_DECK_B0 + X_DECK_B1) / 2, 0.0, DECK_Z], [X_DECK_B1 - X_DECK_B0,
                                                                                        WIDTH, DECK_Z]),
                obj("crawl_space", "crawl tunnel", "aluminium", [X_TUNNEL0 + TUNNEL_LEN / 2, 0.0, TUNNEL_ROOF / 2],
                    [TUNNEL_LEN, TUNNEL_W, TUNNEL_ROOF]),
                obj("crawl_mat", "mat", "blue", [(MAT_X0 + MAT_X1) / 2, 0.0, 0.0], [MAT_X1 - MAT_X0, 2 * MAT_HY, 0.012]),
                obj("finish", "floor circle", "green", [X_FINISH, 0.0, 0.0], [2 * FINISH_R, 2 * FINISH_R, 0.0])]

    # ------------------------------------------------------------------ oracle
    def skip_to_deck_b(self, env) -> None:
        """Validation only: the walking policy cannot climb the stairs or ride the swing, so the reference solution
        starts on deck B and reports the skip (``reference_skipped`` in the metrics)."""
        r, d = env.robot, env.data
        d.qpos[r.base_qadr:r.base_qadr + 3] = [X_DECK_B0 + 0.45, 0.0, r.base_pos[2] + DECK_Z]
        d.qvel[r.base_vadr:r.base_vadr + 6] = 0.0
        mujoco.mj_forward(env.model, d)
        self.done = ["stairs", "swing"]
        self.skipped = ["stairs", "swing"]
        for ms in ("top_of_stairs", "boarded_swing", "across_the_pit"):
            self.milestones.hit(ms, env.t)

    def skip_crawl(self, env) -> None:
        """Validation only: the walking policy cannot crawl - move the robot (standing) past the crawl space."""
        r, d = env.robot, env.data
        d.qpos[r.base_qadr:r.base_qadr + 2] = [X_TUNNEL0 + TUNNEL_LEN + 0.5, 0.0]
        d.qvel[r.base_vadr:r.base_vadr + 6] = 0.0
        mujoco.mj_forward(env.model, d)
        self.skipped.append("crawl")

    def oracle(self, api):
        api.step(25)
        self.skip_to_deck_b(api.env)
        api.step(50)
        vwalk(api, MAT_X0 + 0.3, 0.0, speed=0.4, tol=0.06, heading=0.0)  # down the ramp onto the crawl mat
        api.stop(25)
        self.skip_crawl(api.env)
        api.step(50)
        vwalk(api, X_FINISH, 0.0, speed=0.5, tol=0.08)
        api.step(60, {"base_velocity": [0.0, 0.0, 0.0]})
