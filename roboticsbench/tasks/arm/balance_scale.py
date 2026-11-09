"""Balance a beam scale: a sealed load of unknown mass sits on the left pan; put labelled calibrated weights on the right
pan until the beam settles level (within 2 deg for 2 s)."""

from __future__ import annotations

import math
from itertools import combinations

import mujoco
import numpy as np

from roboticsbench.core import visuals as V
from roboticsbench.tasks.arm._areason_props import (BOX, CAP, CYL, SPH, custom_material, free_speed, image_texture,
                                                    joint_adr, label_geom, label_material, quat_axis, quat_mul,
                                                    quat_yaw, ring_mesh, robot_touching, set_free, task_status)
from roboticsbench.tasks.arm.common import ArmTask, build_table
from roboticsbench.tasks.arm.oracle_utils import OracleFailure, need
from roboticsbench.tasks.base import Evaluation, Milestones, obj, register

ARM = 0.16  # beam half length: pan centres are 16 cm from the pivot
PIVOT_Z = 0.12
PAN_Z = 0.15  # pan top surface (beam level)
PAN_R = 0.07
PAN_T = 0.004
# The pan collides as a hidden 3 cm slab whose top is flush with the 4 mm brass plate, with stiff (brass, not
# rubber) contacts: a weight pressed onto it by the gripper must never sink past a thin plate's midline and drop
# through when released (2026-10-09, Sol full3). condim 4 as the weights have (priority makes the pan's count).
PAN_COL_H = 0.015
PAN_CONTACT = dict(priority=1, condim=4, solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
TILT_MAX = math.radians(10.0)
BOB_M, BOB_D = 0.33, 0.07  # pendulum bob under the pivot: 0.4 deg of tilt per gram of imbalance
TOL_DEG = 2.0
WEIGHTS = (("w200", 200), ("w100", 100), ("w50", 50), ("w20a", 20), ("w20b", 20), ("w10", 10))  # grams
W_HALF = (0.015, 0.015, 0.013)  # 3 x 3 x 2.6 cm blocks
LOAD_HALF = 0.02  # 4 cm mystery box
PAN_MASS = 0.12


def best_combo(target_g: float) -> list[str]:
    """Subset of the weights whose sum is closest to the target (fewest weights on ties)."""
    best, key = [], (abs(target_g), 0)
    names = [n for n, _ in WEIGHTS]
    grams = dict(WEIGHTS)
    for k in range(1, len(names) + 1):
        for sub in combinations(names, k):
            kk = (abs(target_g - sum(grams[n] for n in sub)), k)
            if kk < key:
                best, key = list(sub), kk
    return best


@register
class BalanceScale(ArmTask):
    name = "arm/balance_scale"
    category = "physical_reasoning"
    difficulty = "hard"
    summary = ("A beam balance carries a sealed box of unknown mass on its left pan; find its mass (wrist force "
               "sensor or trial and error) and put the right labelled calibrated weights (10-200 g) on the right pan "
               "so that the beam settles level within 2 deg for 2 s - the tolerance is finer than the smallest weight.")
    time_limit = 180.0
    max_actions = 100
    success_hold = 2.0
    controls = ("skills", "motor")
    oracle_control = "skills"
    cameras = ("balance_cam", "top", "wrist")
    video_camera = "balance_cam"

    def build(self, scene, rng):
        build_table(scene)
        self.c = np.array([rng.uniform(0.56, 0.60), rng.uniform(-0.03, 0.03)])
        # 120-370 g, a multiple of 10 g (+-2 g) that needs at least three of the weights
        ks = [k for k in range(12, 38) if len(best_combo(10.0 * k)) >= 3]
        k = int(rng.choice(ks))
        self.load_g = 10.0 * k + float(rng.uniform(-2.0, 2.0))  # hidden
        self.rack_x = self.c[0] - 0.19
        order = rng.permutation(len(WEIGHTS))
        self.rack_slot = {WEIGHTS[i][0]: int(p) for i, p in zip(order, range(len(WEIGHTS)))}
        self._build_balance(scene)
        self._build_weights(scene)
        cam = [self.c[0] - 0.40, self.c[1] - 0.36, 0.40]
        scene.add_camera("balance_cam", pos=cam, target=[self.c[0] - 0.07, self.c[1], 0.11], fovy=52)
        self.instruction = (
            "A beam balance stands on the table. A sealed wooden box of UNKNOWN mass (the 'mystery load', a 4 cm cube) "
            "sits on its LEFT pan (the +y side, on the robot's left). A rack in front of the balance holds six "
            "calibrated weights, labelled on top: 200 g, 100 g, 50 g, 20 g, 20 g and 10 g (3 x 3 x 2.6 cm blocks; "
            "list_objects gives their masses). Put calibrated weights on the RIGHT pan so that the beam balances: "
            f"success = beam within {TOL_DEG:.0f} deg of level and still for {self.success_hold:.0f} s, with the mystery "
            "load resting on the left pan, only calibrated weights (at least one) on the right pan, no weight on the "
            "left pan or touching the balance anywhere else, and nothing touched by the robot. The balance is "
            "sensitive: 1 g of imbalance tilts the beam by about 0.4 deg (up to its stops at +-10 deg), so the "
            "right combination of weights is needed. The pans (14 cm diameter, top 15 cm above the table when level) "
            "are guided so that they stay horizontal; they move up and down by up to 2.8 cm as the beam tilts. "
            "The task status shows the beam tilt (+ = left pan down). You may lift the mystery load to weigh it with "
            "the wrist force/torque sensor (a held, motionless mass m reads force_z ~= -9.81 m N) but it must be back "
            "on the left pan at the end. A weight or the load falling off the table fails the task.")
        self.milestones = Milestones([("weight_on_right_pan", 0.1), ("beam_off_stop", 0.3), ("within_5_deg", 0.5),
                                      ("within_2_deg", 0.75), ("balanced", 1.0)])

    # ------------------------------------------------------------------ construction
    def pan_xy(self, side: int) -> np.ndarray:  # side +1 = left (+y), -1 = right
        return np.array([self.c[0], self.c[1] + side * ARM])

    def rack_xy(self, k: int) -> np.ndarray:
        return np.array([self.rack_x, self.c[1] + (2.5 - k) * 0.062])

    def _build_balance(self, scene):
        spec = scene.spec
        brass = V.material(scene, "brass")
        wood = V.material(scene, "walnut", repeat=(5, 5))
        cx, cy = self.c
        base = scene.body("balance", [cx, cy, 0.0])
        base.add_geom(name="balance_base", type=BOX, size=[0.06, 0.12, 0.012], pos=[0, 0, 0.012], material=wood)
        V.visual_geom(scene, base, gtype=BOX, size=(0.045, 0.105, 0.0015), pos=(0, 0, 0.0255),
                      material=V.material(scene, "felt_green"))
        base.add_geom(name="balance_pillar", type=CYL, size=[0.011, (PIVOT_Z - 0.028) / 2, 0],
                      pos=[0.012, 0, 0.028 + (PIVOT_Z - 0.028) / 2], material=brass)
        V.visual_geom(scene, base, gtype=CYL, size=(0.02, 0.006), pos=(0.012, 0, 0.034), material=brass)
        # knife-edge bearing block
        V.visual_geom(scene, base, gtype=BOX, size=(0.012, 0.008, 0.008), pos=(0.012, 0, PIVOT_Z - 0.008),
                      material=V.material(scene, "steel_dark"))
        # the beam (hinge about x) with the pointer needle and the pendulum bob
        beam = base.add_body(name="beam", pos=[0.0, 0.0, PIVOT_Z])
        j = beam.add_joint(name="beam_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                           range=[-TILT_MAX, TILT_MAX], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        j.damping = [0.15, 0, 0]
        j.armature = 0.001
        j.solref_limit = [0.01, 1.0]
        beam.add_geom(name="beam_bar", type=BOX, size=[0.005, ARM + 0.012, 0.005], material=brass, mass=0.08,
                      contype=0, conaffinity=0)
        for sy in (-1, 1):
            V.visual_geom(scene, beam, gtype=SPH, size=(0.009,), pos=(0, sy * (ARM + 0.012), 0), material=brass)
        V.visual_geom(scene, beam, gtype=CYL, size=(0.008, 0.02), pos=(0.0, 0, 0),
                      quat=quat_axis([0, 1, 0], math.pi / 2), material=brass)
        beam.add_geom(name="beam_needle", type=BOX, size=[0.001, 0.0015, 0.06], pos=[-0.024, 0, 0.06],
                      material=custom_material(scene, "bal_needle", (0.1, 0.1, 0.1, 1)), mass=0.002, contype=0,
                      conaffinity=0)
        V.visual_geom(scene, beam, gtype=BOX, size=(0.003, 0.002, 0.004), pos=(-0.024, 0, 0.004), material=brass)
        V.visual_geom(scene, beam, gtype=CAP, size=(0.0025,), fromto=(-0.024, 0, 0, -0.024, 0, -BOB_D), material=brass)
        beam.add_geom(name="beam_bob", type=SPH, size=[0.02, 0, 0], pos=[-0.024, 0, -BOB_D], material=brass,
                      mass=BOB_M, contype=0, conaffinity=0)
        # scale plate behind the needle (faces the robot), ticks every 2 deg along the needle tip's arc
        img = self._scale_image()
        image_texture(scene, "bal_scale", img)
        smat = custom_material(scene, "bal_scale", (1, 1, 1, 1), specular=0.2, shininess=0.3, texture="bal_scale",
                               texuniform=False)
        q = quat_mul(quat_yaw(-math.pi / 2), quat_axis([1, 0, 0], math.pi / 2)).tolist()
        sp = base.add_body(name="balance_scale_plate", pos=[-0.018, 0.0, PIVOT_Z + 0.105], quat=q)
        label_geom(scene, sp, smat, pos=(0, 0, 0), half_w=0.05, half_h=0.025, thick=0.0015, up_axis="y")
        V.visual_geom(scene, base, gtype=BOX, size=(0.003, 0.004, 0.04), pos=(-0.014, 0.047, PIVOT_Z + 0.07),
                      material=brass)
        V.visual_geom(scene, base, gtype=BOX, size=(0.003, 0.004, 0.04), pos=(-0.014, -0.047, PIVOT_Z + 0.07),
                      material=brass)
        # pans on vertical guides, coupled to the beam angle (Roberval: they stay level, z = +-ARM sin(tilt))
        pan_mesh = ring_mesh(scene, "bal_pan_rim", PAN_R - 0.003, PAN_R + 0.002, 0.008)
        for nm, sy in (("left", 1), ("right", -1)):
            pan = base.add_body(name=f"pan_{nm}", pos=[0.0, sy * ARM, PAN_Z - PAN_T / 2])
            pj = pan.add_joint(name=f"pan_{nm}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                               range=[-0.035, 0.035], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
            pj.damping = [0.5, 0, 0]
            pj.armature = 0.001
            col = pan.add_geom(name=f"pan_{nm}", type=CYL, size=[PAN_R, PAN_COL_H, 0], pos=[0, 0, PAN_T / 2 - PAN_COL_H],
                               material=brass, mass=PAN_MASS, friction=[0.9, 0.01, 0.001], **PAN_CONTACT)
            V.hide_collision(col)
            V.visual_geom(scene, pan, gtype=CYL, size=(PAN_R, PAN_T / 2), material=brass)
            V.visual_mesh(scene, pan, pan_mesh, material=brass, pos=(0, 0, -PAN_T / 2))
            V.visual_geom(scene, pan, gtype=CYL, size=(0.006, 0.014), pos=(0, 0, -PAN_T / 2 - 0.014), material=brass)
            V.visual_geom(scene, pan, gtype=CYL, size=(0.012, 0.002), pos=(0, 0, -PAN_T / 2 - 0.029), material=brass)
            e = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"pan_{nm}_slide", name2="beam_hinge",
                                  name=f"pan_{nm}_link")
            e.data = [0.0, sy * ARM, 0.0, -sy * ARM / 6.0, 0.0] + [0.0] * 6
            e.solref = [0.005, 1.0]
            spec.add_exclude(bodyname1="balance", bodyname2=f"pan_{nm}")
            spec.add_exclude(bodyname1="beam", bodyname2=f"pan_{nm}")
        spec.add_exclude(bodyname1="balance", bodyname2="beam")

    def _scale_image(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        from roboticsbench.tasks.arm._areason_props import _font

        W_, H_ = 400, 200
        im = Image.new("RGB", (W_, H_), (242, 236, 214))
        dr = ImageDraw.Draw(im)
        dr.rectangle([2, 2, W_ - 3, H_ - 3], outline=(120, 90, 40), width=4)
        # plate: 10 x 5 cm centred 10.5 cm above the pivot; the needle tip is 12 cm from the pivot
        def to_px(y, z):  # plate-local metres (y to the left, z up, origin at the plate centre) -> pixels
            return (W_ / 2 - y / 0.1 * W_, H_ / 2 - z / 0.05 * H_)
        f = _font(22)
        for deg in range(-10, 11, 2):
            a = math.radians(deg)
            r0, r1 = (0.108, 0.122) if deg % 10 == 0 else (0.112, 0.12)
            p0 = to_px(r0 * math.sin(a), r0 * math.cos(a) - 0.105)
            p1 = to_px(r1 * math.sin(a), r1 * math.cos(a) - 0.105)
            col = (180, 20, 20) if deg == 0 else (40, 30, 20)
            dr.line([p0, p1], fill=col, width=5 if deg == 0 else 3)
        for deg, txt in ((-10, "10"), (0, "0"), (10, "10")):
            a = math.radians(deg)
            x, y = to_px(0.097 * math.sin(a), 0.097 * math.cos(a) - 0.105)
            bb = dr.textbbox((0, 0), txt, font=f)
            dr.text((x - (bb[2] - bb[0]) / 2 - bb[0], y - (bb[3] - bb[1]) / 2 - bb[1]), txt, font=f, fill=(40, 30, 20))
        # zone of the 2 deg tolerance
        for deg in (-2, 2):
            a = math.radians(deg)
            dr.line([to_px(0.124 * math.sin(a), 0.124 * math.cos(a) - 0.105),
                     to_px(0.13 * math.sin(a), 0.13 * math.cos(a) - 0.105)], fill=(30, 140, 60), width=4)
        fl = _font(26)
        dr.text((18, H_ - 44), "L", font=fl, fill=(40, 30, 20))
        dr.text((W_ - 36, H_ - 44), "R", font=fl, fill=(40, 30, 20))
        return np.asarray(im)

    def _build_weights(self, scene):
        wood = V.material(scene, "walnut", repeat=(5, 5))
        felt = custom_material(scene, "bal_felt", (0.12, 0.24, 0.16, 1), specular=0.02, shininess=0.05)
        rack = scene.body("weight_rack", [self.rack_x, self.c[1], 0.0])
        rack.add_geom(name="weight_rack", type=BOX, size=[0.03, 0.19, 0.006], pos=[0, 0, 0.006], material=wood)
        V.visual_geom(scene, rack, gtype=BOX, size=(0.028, 0.188, 0.0005), pos=(0, 0, 0.0125), material=felt)
        finish = {200: "brass", 100: "brass", 50: "steel", 20: "steel", 10: "aluminum"}
        mesh = V.rounded_box_mesh(scene, "bal_weight_mesh", W_HALF, 0.002)
        for nm, g in WEIGHTS:
            k = self.rack_slot[nm]
            xy = self.rack_xy(k)
            mat = V.material(scene, finish[g])
            b = scene.body(nm, [xy[0], xy[1], 0.012 + W_HALF[2]], free=True)
            geom = b.add_geom(name=nm, type=BOX, size=list(W_HALF), material=mat, mass=g / 1000.0,
                              friction=[0.9, 0.01, 0.001], condim=4)
            V.hide_collision(geom)
            V.visual_mesh(scene, b, mesh, material=mat)
            lm = label_material(scene, f"bal_lab_{g}", [f"{g}", "g"], size=(256, 256), font_px=100, bg=(30, 30, 34),
                                fg=(245, 240, 225))
            label_geom(scene, b, lm, pos=(0, 0, W_HALF[2] + 0.0004), half_w=0.0135, half_h=0.0135, thick=0.0004,
                       up_axis="x")
        # the mystery load: a dark lacquered box with a "?" on top
        lacq = custom_material(scene, "bal_mystery", (0.18, 0.08, 0.06, 1), specular=0.7, shininess=0.8,
                               reflectance=0.05)
        lb = scene.body("mystery_load", [*self.pan_xy(1), PAN_Z + LOAD_HALF], free=True)
        geom = lb.add_geom(name="mystery_load", type=BOX, size=[LOAD_HALF] * 3, material=lacq,
                           mass=self.load_g / 1000.0, friction=[0.9, 0.01, 0.001], condim=4)
        V.hide_collision(geom)
        V.visual_mesh(scene, lb, V.rounded_box_mesh(scene, "bal_load_mesh", [LOAD_HALF] * 3, 0.003), material=lacq)
        lm = label_material(scene, "bal_lab_q", "?", size=(128, 128), font_px=110, bg=(200, 160, 70), fg=(40, 20, 10))
        label_geom(scene, lb, lm, pos=(0, 0, LOAD_HALF + 0.0004), half_w=0.012, half_h=0.012, thick=0.0004,
                   up_axis="x")

    # ------------------------------------------------------------------ runtime
    def bind(self, env):
        m = env.model
        self.q_beam, self.v_beam = joint_adr(m, "beam_hinge")
        self.items = [n for n, _ in WEIGHTS] + ["mystery_load"]
        self.gid = {n: m.geom(n).id for n in self.items}
        self.g2n = {g: n for n, g in self.gid.items()}
        self.pan_geom = {+1: m.geom("pan_left").id, -1: m.geom("pan_right").id}
        bal_bodies = {m.body(b).id for b in ("balance", "beam", "pan_left", "pan_right")}
        self.balance_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bal_bodies}
        self.other_bal = self.balance_geoms - set(self.pan_geom.values())

    def reset(self, env, rng):
        for nm, _ in WEIGHTS:
            xy = self.rack_xy(self.rack_slot[nm]) + rng.uniform(-0.002, 0.002, size=2)
            set_free(env, nm, [xy[0], xy[1], 0.013 + W_HALF[2]], quat_yaw(math.radians(rng.uniform(-5, 5))))
        p = self.pan_xy(1) + rng.uniform(-0.004, 0.004, size=2)
        # start at the left stop (the load is heavier than the empty right pan)
        env.data.qpos[self.q_beam] = -TILT_MAX
        m = env.model
        for nm, sy in (("left", 1), ("right", -1)):
            env.data.qpos[joint_adr(m, f"pan_{nm}_slide")[0]] = sy * ARM * math.sin(-TILT_MAX)
        set_free(env, "mystery_load", [p[0], p[1], PAN_Z - ARM * math.sin(TILT_MAX) + LOAD_HALF + 0.001],
                 quat_yaw(math.radians(rng.uniform(-8, 8))))
        self.best_abs = 99.0

    # --- state
    def tilt_deg(self, env) -> float:
        """Public tilt: + = left pan down."""
        return -math.degrees(float(env.data.qpos[self.q_beam]))

    def _support(self, env) -> dict[str, set[int]]:
        """geoms each item touches (other than the robot)."""
        d = env.data
        out: dict[str, set[int]] = {n: set() for n in self.items}
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a in self.g2n:
                out[self.g2n[a]].add(b)
            if b in self.g2n:
                out[self.g2n[b]].add(a)
        return out

    def _placement(self, env):
        """(weights on the right pan, problems) - an item is on a pan when it rests inside its rim (on the pan or on
        items that are)."""
        sup = self._support(env)
        pos = {n: env.data.body(n).xpos for n in self.items}
        on = {+1: set(), -1: set()}
        for side in (+1, -1):
            c = self.pan_xy(side)
            for n in self.items:
                if math.hypot(pos[n][0] - c[0], pos[n][1] - c[1]) <= PAN_R and pos[n][2] > PAN_Z - 0.04:
                    on[side].add(n)
        problems = []
        if "mystery_load" not in on[+1]:
            problems.append("mystery load not on the left pan")
        if any(n != "mystery_load" for n in on[+1]):
            problems.append("a calibrated weight is on the left pan")
        if "mystery_load" in on[-1]:
            problems.append("mystery load on the right pan")
        for n in self.items:
            if sup[n] & self.other_bal:
                problems.append(f"{n} touches the balance frame")
            if n not in on[+1] and n not in on[-1] and sup[n] & set(self.pan_geom.values()):
                problems.append(f"{n} leans on a pan")
        return on[-1], problems

    def step(self, env):
        right, problems = self._placement(env)
        if right:
            self.milestones.hit("weight_on_right_pan", env.t)
        if right and not problems:
            t = abs(self.tilt_deg(env))
            if t < math.degrees(TILT_MAX) - 0.5:
                self.milestones.hit("beam_off_stop", env.t)
            if t < 5.0:
                self.milestones.hit("within_5_deg", env.t)
            if t <= TOL_DEG and not robot_touching(env, set(self.gid.values()) | self.balance_geoms):
                self.milestones.hit("within_2_deg", env.t)

    def evaluate(self, env):
        right, problems = self._placement(env)
        tilt = self.tilt_deg(env)
        omega = abs(float(env.data.qvel[self.v_beam]))
        metrics = {"tilt_deg": round(tilt, 2), "right_pan": sorted(right), "problems": problems,
                   "load_g": round(self.load_g, 1), "right_g": sum(dict(WEIGHTS).get(n, 0) for n in right)}
        prog = min(self.milestones.progress, 0.95)
        for n in self.items:
            if env.data.body(n).xpos[2] < -0.05:
                return Evaluation(failed=True, progress=prog, reason=f"{n} fell off the table", metrics=metrics)
        ok = (bool(right) and not problems and abs(tilt) <= TOL_DEG and omega < 0.05
              and not robot_touching(env, set(self.gid.values()) | self.balance_geoms)
              and all(free_speed(env, n)[0] < 0.02 for n in self.items))
        if ok:
            self.milestones.hit("balanced", env.t)
        return Evaluation(success=ok, progress=prog, metrics=metrics)

    # ------------------------------------------------------------------ agent-facing
    def objects(self, env):
        out = [obj("balance", "beam balance (static frame; pans move)", "brass", [self.c[0], self.c[1], PIVOT_Z],
                   [0.12, 2 * (ARM + PAN_R), 0.24], note="position = beam pivot; pans 16 cm to either side")]
        for nm, side in (("left", 1), ("right", -1)):
            p = env.data.body(f"pan_{nm}").xpos
            out.append(obj(f"pan_{nm}", "round balance pan (stays horizontal)", "brass",
                           [p[0], p[1], p[2] + PAN_T / 2], [2 * PAN_R, 2 * PAN_R, PAN_T],
                           note="position = centre of the pan's top surface (moves up/down with the beam)"))
        for nm, g in WEIGHTS:
            R = env.data.body(nm).xmat.reshape(3, 3)
            yaw = (math.degrees(math.atan2(R[1, 0], R[0, 0])) + 45.0) % 90.0 - 45.0
            out.append(obj(nm, f"calibrated weight block labelled '{g} g'", "brass" if g >= 100 else "steel",
                           env.data.body(nm).xpos, [2 * W_HALF[0], 2 * W_HALF[1], 2 * W_HALF[2]], mass_g=g,
                           yaw_deg=round(yaw, 1)))
        R = env.data.body("mystery_load").xmat.reshape(3, 3)
        yaw = (math.degrees(math.atan2(R[1, 0], R[0, 0])) + 45.0) % 90.0 - 45.0
        out.append(obj("mystery_load", "sealed wooden box (unknown mass)", "dark red",
                       env.data.body("mystery_load").xpos, [2 * LOAD_HALF] * 3, yaw_deg=round(yaw, 1)))
        return out

    def status(self, env):
        t = self.tilt_deg(env)
        return {"beam_tilt_deg": round(t, 2), "at_stop": abs(t) > math.degrees(TILT_MAX) - 0.3}

    # ------------------------------------------------------------------ oracle
    def oracle(self, sk):
        def objs():
            return {o["name"]: o for o in sk.list_objects().data["objects"]}

        def tcp():
            return np.array(sk.robot.proprio()["gripper_position"], float)

        def grab(n, lift_z):
            o = objs()[n]
            p = np.array(o["position"], float)
            yaw = 90.0 + float(o["yaw_deg"])
            need(sk.open_gripper(4.2 if n != "mystery_load" else 5.4), "open")
            need(sk.move_to(p[0], p[1], max(p[2] + 0.08, 0.24), yaw_deg=yaw, speed="fast"), f"above {n}")
            need(sk.move_to(p[0], p[1], p[2] + 0.004, yaw_deg=yaw, speed="normal"), f"down to {n}")
            r = need(sk.close_gripper(30.0), f"grip {n}")
            if "holding something" not in r.message:
                raise OracleFailure(f"missed {n}: {r.message}")
            need(sk.move_to(p[0], p[1], lift_z, yaw_deg=yaw, speed="slow"), f"lift {n}")
            return yaw

        def put(n, xy, yaw, half_h, open_cm):
            need(sk.move_to(xy[0], xy[1], 0.26, yaw_deg=yaw, speed="normal"), f"carry {n}")
            off = np.array(objs()[n]["position"], float) - tcp()
            surf = float(objs()["pan_right" if n != "mystery_load" else "pan_left"]["position"][2])
            tz = surf + half_h + 0.004 - off[2]
            need(sk.move_to(xy[0] - off[0], xy[1] - off[1], tz + 0.03, yaw_deg=yaw, speed="normal"), f"over pan")
            need(sk.move_to(xy[0] - off[0], xy[1] - off[1], tz, yaw_deg=yaw, speed="slow"), f"lower {n}")
            need(sk.open_gripper(open_cm), f"release {n}")
            p = tcp()
            need(sk.move_to(p[0], p[1], 0.26, yaw_deg=yaw, speed="normal"), "retreat")

        # 1) weigh the mystery load: lift it off the left pan, hold still, read the wrist force, put it back
        load_xy = np.array(objs()["mystery_load"]["position"][:2], float)
        yaw = grab("mystery_load", 0.24)
        fz = need(sk.read_wrist_force(), "weigh").data["force"][2]
        grams = -fz / 9.81 * 1000.0
        put("mystery_load", objs()["pan_left"]["position"][:2], yaw, LOAD_HALF, 5.4)
        del load_xy
        # 2) the closest combination; heaviest first onto a 2 x 2 pattern on the right pan
        combo = sorted(best_combo(grams), key=lambda n: -dict(WEIGHTS)[n])
        pc = np.array(objs()["pan_right"]["position"][:2], float)
        spots = [pc + np.array(v) for v in ((-0.036, 0.018), (-0.036, -0.018), (0.036, 0.018), (0.036, -0.018))]
        for n, xy in zip(combo, spots):
            yaw = grab(n, 0.24)
            put(n, xy, yaw, W_HALF[2], 4.2)
        # 3) step back and let it settle
        p = tcp()
        need(sk.move_to(p[0] - 0.12, p[1], 0.30, yaw_deg=yaw, speed="normal"), "step back")
        sk.wait(3.0)
        if abs(task_status(sk)["beam_tilt_deg"]) > TOL_DEG:
            raise OracleFailure(f"not balanced: {task_status(sk)} (weighed {grams:.1f} g, combo {combo})")
        sk.wait(2.0)
