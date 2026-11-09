"""Skills for the Franka Panda (Cartesian motions via IK, force-limited gripper, wrist F/T)."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roboticsbench.control.ik import SiteIK
from roboticsbench.core.geometry import quat_to_mat, rotvec_error, slerp, smoothstep
from roboticsbench.robots.panda import ARM_JOINTS, GRIP_FORCE_MAX, HOME_Q, MAX_OPENING, tcp_quat
from roboticsbench.skills.base import SkillAPI, SkillError, SkillResult, skill

SPEEDS = {"slow": 0.08, "normal": 0.2, "fast": 0.4}
ARM_COLLISION_N = 45.0  # protective stop threshold (non-finger links)
FINGER_TABLE_N = 80.0  # fingers pressing into static geometry


class ArmSkills(SkillAPI):
    robot_kind = "arm"

    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        self.ik = SiteIK(env.model, "tcp", ARM_JOINTS, posture=HOME_Q, rot_weight=0.6, posture_gain=0.03)
        m = env.model
        self._static = {g for g in range(m.ngeom) if m.body_weldid[m.geom_bodyid[g]] == 0}
        self._robot_nonfinger = self.robot.geom_ids - self.robot.finger_geoms
        self._force = np.zeros(6)
        self._cmd_quat: np.ndarray | None = None  # last commanded TCP orientation

    # ------------------------------------------------------------ internals
    def _collision(self) -> str | None:
        m, d = self.env.model, self.env.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            ours = g1 in self._robot_nonfinger or g2 in self._robot_nonfinger
            fingers_static = ((g1 in self.robot.finger_geoms and g2 in self._static) or
                              (g2 in self.robot.finger_geoms and g1 in self._static))
            if not ours and not fingers_static:
                continue
            other = g2 if g1 in self.robot.geom_ids else g1
            if other in self.robot.geom_ids:
                continue
            mujoco.mj_contactForce(m, d, i, self._force)
            f = abs(self._force[0])
            if (ours and f > ARM_COLLISION_N) or (fingers_static and f > FINGER_TABLE_N):
                name = m.body(m.geom_bodyid[other]).name or m.geom(other).name or "environment"
                part = m.body(m.geom_bodyid[g1 if g1 in self.robot.geom_ids else g2]).name
                return f"{part} collided with {name} ({f:.0f} N)"
        return None

    def _target_quat(self, yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
        r = self.robot
        yaw = math.radians(yaw_deg) if yaw_deg is not None else self._cmd_yaw()
        return tcp_quat(yaw, math.radians(pitch_deg or 0.0), math.radians(roll_deg or 0.0))

    def _cmd_yaw(self) -> float:
        if self._cmd_quat is None:
            return self.robot.tcp_yaw()
        R = quat_to_mat(self._cmd_quat)
        # the last commanded yaw as is: wrapping it into [-90, 90) would spin the wrist 180 deg after e.g. yaw 90
        return math.atan2(R[1, 0], R[0, 0])

    def _move(self, pos, quat, speed: float, settle: float = 0.4) -> SkillResult:
        env, r = self.env, self.robot
        pos = np.asarray(pos, float)
        # plan: final IK must be reachable before we move
        final = self.ik.solve(env.data.qpos, pos, quat, q_init=r.q_target, iters=80)
        if not final.converged:
            msg = (f"target ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) with this orientation is out of reach "
                   f"(IK residual {100 * final.pos_err:.1f} cm / {math.degrees(final.rot_err):.0f} deg); arm did not move")
            return SkillResult(False, msg)
        p0, q0 = r.tcp_pos(), r.tcp_quat()
        dist = float(np.linalg.norm(pos - p0))
        ang = float(np.linalg.norm(rotvec_error(quat, q0)))
        T = max(dist / speed, ang / 1.2, 0.25)
        n = int(math.ceil(T / env.control_dt))
        qcur = r.q_target.copy()
        for k in range(1, n + 1):
            a = smoothstep(k / n)
            p = p0 + a * (pos - p0)
            q = slerp(q0, quat, a)
            sol = self.ik.solve(env.data.qpos, p, q, q_init=qcur, iters=6)
            qcur = sol.q
            r.q_target = qcur.copy()
            if not env.tick():
                return SkillResult(False, "episode ended during motion" + self._outcome_suffix())
            hit = self._collision()
            if hit:
                r.q_target = env.data.qpos[r.q_adr].copy()
                env.run_for(0.2)
                return SkillResult(False, f"protective stop: {hit}. Arm halted at "
                                          f"{np.round(r.tcp_pos(), 3).tolist()}" + self._outcome_suffix())
        # refine from the tracked solution (jumping to `final.q` could snap the arm in its null space)
        r.q_target = self.ik.solve(env.data.qpos, pos, quat, q_init=qcur, iters=40).q
        self._cmd_quat = np.asarray(quat, float).copy()
        t_end = env.t + settle
        while env.running and env.t < t_end:
            env.tick()
            if np.linalg.norm(r.tcp_pos() - pos) < 0.002:
                break
            hit = self._collision()
            if hit:
                r.q_target = env.data.qpos[r.q_adr].copy()
                env.run_for(0.2)
                return SkillResult(False, f"protective stop: {hit}" + self._outcome_suffix())
        err = float(np.linalg.norm(r.tcp_pos() - pos))
        ok = err < 0.012
        tp = r.tcp_pos()
        msg = f"gripper at ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f})"
        if not ok:
            msg += f", {100 * err:.1f} cm short of the target (blocked or holding something heavy?)"
        rot_err = math.degrees(float(np.linalg.norm(rotvec_error(quat, r.tcp_quat()))))
        if rot_err > 2.0:  # position reached but the wrist could not match the commanded orientation
            msg += f"; orientation {rot_err:.0f} deg off the commanded one (wrist limit?)"
        return SkillResult(ok, msg + self._grip_msg() + self._outcome_suffix())

    def _grip_msg(self) -> str:
        r = self.robot
        w = r.finger_width()
        if r.grip_closed:
            held = w > 0.004
            return f"; gripper closed, opening {100 * w:.1f} cm" + (" (holding something)" if held else " (empty)")
        return f"; gripper open {100 * w:.1f} cm"

    # --------------------------------------------------------------- skills
    @skill("Move the gripper tip (TCP, the point between the fingertips) in a straight line to (x, y, z) "
           "in the world frame (metres; table surface is z=0, robot base at the origin, +x away from the "
           "robot). Orientation: pointing straight down, rotated by yaw_deg about vertical (0 = fingers "
           "close along the y axis); omit yaw_deg to keep the current yaw. pitch_deg / roll_deg tilt the "
           "gripper away from vertical (e.g. for pouring).",
           x={"type": "number"}, y={"type": "number"}, z={"type": "number"},
           yaw_deg={"type": "number", "optional": True}, pitch_deg={"type": "number", "optional": True},
           roll_deg={"type": "number", "optional": True},
           speed={"type": "string", "enum": list(SPEEDS)})
    def move_to(self, x: float, y: float, z: float, yaw_deg: float | None = None, pitch_deg: float | None = None,
                roll_deg: float | None = None, speed: str = "normal") -> SkillResult:
        quat = self._target_quat(yaw_deg, pitch_deg, roll_deg)
        return self._move([x, y, z], quat, SPEEDS.get(speed, 0.2))

    @skill("Move the gripper by a relative offset (metres, world frame), keeping its orientation.",
           dx={"type": "number"}, dy={"type": "number"}, dz={"type": "number"},
           speed={"type": "string", "enum": list(SPEEDS)})
    def move_by(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, speed: str = "normal") -> SkillResult:
        p = self.robot.tcp_pos() + np.array([dx, dy, dz])
        q = self._cmd_quat if self._cmd_quat is not None else self.robot.tcp_quat()
        return self._move(p, q, SPEEDS.get(speed, 0.2))

    @skill("Rotate the gripper about the vertical axis to an absolute yaw (degrees), staying in place.",
           yaw_deg={"type": "number"})
    def rotate_gripper(self, yaw_deg: float) -> SkillResult:
        return self._move(self.robot.tcp_pos(), self._target_quat(yaw_deg, 0.0, 0.0), 0.2)

    @skill("Open the gripper (optionally to a specific opening in cm, max 8).",
           width_cm={"type": "number", "optional": True})
    def open_gripper(self, width_cm: float | None = None) -> SkillResult:
        r = self.robot
        r.grip_closed = False
        r.grip_width_target = MAX_OPENING if width_cm is None else float(np.clip(width_cm / 100, 0.0, MAX_OPENING))
        self.env.run_for(0.6)
        return SkillResult(True, "gripper opened" + self._grip_msg() + self._outcome_suffix())

    @skill("Close the gripper with a limited squeeze force (newtons, 5-70; default 40). Reports whether "
           "something ended up between the fingers.",
           force_n={"type": "number"})
    def close_gripper(self, force_n: float = 40.0) -> SkillResult:
        r = self.robot
        r.grip_force = float(np.clip(force_n, 5.0, GRIP_FORCE_MAX))
        r.grip_closed = True
        self.env.run_for(0.8)
        return SkillResult(True, "gripper closed" + self._grip_msg() + self._outcome_suffix())

    @skill("Return to the home (ready) posture above the table.")
    def home(self) -> SkillResult:
        env, r = self.env, self.robot
        q0 = r.q_target.copy()
        n = int(1.5 / env.control_dt)
        for k in range(1, n + 1):
            r.q_target = q0 + smoothstep(k / n) * (HOME_Q - q0)
            if not env.tick():
                break
            hit = self._collision()
            if hit:
                r.q_target = env.data.qpos[r.q_adr].copy()
                return SkillResult(False, f"protective stop: {hit}" + self._outcome_suffix())
        env.run_for(0.3)
        self._cmd_quat = None
        return SkillResult(True, "at home posture" + self._grip_msg() + self._outcome_suffix())

    @skill("Read the wrist force/torque sensor (world frame, tared at episode start). A held object of mass m "
           "adds roughly (0, 0, -9.81*m) N of force.")
    def read_wrist_force(self) -> SkillResult:
        self.env.run_for(0.3)
        w = self.robot.wrist_ft()
        f, t = w[:3], w[3:]
        return SkillResult(True, f"force_N=({f[0]:+.2f}, {f[1]:+.2f}, {f[2]:+.2f}), "
                                 f"torque_Nm=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})",
                           {"force": [round(float(x), 3) for x in f], "torque": [round(float(x), 4) for x in t]})

    # expose the orientation helper for oracles
    @staticmethod
    def quat(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
        return tcp_quat(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))


def rot_z_mat(yaw: float) -> np.ndarray:
    return quat_to_mat(np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))


__all__ = ["ArmSkills", "SkillError"]
