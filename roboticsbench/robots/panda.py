"""Franka Emika Panda (MuJoCo Menagerie model) on a table.

* Arm: 7 position servos (Menagerie gains) + gravity compensation on the links (as the real
  robot's controller does).
* Gripper: force-limited parallel gripper. The Menagerie tendon actuator is converted to a
  force actuator and closed with a Franka-Hand-like controller: position PD while moving, capped
  squeeze force (default 40 N, max 70 N) once in contact.
* Wrist force/torque sensor between link7 and the hand (tare-able).
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from roboticsbench.assets import PANDA_DIR, PANDA_XML, ensure_assets
from roboticsbench.core.geometry import lookat_xyaxes, mat_to_quat, quat_mul, quat_to_mat, quat_yaw, yaw_of_mat
from roboticsbench.robots.base import Robot

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
MAX_JOINT_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s (Franka datasheet)
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
MAX_OPENING = 0.08  # m between fingers
GRIP_FORCE_DEFAULT = 40.0
GRIP_FORCE_MAX = 70.0
GRIP_KP = 6000.0
GRIP_KV = 60.0
GRIP_SPEED = 0.12  # m/s of tendon length (fingers close at ~2x this relative speed)
# TCP orientation "pointing down" with fingers opening along world y when yaw=0.
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x


def tcp_quat(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Top-down grasp orientation with extra yaw (about world z), then pitch/roll tilts
    (about the gripper's own y / x axes)."""
    q = quat_mul(quat_yaw(yaw), DOWN)
    if pitch:
        q = quat_mul(q, np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    if roll:
        q = quat_mul(q, np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]))
    return q


class Panda(Robot):
    name = "panda"
    physics_dt = 0.002
    control_dt = 0.02
    default_cameras = ("front", "wrist")
    video_camera = "front"
    # every arm task, the same two: the workcell's fixed overhead camera (tasks/arm/common.py) and the gripper camera;
    # task-specific close-ups stay for videos and the live view only
    agent_cameras = ("top", "wrist")

    def __init__(self, base_pos=(0.0, 0.0, 0.0)):
        super().__init__()
        ensure_assets()
        self.base_pos = np.asarray(base_pos, float)
        self.q_target = HOME_Q.copy()
        self.grip_closed = False
        self.grip_force = GRIP_FORCE_DEFAULT
        self.grip_width_target = MAX_OPENING
        self.grip_force_open = GRIP_FORCE_MAX  # force limit while driving to grip_width_target (not closed)
        self.control_mode = "skills"
        self.q_goal: np.ndarray | None = None  # motor control: q_target moves toward this at MAX_JOINT_VEL
        self._ft_bias = np.zeros(6)

    @property
    def root_body(self) -> str:
        return "link0"

    def build_spec(self) -> mujoco.MjSpec:
        xml = PANDA_XML.read_text().replace('meshdir="assets"', f'meshdir="{(PANDA_DIR / "assets").as_posix()}"')
        spec = mujoco.MjSpec.from_string(xml)
        spec.option.timestep = self.physics_dt
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.noslip_iterations = 3
        for k in list(spec.keys):
            spec.delete(k)
        spec.body("link0").pos = self.base_pos.tolist()
        for i in range(1, 8):
            spec.body(f"link{i}").gravcomp = 1.0
        for n in ("hand", "left_finger", "right_finger"):
            spec.body(n).gravcomp = 1.0
        hand = spec.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.006] * 3, group=4, rgba=[1, 0, 0, 0.6])
        hand.add_site(name="ft_site", pos=[0, 0, 0], size=[0.01] * 3, group=4)
        cam_pos = [0.055, 0.0, 0.03]
        hand.add_camera(name="wrist", pos=cam_pos, fovy=75,
                        xyaxes=lookat_xyaxes(cam_pos, [0.0, 0.0, 0.25], up=(1, 0, 0)))
        spec.add_sensor(name="wrist_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft_site")
        spec.add_sensor(name="wrist_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft_site")
        # gripper: stiff tendon position servo whose force limit is set at runtime -> the squeeze force
        # saturates at the commanded grasp force (Franka-Hand-like force-limited grasp). ctrl is the
        # target tendon length (= half the finger opening); negative targets keep squeezing thin parts.
        g = spec.actuator("actuator8")
        g.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        g.gainprm[0] = GRIP_KP
        g.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        g.biasprm[:3] = [0.0, -GRIP_KP, -GRIP_KV]
        g.ctrlrange = [-0.012, MAX_OPENING / 2]
        g.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        g.forcerange = [-GRIP_FORCE_MAX, GRIP_FORCE_MAX]
        # grippy finger pads
        for gm in spec.geoms:
            if gm.classname and gm.classname.name.startswith("fingertip_pad"):
                gm.friction = [1.5, 0.02, 0.002]
                gm.condim = 4
        return spec

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().bind(model, data)
        m = model
        ids = [m.joint(j).id for j in ARM_JOINTS]
        self.q_adr = np.array([m.jnt_qposadr[j] for j in ids])
        self.v_adr = np.array([m.jnt_dofadr[j] for j in ids])
        self.lo = m.jnt_range[ids, 0]
        self.hi = m.jnt_range[ids, 1]
        self.arm_act = np.array([m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.grip_act = m.actuator("actuator8").id
        self.f1 = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.f2 = m.jnt_qposadr[m.joint("finger_joint2").id]
        self.f1v = m.jnt_dofadr[m.joint("finger_joint1").id]
        self.f2v = m.jnt_dofadr[m.joint("finger_joint2").id]
        self.tcp = m.site("tcp").id
        self.finger_bodies = {m.body("left_finger").id, m.body("right_finger").id}
        self.finger_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.finger_bodies}
        fs, ts = m.sensor("wrist_force"), m.sensor("wrist_torque")
        self._fs = slice(fs.adr[0], fs.adr[0] + 3)
        self._ts = slice(ts.adr[0], ts.adr[0] + 3)

    def reset(self, rng: np.random.Generator, q: np.ndarray | None = None, **_: Any) -> None:
        d = self.data
        q0 = HOME_Q if q is None else np.asarray(q, float)
        d.qpos[self.q_adr] = q0
        d.qpos[self.f1] = d.qpos[self.f2] = MAX_OPENING / 2
        self.q_target = q0.copy()
        self.grip_closed = False
        self.grip_force = GRIP_FORCE_DEFAULT
        self.grip_width_target = MAX_OPENING
        self.grip_force_open = GRIP_FORCE_MAX
        self._grip_ctrl = MAX_OPENING / 2
        self.q_goal = None
        d.ctrl[self.arm_act] = q0

    # ---------------------------------------------------------------- control
    def pre_tick(self) -> None:
        if self.q_goal is not None:  # motor control: rate-limited joint targets (Franka motion-generator limits)
            step = MAX_JOINT_VEL * self.control_dt
            self.q_target = self.q_target + np.clip(self.q_goal - self.q_target, -step, step)

    def substep(self) -> None:
        d, m = self.data, self.model
        d.ctrl[self.arm_act] = np.clip(self.q_target, self.lo, self.hi)
        if self.grip_closed:
            goal, fmax = -0.012, self.grip_force
        else:
            goal, fmax = min(self.grip_width_target, MAX_OPENING) / 2, self.grip_force_open
            # opening: restart the (virtual) target at the fingers - while closed it runs up to 3 cm inside them,
            # and ramping it back under the 70 N opening force limit would first squeeze the held object hard
            half = 0.5 * (d.qpos[self.f1] + d.qpos[self.f2])
            if goal >= half:
                self._grip_ctrl = max(self._grip_ctrl, half - 0.0005)
        step = GRIP_SPEED * m.opt.timestep
        self._grip_ctrl += float(np.clip(goal - self._grip_ctrl, -step, step))
        if self.grip_closed:
            # don't let the (virtual) target run away from the fingers while squeezing
            L = 0.5 * (d.qpos[self.f1] + d.qpos[self.f2])
            self._grip_ctrl = max(self._grip_ctrl, L - 0.03)
        d.ctrl[self.grip_act] = self._grip_ctrl
        m.actuator_forcerange[self.grip_act] = (-fmax, fmax)

    # ------------------------------------------------------------------ state
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    def tcp_quat(self) -> np.ndarray:
        return mat_to_quat(self.data.site_xmat[self.tcp])

    def tcp_yaw(self) -> float:
        """Gripper yaw about world z (0 = fingers close along world y). Defined modulo 180 deg."""
        R = self.data.site_xmat[self.tcp].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        return (yaw + math.pi / 2) % math.pi - math.pi / 2

    def tcp_down_angle_deg(self) -> float:
        R = self.data.site_xmat[self.tcp].reshape(3, 3)
        return math.degrees(math.acos(float(np.clip(-R[2, 2], -1, 1))))

    def finger_width(self) -> float:
        return float(self.data.qpos[self.f1] + self.data.qpos[self.f2])

    def wrist_ft(self, raw: bool = False) -> np.ndarray:
        """External wrench on the hand (N, Nm) in the world frame, minus the tare taken at reset.
        A held mass m therefore reads about (0, 0, -9.81 m) N."""
        d = self.data
        R = d.site("ft_site").xmat.reshape(3, 3)
        f = -(R @ d.sensordata[self._fs])
        t = -(R @ d.sensordata[self._ts])
        w = np.concatenate([f, t])
        return w if raw else w - self._ft_bias

    def tare(self) -> None:
        self._ft_bias = self.wrist_ft(raw=True)

    def contacts_with(self, geoms: set[int]) -> list[int]:
        """Geoms (from ``geoms``) currently touching a finger."""
        d = self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if a in self.finger_geoms and b in geoms:
                out.append(b)
            elif b in self.finger_geoms and a in geoms:
                out.append(a)
        return out

    def proprio(self) -> dict[str, Any]:
        p = self.tcp_pos()
        return {
            "gripper_position": [round(float(x), 4) for x in p],
            "gripper_yaw_deg": round(math.degrees(self.tcp_yaw()), 1),
            "gripper_tilt_deg": round(self.tcp_down_angle_deg(), 1),
            "finger_opening_cm": round(100 * self.finger_width(), 2),
            "gripper_closed": self.grip_closed,
            "joint_positions": [round(float(x), 3) for x in self.data.qpos[self.q_adr]],
        }


__all__ = ["Panda", "tcp_quat", "HOME_Q", "ARM_JOINTS", "quat_to_mat", "yaw_of_mat"]
