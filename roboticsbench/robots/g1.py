"""Unitree G1 humanoid (29 body DoF + two 7-DoF Dex3 hands).

Lower body: GR00T WBC balance/walk policy (50 Hz) -> PD torques at 200 Hz.
Upper body: arm joint-position targets (PD, real-robot gains), hands via position servos.

Control modes (``G1.mode``):
    "wbc"   -- velocity/height/torso commands drive the legs; arms follow ``arm_target``.
    "joint" -- Unitree-style low-level motor commands for all 29 body joints (no balance help!): per joint
               tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff (``joint_target``, ``joint_dq_target``,
               ``joint_kp``, ``joint_kd``, ``joint_tau_ff``), clipped by the motors' torque limits (in the MJCF).
    "torque"-- raw torques from ``torque_cmd`` (29).
Hands: either the scalar open/close ``hand_cmd`` (skills) or per-joint position targets ``hand_target`` (14,
used when ``hand_direct`` is set, e.g. under motor control).

``control_mode`` = "motor" makes :meth:`reset` start in "joint" mode holding the standing posture with the stiff
``HOLD_KP``/``HOLD_KD`` gains (the robot stands still like a statue; walking needs a controller).
"""

from __future__ import annotations

import math
import re
from typing import Any

import mujoco
import numpy as np

from roboticsbench.assets import G1_DIR, G1_XML, ensure_assets
from roboticsbench.control import wbc as W
from roboticsbench.core.geometry import gravity_in_body, lookat_xyaxes, quat_yaw, yaw_of_quat
from roboticsbench.robots.base import Robot

LEG_WAIST = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
ARM_SUFFIX = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
LEFT_ARM = [f"left_{s}_joint" for s in ARM_SUFFIX]
RIGHT_ARM = [f"right_{s}_joint" for s in ARM_SUFFIX]
ARMS = LEFT_ARM + RIGHT_ARM
BODY = LEG_WAIST + ARMS  # policy joint order (29)
HAND_SUFFIX = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
LEFT_HAND = [f"left_hand_{s}_joint" for s in HAND_SUFFIX]
RIGHT_HAND = [f"right_hand_{s}_joint" for s in HAND_SUFFIX]

ARM_KP = np.array([100, 100, 40, 40, 20, 20, 20] * 2, dtype=np.float64)
ARM_KD = np.array([5, 5, 2, 2, 2, 2, 2] * 2, dtype=np.float64)
BODY_KP = np.concatenate([W.LEG_WAIST_KP, ARM_KP])
BODY_KD = np.concatenate([W.LEG_WAIST_KD, ARM_KD])

# Hand closure poses (fully closed "power grasp"); open = zeros.
LEFT_HAND_CLOSED = np.array([0.0, 0.55, 1.05, -1.1, -1.2, -1.1, -1.2])
RIGHT_HAND_CLOSED = np.array([0.0, -0.55, -1.05, 1.1, 1.2, 1.1, 1.2])

# Arm postures (left 7, right 7).  Elbow at 0 = forearm horizontal forward on the G1.
ARM_POSES: dict[str, np.ndarray] = {
    "ready": np.array([0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "zero": np.zeros(14),
    "forward": np.array([-0.9, 0.15, 0.0, 0.3, 0.0, 0.0, 0.0, -0.9, -0.15, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "push": np.array([-1.25, 0.18, 0.0, 0.25, 0.0, 0.0, 0.0, -1.25, -0.18, 0.0, 0.25, 0.0, 0.0, 0.0]),
    "tucked": np.array([0.0, -0.05, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.3, 0.0, 0.0, 0.0]),
}

# motor-mode start: hold the standing posture with stiff leg gains (the real-robot gains alone cannot hold a static
# stance - the ankles are too soft for the body's inverted-pendulum moment)
HOLD_KP = BODY_KP.copy()
HOLD_KP[:12] *= 2.0
HOLD_KP[[4, 5, 10, 11]] = 300.0
HOLD_KD = BODY_KD.copy()
HOLD_KD[[4, 5, 10, 11]] = 6.0

# motor-mode start pose (legs 12 + waist 3): a symmetric crouch that the hold gains keep standing indefinitely
# Motor torque-speed curves (the real G1's joint speed limits by motor class: 139 N m -> 20 rad/s, 88 N m -> 32,
# 50 / 25 N m -> 37, 5 N m -> 22): full torque up to 70 % of the no-load speed, falling to zero at it while the motor
# drives the joint faster (braking is not limited). Without them a full-torque, zero-gain command spun a fallen
# robot's joints up until the physics diverged (2026-10-09).
NO_LOAD_SPEED = {139.0: 20.0, 88.0: 32.0, 50.0: 37.0, 25.0: 37.0, 5.0: 22.0}
FULL_TORQUE_FRAC = 0.7


def motor_torque_limit(tau: np.ndarray, dq: np.ndarray, tau_max: np.ndarray, dq_max: np.ndarray) -> np.ndarray:
    """Clip commanded torques to what the motors can deliver at the current joint speeds."""
    fade = np.clip((dq_max - np.abs(dq)) / ((1.0 - FULL_TORQUE_FRAC) * dq_max), 0.0, 1.0)
    lim = np.where(tau * dq > 0.0, tau_max * fade, tau_max)
    return np.clip(tau, -lim, lim)


MOTOR_STANCE = np.array([-0.35, 0.0, 0.0, 0.72, -0.32, 0.0, -0.35, 0.0, 0.0, 0.72, -0.32, 0.0, 0.0, 0.0, 0.08])

STAND_PELVIS_Z = 0.793
SLEW_V = np.array([1.2, 1.2, 2.5])  # max command change per second (m/s^2, m/s^2, rad/s^2)
SLEW_H = 0.25  # m/s
ARM_MAX_VEL = 2.5  # rad/s on arm targets (default; velocity-mode commands may raise it up to ARM_SPEED_MAX)
ARM_SPEED_MAX = 10.0  # rad/s - fast swings (throwing); the real arm joints are faster, the balance policy is the limit


def _robot_xml(hands: bool) -> str:
    xml = G1_XML.read_text()
    # drop the demo scene (sky, floor, light) that ships with the model
    xml = xml[: xml.index("<!-- setup scene -->")] + "</mujoco>\n"
    if hands:
        # un-comment Dex3 finger joints; actuators are added programmatically
        xml = re.sub(r"<!--\s*(<joint name=\"(left|right)_hand_[a-z_0-9]+\"[^>]*/>)\s*-->", r"\1", xml)
    xml = xml.replace('meshdir="meshes"', f'meshdir="{(G1_DIR / "meshes").as_posix()}"')
    return xml


class G1(Robot):
    name = "g1"
    physics_dt = 0.005
    control_dt = 0.02
    default_cameras = ("head", "chase")
    video_camera = "chase"
    agent_cameras = ("head",)  # a walking robot sees only through its own head camera - no chase / map / drone views

    def __init__(self, hands: bool = True):
        super().__init__()
        ensure_assets()
        self.hands = hands
        self.arm_gravity_comp = True
        self.wbc = W.GearWbcPolicy()
        self.mode = "wbc"
        # commands (targets); see reset() for defaults
        self.vel_cmd = np.zeros(3)
        self.height_cmd = W.DEFAULT_HEIGHT
        self.torso_rpy_cmd = np.zeros(3)
        self.arm_target = ARM_POSES["zero"].copy()
        self.arm_max_vel = ARM_MAX_VEL  # slew limit on arm targets (rad/s)
        self.hand_cmd = np.zeros(2)  # 0 open .. 1 closed (left, right)
        self.hand_target = np.zeros(14)  # per-joint hand targets (left 7, right 7) when hand_direct
        self.hand_direct = False
        self.joint_target = np.zeros(29)
        self.joint_dq_target = np.zeros(29)
        self.joint_kp = BODY_KP.copy()
        self.joint_kd = BODY_KD.copy()
        self.joint_tau_ff = np.zeros(29)
        self.torque_cmd = np.zeros(29)
        self.control_mode = "wbc"  # "motor": start in joint mode holding the stance (see module docstring)

    # ------------------------------------------------------------------ model
    @property
    def root_body(self) -> str:
        return "pelvis"

    def build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_string(_robot_xml(self.hands))
        spec.option.timestep = self.physics_dt
        for g in spec.geoms:  # collision copies of the visual meshes: hide them (group 3) -> half the render cost
            if g.contype != 0 or g.conaffinity != 0:
                g.group = 3
        torso = spec.body("torso_link")
        head_pos = [0.075, 0.0, 0.43]
        pitch = math.radians(28)
        fwd = [math.cos(pitch), 0.0, -math.sin(pitch)]
        torso.add_camera(name="head", pos=head_pos, fovy=80,
                         xyaxes=lookat_xyaxes(head_pos, np.add(head_pos, fwd)))
        torso.add_site(name="head_site", pos=head_pos, size=[0.01] * 3, group=4)
        for side in ("left", "right"):
            wy = spec.body(f"{side}_wrist_yaw_link")
            wy.add_site(name=f"{side}_palm", pos=[0.10, 0.0, 0.0], size=[0.012] * 3, group=4,
                        rgba=[1, 0, 1, 0.5])
            spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot", pos=[0.035, 0, -0.03],
                                                          size=[0.01] * 3, group=4)
        if self.hands:
            for j in LEFT_HAND + RIGHT_HAND:
                a = spec.add_actuator(name=f"{j}_pos", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                a.set_to_position(kp=4.0, kv=0.15)
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                a.forcerange = [-2.45, 2.45] if "thumb_0" in j else [-1.4, 1.4]
        return spec

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().bind(model, data)
        m = model
        jid = lambda n: m.joint(n).id  # noqa: E731
        self.q_adr = np.array([m.jnt_qposadr[jid(n)] for n in BODY])
        self.v_adr = np.array([m.jnt_dofadr[jid(n)] for n in BODY])
        self.act_ids = np.array([m.actuator(n).id for n in BODY])
        self.tau_max = m.jnt_actfrcrange[[jid(n) for n in BODY], 1].copy()
        self.dq_max = np.array([NO_LOAD_SPEED[float(round(t))] for t in self.tau_max])
        self.base_qadr = m.jnt_qposadr[m.body("pelvis").jntadr[0]]
        self.base_vadr = m.jnt_dofadr[m.body("pelvis").jntadr[0]]
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id
        if self.hands:
            self.hand_act = np.array([m.actuator(f"{j}_pos").id for j in LEFT_HAND + RIGHT_HAND])
            self.hand_qadr = np.array([m.jnt_qposadr[jid(j)] for j in LEFT_HAND + RIGHT_HAND])
        self.foot_geoms = set()
        for side in ("left", "right"):
            b = m.body(f"{side}_ankle_roll_link").id
            self.foot_geoms |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]}
        self.foot_bodies = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
        lo = m.jnt_range[[jid(n) for n in ARMS], 0]
        hi = m.jnt_range[[jid(n) for n in ARMS], 1]
        self.arm_lo, self.arm_hi = lo + 0.02, hi - 0.02
        self._leg_target = W.LEG_WAIST_DEFAULT.astype(np.float64).copy()
        self._arm_cmd = self.arm_target.copy()
        self._vel = np.zeros(3)
        self._height = W.DEFAULT_HEIGHT
        # pelvis height that puts the soles on the floor in MOTOR_STANCE (forward kinematics on a scratch copy)
        dd = mujoco.MjData(m)
        dd.qpos[self.base_qadr:self.base_qadr + 3] = [0.0, 0.0, 1.0]
        dd.qpos[self.base_qadr + 3:self.base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        dd.qpos[self.q_adr[:15]] = MOTOR_STANCE
        mujoco.mj_kinematics(m, dd)
        self._motor_z = 1.0 - min(dd.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geoms) + 0.002

    def reset(self, rng: np.random.Generator, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, **_: Any) -> None:
        d = self.data
        d.qpos[self.base_qadr:self.base_qadr + 3] = [x, y, STAND_PELVIS_Z]
        d.qpos[self.base_qadr + 3:self.base_qadr + 7] = quat_yaw(yaw)
        d.qpos[self.q_adr[:15]] = W.LEG_WAIST_DEFAULT
        d.qpos[self.q_adr[15:]] = ARM_POSES["zero"]
        self.wbc.reset()
        self.mode = "wbc"
        self.vel_cmd[:] = 0
        self._vel[:] = 0
        self.height_cmd = self._height = W.DEFAULT_HEIGHT
        self.torso_rpy_cmd[:] = 0
        self.arm_target = ARM_POSES["zero"].copy()
        self._arm_cmd = self.arm_target.copy()
        self.arm_max_vel = ARM_MAX_VEL
        self.hand_cmd[:] = 0
        self.hand_target[:] = 0
        self.hand_direct = False
        self._leg_target = W.LEG_WAIST_DEFAULT.astype(np.float64).copy()
        self.joint_target = d.qpos[self.q_adr].copy()
        self.joint_dq_target = np.zeros(29)
        self.joint_kp = BODY_KP.copy()
        self.joint_kd = BODY_KD.copy()
        self.joint_tau_ff = np.zeros(29)
        if self.control_mode == "motor":
            d.qpos[self.base_qadr + 2] = self._motor_z
            d.qpos[self.q_adr[:15]] = MOTOR_STANCE
            self.mode = "joint"
            self.joint_target = d.qpos[self.q_adr].copy()
            self.joint_kp, self.joint_kd = HOLD_KP.copy(), HOLD_KD.copy()
            self.hand_direct = True

    # ---------------------------------------------------------------- control
    def pre_tick(self) -> None:
        d, dt = self.data, self.control_dt
        if self.mode == "wbc":
            tgt = np.array([np.clip(self.vel_cmd[0], *W.VX_RANGE), np.clip(self.vel_cmd[1], *W.VY_RANGE),
                            np.clip(self.vel_cmd[2], *W.WZ_RANGE)])
            self._vel += np.clip(tgt - self._vel, -SLEW_V * dt, SLEW_V * dt)
            if np.linalg.norm(tgt) < 1e-6 and np.linalg.norm(self._vel) < 0.06:
                self._vel[:] = 0.0  # snap to the balance policy when stopping
            h = float(np.clip(self.height_cmd, *W.HEIGHT_RANGE))
            self._height += float(np.clip(h - self._height, -SLEW_H * dt, SLEW_H * dt))
            base_quat = d.qpos[self.base_qadr + 3:self.base_qadr + 7]
            angvel = d.qvel[self.base_vadr + 3:self.base_vadr + 6]
            self._leg_target = self.wbc.step(d.qpos[self.q_adr], d.qvel[self.v_adr], base_quat, angvel,
                                             self._vel, self._height, self.torso_rpy_cmd).astype(np.float64)
        tgt = np.clip(self.arm_target, self.arm_lo, self.arm_hi)
        step = self.arm_max_vel * dt
        self._arm_cmd += np.clip(tgt - self._arm_cmd, -step, step)

    def substep(self) -> None:
        d = self.data
        q = d.qpos[self.q_adr]
        dq = d.qvel[self.v_adr]
        if self.mode == "wbc":
            target = np.concatenate([self._leg_target, self._arm_cmd])
            tau = BODY_KP * (target - q) - BODY_KD * dq
            if self.arm_gravity_comp:
                # model-based gravity/Coriolis feed-forward on the arms (as real arm controllers do)
                tau[15:] += d.qfrc_bias[self.v_adr[15:]]
        elif self.mode == "joint":
            tau = (self.joint_kp * (self.joint_target - q) + self.joint_kd * (self.joint_dq_target - dq)
                   + self.joint_tau_ff)
        else:
            tau = self.torque_cmd
        d.ctrl[self.act_ids] = motor_torque_limit(tau, dq, self.tau_max, self.dq_max)
        if self.hands:
            if self.hand_direct:
                d.ctrl[self.hand_act] = self.hand_target
            else:
                g = np.clip(self.hand_cmd, 0.0, 1.0)
                d.ctrl[self.hand_act[:7]] = g[0] * LEFT_HAND_CLOSED
                d.ctrl[self.hand_act[7:]] = g[1] * RIGHT_HAND_CLOSED

    # ------------------------------------------------------------------ state
    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self.base_qadr:self.base_qadr + 3].copy()

    @property
    def base_quat(self) -> np.ndarray:
        return self.data.qpos[self.base_qadr + 3:self.base_qadr + 7].copy()

    @property
    def yaw(self) -> float:
        return yaw_of_quat(self.base_quat)

    @property
    def base_vel_world(self) -> np.ndarray:
        return self.data.qvel[self.base_vadr:self.base_vadr + 3].copy()

    def tilt_deg(self) -> float:
        g = gravity_in_body(self.base_quat)
        return math.degrees(math.acos(float(np.clip(-g[2], -1, 1))))

    def palm_pos(self, side: str) -> np.ndarray:
        return self.data.site(f"{side}_palm").xpos.copy()

    def head_pos(self) -> np.ndarray:
        return self.data.site("head_site").xpos.copy()

    def world_to_robot(self, p: np.ndarray) -> np.ndarray:
        """World point -> robot heading frame (x forward, y left, z up from floor)."""
        b = self.base_pos
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = p[0] - b[0], p[1] - b[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def robot_to_world(self, p: np.ndarray) -> np.ndarray:
        b = self.base_pos
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([b[0] + c * p[0] - s * p[1], b[1] + s * p[0] + c * p[1], p[2]])

    def is_fallen(self) -> bool:
        if self.base_pos[2] < 0.35 or self.tilt_deg() > 55:
            return True
        return False

    def proprio(self) -> dict[str, Any]:
        b = self.base_pos
        v = self.base_vel_world
        return {
            "position_xy": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "heading_deg": round(math.degrees(self.yaw), 1),
            "pelvis_height": round(float(b[2]), 3),
            "speed": round(float(np.linalg.norm(v[:2])), 3),
            "tilt_deg": round(self.tilt_deg(), 1),
            "left_palm": [round(float(x), 3) for x in self.palm_pos("left")],
            "right_palm": [round(float(x), 3) for x in self.palm_pos("right")],
            "hands_closed": [round(float(x), 2) for x in self.hand_cmd],
            "fallen": self.is_fallen(),
        }

    def joint_state(self) -> dict[str, np.ndarray]:
        d = self.data
        return {"q": d.qpos[self.q_adr].copy(), "dq": d.qvel[self.v_adr].copy(),
                "base_pos": self.base_pos, "base_quat": self.base_quat,
                "base_angvel": d.qvel[self.base_vadr + 3:self.base_vadr + 6].copy(),
                "base_linvel": self.base_vel_world}

    # chase camera (free camera) parameters: lookat, distance, azimuth, elevation
    def chase_view(self, distance: float = 3.2, elevation: float = -18.0, side_deg: float = 25.0):
        b = self.base_pos
        return [b[0], b[1], 0.8], distance, math.degrees(self.yaw) + side_deg, elevation
