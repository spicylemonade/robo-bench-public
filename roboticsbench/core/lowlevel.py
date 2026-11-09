"""Low-level (policy) control interface: gym-style ``step(action)`` over several action spaces.

This is the interface for custom robot-learning models (VLAs, RL policies, whole-body controllers) that act at a
fixed control rate, as opposed to the high-level skill API used by LLM agents.

Action spaces
-------------
Panda
    ``ee_delta`` (7)   [dx, dy, dz (m), droll, dpitch, dyaw (rad, world frame), gripper (0 open .. 1 closed)]
    ``ee_pose``  (8)   [x, y, z, qw, qx, qy, qz, gripper]  (absolute TCP pose)
    ``joint_pos`` (8)  [q1..q7 (rad), gripper]
G1
    ``wbc``      (20)  [vx, vy (m/s), wz (rad/s), pelvis height (m), 14 arm joint targets (rad), left hand, right hand]
                       - legs are driven by the built-in GR00T whole-body controller
    ``joint_pos`` (31) [29 body joint targets (rad, PD with real-robot gains), left hand, right hand]
                       - no balance help: your policy must keep the robot upright
    ``torque``   (29)  [29 body joint torques (N m)]

Observation (dict)
------------------
    images       {camera: HxWx3 uint8}
    state        1-D float32 proprioceptive vector (layout in ``state_layout()``)
    instruction  str
    objects      privileged object list (only when ``privileged=True``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from roboticsbench.core.geometry import quat_axis_angle, quat_mul

SPACES = {"panda": {"ee_delta": 7, "ee_pose": 8, "joint_pos": 8},
          "g1": {"wbc": 20, "joint_pos": 31, "torque": 29}}
DEFAULT_SPACE = {"panda": "ee_delta", "g1": "wbc"}


@dataclass
class StepResult:
    obs: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class LowLevelControl:
    def __init__(self, env, space: str | None = None, control_hz: float = 10.0,
                 cameras: tuple[str, ...] | None = None, image_size: tuple[int, int] = (256, 256),
                 privileged: bool = False):
        self.env = env
        self.robot = env.robot
        rname = self.robot.name
        self.space = space or DEFAULT_SPACE[rname]
        if self.space not in SPACES[rname]:
            raise ValueError(f"action space {self.space!r} not available for {rname}: {sorted(SPACES[rname])}")
        self.dim = SPACES[rname][self.space]
        self.control_hz = float(control_hz)
        self.repeat = max(1, int(round(1.0 / (self.control_hz * env.control_dt))))
        self.cameras = tuple(cameras or env.obs_cameras)
        self.image_size = image_size
        self.privileged = privileged
        self._last_progress = env.progress
        self._ik = None
        if rname == "panda" and self.space in ("ee_delta", "ee_pose"):
            from roboticsbench.control.ik import SiteIK
            from roboticsbench.robots.panda import ARM_JOINTS, HOME_Q

            self._ik = SiteIK(env.model, "tcp", ARM_JOINTS, posture=HOME_Q, rot_weight=0.6, posture_gain=0.03)
            self._tgt_pos = self.robot.tcp_pos()
            self._tgt_quat = self.robot.tcp_quat()
        if rname == "g1" and self.space in ("joint_pos", "torque"):
            self.robot.mode = "joint" if self.space == "joint_pos" else "torque"
            self.robot.joint_target = env.data.qpos[self.robot.q_adr].copy()

    # ----------------------------------------------------------------- spaces
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r, s = self.robot.name, self.space
        if r == "panda" and s == "ee_delta":
            return np.array([-.05] * 3 + [-.3] * 3 + [0.0]), np.array([.05] * 3 + [.3] * 3 + [1.0])
        if r == "panda" and s == "ee_pose":
            return np.array([-1, -1, -0.1, -1, -1, -1, -1, 0.0]), np.array([1, 1, 1, 1, 1, 1, 1, 1.0])
        if r == "panda":
            return np.concatenate([self.robot.lo, [0.0]]), np.concatenate([self.robot.hi, [1.0]])
        if s == "wbc":
            lo = np.concatenate([[-0.6, -0.5, -1.0, 0.55], self.robot.arm_lo, [0, 0]])
            hi = np.concatenate([[1.0, 0.5, 1.0, 0.80], self.robot.arm_hi, [1, 1]])
            return lo, hi
        if s == "joint_pos":
            m = self.env.model
            rng = m.jnt_range[[m.joint(n).id for n in _g1_body()]]
            return np.concatenate([rng[:, 0], [0, 0]]), np.concatenate([rng[:, 1], [1, 1]])
        return -np.full(29, 139.0), np.full(29, 139.0)

    def state_layout(self) -> list[tuple[str, int]]:
        if self.robot.name == "panda":
            return [("q", 7), ("dq", 7), ("gripper_width", 1), ("tcp_pos", 3), ("tcp_quat", 4), ("wrist_ft", 6)]
        return [("base_pos", 3), ("base_quat", 4), ("base_linvel", 3), ("base_angvel", 3), ("q", 29), ("dq", 29),
                ("hands", 2)]

    def state(self) -> np.ndarray:
        r, d = self.robot, self.env.data
        if r.name == "panda":
            parts = [d.qpos[r.q_adr], d.qvel[r.v_adr], [r.finger_width()], r.tcp_pos(), r.tcp_quat(), r.wrist_ft()]
        else:
            js = r.joint_state()
            parts = [js["base_pos"], js["base_quat"], js["base_linvel"], js["base_angvel"], js["q"], js["dq"],
                     r.hand_cmd]
        return np.concatenate([np.asarray(p, np.float64).ravel() for p in parts]).astype(np.float32)

    def observe(self) -> dict[str, Any]:
        w, h = self.image_size
        obs = {"images": {c: self.env.render(c, w, h) for c in self.cameras}, "state": self.state(),
               "instruction": self.env.task.instruction, "time": self.env.t}
        if self.privileged:
            obs["objects"] = self.env.task.objects(self.env)
        return obs

    # ------------------------------------------------------------------- step
    def _apply(self, a: np.ndarray) -> None:
        r, s = self.robot, self.space
        if r.name == "panda":
            grip = float(a[-1])
            r.grip_closed = grip > 0.5
            if s == "joint_pos":
                r.q_target = np.clip(a[:7], r.lo, r.hi)
            else:
                if s == "ee_delta":
                    self._tgt_pos = np.clip(self._tgt_pos + a[:3], [-0.2, -0.8, -0.05], [1.0, 0.8, 0.9])
                    rv = a[3:6]
                    ang = float(np.linalg.norm(rv))
                    if ang > 1e-9:
                        self._tgt_quat = quat_mul(quat_axis_angle(rv / ang, ang), self._tgt_quat)
                else:
                    self._tgt_pos = np.asarray(a[:3], float)
                    q = np.asarray(a[3:7], float)
                    self._tgt_quat = q / (np.linalg.norm(q) + 1e-12)
                sol = self._ik.solve(self.env.data.qpos, self._tgt_pos, self._tgt_quat, q_init=r.q_target, iters=15)
                r.q_target = sol.q
        else:
            if s == "wbc":
                r.mode = "wbc"
                r.vel_cmd[:] = a[:3]
                r.height_cmd = float(a[3])
                r.arm_target[:] = a[4:18]
                r.hand_cmd[:] = np.clip(a[18:20], 0, 1)
            elif s == "joint_pos":
                r.mode = "joint"
                r.joint_target = np.asarray(a[:29], float)
                r.hand_cmd[:] = np.clip(a[29:31], 0, 1)
            else:
                r.mode = "torque"
                r.torque_cmd = np.asarray(a[:29], float)

    def step(self, action) -> StepResult:
        a = np.asarray(action, float).ravel()
        if a.shape[0] != self.dim:
            raise ValueError(f"expected action of size {self.dim} for {self.space}, got {a.shape[0]}")
        lo, hi = self.action_bounds()
        a = np.clip(a, lo, hi)
        self._apply(a)
        env = self.env
        env.tick(self.repeat)
        reward = float(env.progress - self._last_progress)
        self._last_progress = env.progress
        terminated = env.status in ("success", "failure")
        truncated = env.status == "timeout"
        return StepResult(self.observe(), reward, terminated, truncated,
                          {"status": env.status, "progress": env.progress, "reason": env.fail_reason})


def _g1_body() -> list[str]:
    from roboticsbench.robots.g1 import BODY

    return BODY
