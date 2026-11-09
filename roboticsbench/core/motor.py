"""Motor-level control: the robot interface for agents that must do everything themselves.

No skills, no inverse kinematics, no walking controller - only what the robot's own low-level interface offers:

* **Panda** (like libfranka joint position control): 7 joint position targets, tracked by the robot's joint servos
  with gravity compensation. Target changes are rate-limited to the Panda's maximum joint speeds (Franka's motion
  generator does the same). The gripper motor drives the finger opening to a width with a force limit. A collision
  reflex (protective stop) halts the arm when a link (not the fingers) hits something with > 45 N.
* **G1** (like unitree_sdk2 low-level commands): per motor q / dq / kp / kd / tau_ff for all 29 body joints,
  tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff, clipped by the motor torque limits; 14 hand joint position
  targets. The episode starts with the motors holding a crouched stance with stiff gains (the robot stands still);
  anything else - balancing, stepping, walking - is up to the agent.
* **G1 "velocity" mode** (optional, easier): the legs run the pretrained GR00T walking policy commanded by body
  velocity + pelvis height; arms and hands are still commanded per joint.

The interface is exchanged as JSON (see :meth:`MotorInterface.spec`), stepped at the 50 Hz control rate.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from roboticsbench.core.jsonutil import jsonable

CONTROL_MODES = ("skills", "motor", "velocity")


def _vec(v: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape[0] != n:
        raise ValueError(f"{name} must have {n} values, got {a.shape[0]}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


class MotorInterface:
    """Validates commands, applies them to the robot and reads its sensors. One per episode."""

    def __init__(self, env, mode: str = "motor"):
        if mode not in ("motor", "velocity"):
            raise ValueError(f"unknown motor control mode {mode!r}")
        self.env = env
        self.robot = env.robot
        self.mode = mode
        self.kind = env.robot.name
        if self.kind == "panda" and mode != "motor":
            raise ValueError("the Panda only has the 'motor' mode")
        self.events: list[str] = []
        self.command_log: list[dict[str, Any]] = []  # (tick, command) for replay
        m = env.model
        if self.kind == "panda":
            from roboticsbench.robots.panda import ARM_JOINTS
            from roboticsbench.skills.arm import ArmSkills

            self._reflex = ArmSkills(env)  # reused for its protective-stop contact check
            self.robot.q_goal = self.robot.q_target.copy()
            self.robot.control_mode = "motor"
            self.joint_names = list(ARM_JOINTS)
            ids = [m.joint(j).id for j in ARM_JOINTS]
            self.q_lo, self.q_hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
            self.stopped = False
            env.tick_callbacks.append(self._panda_reflex)
        else:
            from roboticsbench.robots import g1 as G

            self.G = G
            self.joint_names = list(G.BODY)
            self.hand_names = list(G.LEFT_HAND + G.RIGHT_HAND)
            ids = [m.joint(j).id for j in G.BODY]
            self.q_lo, self.q_hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
            self.tau_max = m.jnt_actfrcrange[ids, 1].copy()
            hids = [m.joint(j).id for j in self.hand_names]
            self.hand_lo, self.hand_hi = m.jnt_range[hids, 0], m.jnt_range[hids, 1]
            self.hand_qadr = np.array([m.jnt_qposadr[j] for j in hids])
            self.hand_vadr = np.array([m.jnt_dofadr[j] for j in hids])
            self.kp_max = np.maximum(3.0 * G.BODY_KP, 1.5 * G.HOLD_KP)
            self.kd_max = np.maximum(4.0 * G.BODY_KD, 2.0 * G.HOLD_KD)
            self.robot.hand_direct = True
            if mode == "motor":
                self.robot.mode = "joint"
            else:
                self.robot.mode = "wbc"
            self._sens = {n: (m.sensor(n).adr[0], m.sensor(n).dim[0]) for n in
                          ("imu-pelvis-angular-velocity", "imu-pelvis-linear-acceleration",
                           "imu-torso-angular-velocity", "imu-torso-linear-acceleration")}
            self._torso = m.body("torso_link").id
            self._pelvis = m.body("pelvis").id
            self._imu_sites = (m.site("imu_in_pelvis").id, m.site("imu_in_torso").id)
            self._foot_sets = []
            for side in ("left", "right"):
                b = m.body(f"{side}_ankle_roll_link").id
                self._foot_sets.append({g for g in range(m.ngeom) if m.geom_bodyid[g] == b})

    # ------------------------------------------------------------------ commands
    def apply(self, cmd: dict[str, Any]) -> None:
        """Validate and apply a command (persistent until changed). Raises ValueError on bad input."""
        if not isinstance(cmd, dict):
            raise ValueError("command must be a JSON object")
        cmd = dict(cmd)
        if self.kind == "panda":
            self._apply_panda(cmd)
        elif self.mode == "motor":
            self._apply_g1_motor(cmd)
        else:
            self._apply_g1_velocity(cmd)
        self.command_log.append({"tick": int(self.env.ticks), "command": jsonable(cmd)})

    def _unknown(self, cmd: dict[str, Any], allowed: set[str]) -> None:
        bad = sorted(set(cmd) - allowed)
        if bad:
            raise ValueError(f"unknown command field(s) {bad}; allowed: {sorted(allowed)}")

    def _apply_panda(self, cmd: dict[str, Any]) -> None:
        self._unknown(cmd, {"joint_positions", "gripper"})
        r = self.robot
        if "joint_positions" in cmd:
            q = _vec(cmd["joint_positions"], 7, "joint_positions")
            r.q_goal = np.clip(q, self.q_lo + 0.005, self.q_hi - 0.005)
            self.stopped = False
        if "gripper" in cmd:
            g = cmd["gripper"]
            if not isinstance(g, dict) or "width" not in g:
                raise ValueError('gripper must be {"width": metres (0-0.08), "force": newtons (5-70, default 40)}')
            from roboticsbench.robots.panda import GRIP_FORCE_MAX, MAX_OPENING

            width = float(g["width"])
            force = float(g.get("force", 40.0))
            if not (math.isfinite(width) and math.isfinite(force)):
                raise ValueError("gripper width/force must be finite")
            r.grip_closed = False
            r.grip_width_target = float(np.clip(width, 0.0, MAX_OPENING))
            r.grip_force_open = float(np.clip(force, 5.0, GRIP_FORCE_MAX))

    def _apply_g1_motor(self, cmd: dict[str, Any]) -> None:
        self._unknown(cmd, {"q", "dq", "kp", "kd", "tau", "joints", "hand_q"})
        r = self.robot
        q, dq = r.joint_target.copy(), r.joint_dq_target.copy()
        kp, kd, tau = r.joint_kp.copy(), r.joint_kd.copy(), r.joint_tau_ff.copy()
        for key, arr in (("q", q), ("dq", dq), ("kp", kp), ("kd", kd), ("tau", tau)):
            if key in cmd:
                arr[:] = _vec(cmd[key], 29, key)
        for name, spec in (cmd.get("joints") or {}).items():
            if name not in self.joint_names:
                raise ValueError(f"unknown joint {name!r}")
            if not isinstance(spec, dict):
                raise ValueError(f'joints[{name!r}] must be an object like {{"q": 0.1, "kp": 100}}')
            i = self.joint_names.index(name)
            for key, arr in (("q", q), ("dq", dq), ("kp", kp), ("kd", kd), ("tau", tau)):
                if key in spec:
                    v = float(spec[key])
                    if not math.isfinite(v):
                        raise ValueError(f"joints[{name!r}].{key} must be finite")
                    arr[i] = v
            unknown = set(spec) - {"q", "dq", "kp", "kd", "tau"}
            if unknown:
                raise ValueError(f"joints[{name!r}]: unknown field(s) {sorted(unknown)}")
        r.joint_target = np.clip(q, self.q_lo, self.q_hi)
        r.joint_dq_target = np.clip(dq, -30.0, 30.0)
        r.joint_kp = np.clip(kp, 0.0, self.kp_max)
        r.joint_kd = np.clip(kd, 0.0, self.kd_max)
        r.joint_tau_ff = np.clip(tau, -self.tau_max, self.tau_max)
        if "hand_q" in cmd:
            r.hand_target = np.clip(_vec(cmd["hand_q"], 14, "hand_q"), self.hand_lo, self.hand_hi)

    def _apply_g1_velocity(self, cmd: dict[str, Any]) -> None:
        self._unknown(cmd, {"base_velocity", "pelvis_height", "arm_q", "arm_speed", "hand_q"})
        r = self.robot
        if "arm_speed" in cmd:  # slew limit of the arm joint targets, rad/s (default 2.5; up to 10 for throwing)
            from roboticsbench.robots.g1 import ARM_SPEED_MAX

            r.arm_max_vel = float(np.clip(float(cmd["arm_speed"]), 0.2, ARM_SPEED_MAX))
        if "base_velocity" in cmd:
            r.vel_cmd[:] = _vec(cmd["base_velocity"], 3, "base_velocity")
        if "pelvis_height" in cmd:
            r.height_cmd = float(np.clip(float(cmd["pelvis_height"]), 0.55, 0.78))
        if "arm_q" in cmd:
            r.arm_target[:] = np.clip(_vec(cmd["arm_q"], 14, "arm_q"), self.q_lo[15:], self.q_hi[15:])
        if "hand_q" in cmd:
            r.hand_target = np.clip(_vec(cmd["hand_q"], 14, "hand_q"), self.hand_lo, self.hand_hi)

    # ------------------------------------------------------------------ reflex
    WRIST_FORCE_STOP = 80.0  # N: external force on the hand that triggers the reflex (e.g. jamming a held part)

    def _panda_reflex(self, env) -> None:
        if self.stopped:
            return
        hit = self._reflex._collision()
        if not hit:
            f = float(np.linalg.norm(self.robot.wrist_ft()[:3]))
            if f > self.WRIST_FORCE_STOP:
                hit = f"external force on the hand {f:.0f} N"
        if hit:
            r = self.robot
            r.q_goal = env.data.qpos[r.q_adr].copy()
            r.q_target = r.q_goal.copy()
            self.stopped = True
            self.events.append(f"t={env.t:.2f}s protective stop: {hit}; the arm halted - send new joint targets")

    # ------------------------------------------------------------------ sensors
    def sensors(self) -> dict[str, Any]:
        env, d, r = self.env, self.env.data, self.robot
        out: dict[str, Any] = {"time": round(env.t, 3), "tick": int(env.ticks)}
        if self.kind == "panda":
            out.update({
                "joint_positions": d.qpos[r.q_adr],
                "joint_velocities": d.qvel[r.v_adr],
                "joint_torques": d.actuator_force[r.arm_act],
                "gripper_width": r.finger_width(),
                "gripper_force": abs(float(d.actuator_force[r.grip_act])),
                "wrist_force_torque": r.wrist_ft(),
                "protective_stop": self.stopped,
            })
        else:
            out.update({
                "imu_pelvis": self._imu(0, self._pelvis, "imu-pelvis"),
                "imu_torso": self._imu(1, self._torso, "imu-torso"),
                "joint_positions": d.qpos[r.q_adr],
                "joint_velocities": d.qvel[r.v_adr],
                "joint_torques": d.qfrc_actuator[r.v_adr],
                "hand_positions": d.qpos[self.hand_qadr],
                "hand_velocities": d.qvel[self.hand_vadr],
                "foot_contact_forces": self._foot_forces(),
                "base_position": r.base_pos,
                "base_velocity": r.base_vel_world,
            })
        return jsonable(_round(out))

    def _imu(self, k: int, body: int, prefix: str) -> dict[str, Any]:
        d = self.env.data
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[self._imu_sites[k]])
        gyro = d.sensordata[slice(*self._span(f"{prefix}-angular-velocity"))]
        acc = d.sensordata[slice(*self._span(f"{prefix}-linear-acceleration"))]
        return {"quaternion": q, "angular_velocity": gyro, "linear_acceleration": acc}

    def _span(self, name: str) -> tuple[int, int]:
        a, n = self._sens[name]
        return a, a + n

    def _foot_forces(self) -> list[float]:
        m, d = self.env.model, self.env.data
        out = [0.0, 0.0]
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for k, feet in enumerate(self._foot_sets):
                if (c.geom1 in feet) != (c.geom2 in feet):
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[k] += abs(float(f6[0]))
        return out

    # ------------------------------------------------------------------ description
    def spec(self) -> dict[str, Any]:
        """The robot's datasheet (commands, sensors, joints with axes and sign conventions, limits, gains, masses,
        dimensions, cameras) - see :mod:`roboticsbench.core.robot_spec`. Rendered into ROBOT.md and saved as
        ``robot_spec.json`` in the agent's workspace."""
        from roboticsbench.core.robot_spec import build_spec

        return jsonable(build_spec(self.env, self))


def _round(o: Any, nd: int = 5) -> Any:
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [round(float(x), nd) for x in o.reshape(-1)]
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), nd)
    return o


_MODEL_XML: dict[str, str] = {}


def robot_model_xml(robot_name: str) -> str:
    """The robot's own MuJoCo model without meshes (bodies, joints, axes, limits, masses, inertias, actuators) -
    the equivalent of the manufacturer's URDF that a roboticist would have. Contains nothing about the task."""
    if robot_name not in _MODEL_XML:
        from roboticsbench.robots import make_robot

        spec = make_robot(robot_name).build_spec()
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(g)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
        spec.meshdir = ""
        spec.compile()
        _MODEL_XML[robot_name] = spec.to_xml()
    return _MODEL_XML[robot_name]
