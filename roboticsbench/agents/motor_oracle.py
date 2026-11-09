"""Motor-level reference-solution helper: everything here uses only the public motor interface (commands, sensors,
objects) plus the published kinematics - i.e. what a coding agent in motor mode has. Tasks whose reference solution
needs raw motor control (throwing, fast or force-sensitive motions) set ``oracle_control = "motor"`` and write
``oracle(self, api)`` against this class."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboticsbench.control import panda_kin as K
from roboticsbench.tasks.arm.oracle_utils import EpisodeOver, OracleFailure


class PandaMotorOracle:
    def __init__(self, env, motor, skills):
        self.env, self.motor, self.sk = env, motor, skills

    # -------------------------------------------------------------- primitives
    def sensors(self) -> dict[str, Any]:
        return self.motor.sensors()

    def objects(self) -> dict[str, dict[str, Any]]:
        from roboticsbench.core.jsonutil import jsonable

        return {o["name"]: o for o in jsonable(self.env.task.objects(self.env))}

    def q(self) -> np.ndarray:
        return np.array(self.sensors()["joint_positions"])

    def step(self, ticks: int = 1, command: dict | None = None) -> None:
        if not self.env.running:
            raise EpisodeOver(self.env.status)
        if command:
            self.motor.apply(command)
        if ticks:
            self.env.tick(ticks)
        if not self.env.running:
            raise EpisodeOver(self.env.status)

    def done(self) -> None:
        if self.env.running:
            self.sk.call("done", {})
        raise EpisodeOver(self.env.status)

    # -------------------------------------------------------------- motions
    def move_joints(self, q_target, seconds: float = 1.5, settle: int = 10) -> None:
        """Smooth (smoothstep) joint-space interpolation, one target per tick."""
        q0 = self.q()
        q1 = np.asarray(q_target, float)
        n = max(2, int(seconds / self.env.control_dt))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            self.step(1, {"joint_positions": (q0 + a * (q1 - q0)).tolist()})
        self.step(settle)

    def move_tcp(self, pos, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, seconds: float = 1.5,
                 tol: float = 0.004) -> np.ndarray:
        q1, err = K.ik(np.asarray(pos, float), K.down_rot(yaw, pitch, roll), self.q())
        if err > tol:
            raise OracleFailure(f"IK failed for {np.round(pos, 3).tolist()} (err {err:.3f} m)")
        self.move_joints(q1, seconds)
        return q1

    def gripper(self, width: float, force: float = 40.0, ticks: int = 30) -> None:
        self.step(ticks, {"gripper": {"width": width, "force": force}})

    def tcp(self) -> np.ndarray:
        return K.fk(self.q())[:3, 3]


class G1VelocityOracle:
    """Reference-solution helper for the G1 through the public *velocity* interface: the legs follow body-velocity
    commands (walking policy), arms and hands take joint targets. Closed-loop walking here uses only the public
    sensors (base position, IMU orientation) - what a velocity-mode agent has. Tasks set ``oracle_control =
    "velocity"`` and write ``oracle(self, api)`` against this class."""

    ARM_ZERO = [0.0] * 14

    def __init__(self, env, motor, skills):
        self.env, self.motor, self.sk = env, motor, skills
        self._cmd = {"base_velocity": [0.0, 0.0, 0.0]}

    # -------------------------------------------------------------- primitives
    def sensors(self) -> dict[str, Any]:
        return self.motor.sensors()

    def objects(self) -> dict[str, dict[str, Any]]:
        from roboticsbench.core.jsonutil import jsonable

        return {o["name"]: o for o in jsonable(self.env.task.objects(self.env))}

    def status(self) -> dict[str, Any]:
        from roboticsbench.core.jsonutil import jsonable

        return jsonable(self.env.task.status(self.env))

    def step(self, ticks: int = 1, command: dict | None = None) -> None:
        if not self.env.running:
            raise EpisodeOver(self.env.status)
        if command:
            self.motor.apply(command)
        if ticks:
            self.env.tick(ticks)
        if not self.env.running:
            raise EpisodeOver(self.env.status)

    def done(self) -> None:
        if self.env.running:
            self.sk.call("done", {})
        raise EpisodeOver(self.env.status)

    def pose(self) -> tuple[float, float, float]:
        import math

        s = self.sensors()
        w, x, y, z = s["imu_pelvis"]["quaternion"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(s["base_position"][0]), float(s["base_position"][1]), yaw

    # -------------------------------------------------------------- locomotion
    def stop(self, ticks: int = 25) -> None:
        self.step(ticks, {"base_velocity": [0.0, 0.0, 0.0]})

    def turn_to(self, heading: float, tol: float = 0.05, timeout: float = 12.0) -> None:
        import math

        t_end = self.env.t + timeout
        while self.env.t < t_end:
            err = (heading - self.pose()[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < tol:
                break
            wz = float(np.clip(1.8 * err, -0.8, 0.8))
            wz = math.copysign(max(abs(wz), 0.2), wz)
            self.step(1, {"base_velocity": [0.0, 0.0, wz]})
        self.stop(20)

    def walk_to(self, x: float, y: float, speed: float = 0.6, tol: float = 0.06, heading: float | None = None,
                timeout: float | None = None) -> None:
        """Turn toward the target, walk with heading feedback, fine-step in, optionally face ``heading``."""
        import math

        tgt = np.array([x, y], float)
        px, py, _ = self.pose()
        d0 = float(np.linalg.norm(tgt - [px, py]))
        t_end = self.env.t + (timeout or 8.0 + 2.5 * d0 / max(speed, 0.1))
        if d0 > 0.45:
            self.turn_to(math.atan2(y - py, x - px), tol=0.12)
        while self.env.t < t_end:
            px, py, yaw = self.pose()
            e = tgt - [px, py]
            dist = float(np.linalg.norm(e))
            if dist < tol:
                break
            herr = (math.atan2(e[1], e[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            ex = math.cos(yaw) * e[0] + math.sin(yaw) * e[1]
            ey = -math.sin(yaw) * e[0] + math.cos(yaw) * e[1]
            if dist > 0.45:
                cmd = [min(speed, 0.9 * dist) * max(math.cos(herr), 0.0), float(np.clip(0.6 * ey, -0.12, 0.12)),
                       float(np.clip(1.6 * herr, -0.8, 0.8))]
            else:
                vx, vy = float(np.clip(ex, -0.25, 0.35)), float(np.clip(ey, -0.25, 0.25))
                n = math.hypot(vx, vy)
                if n < 0.08:
                    vx, vy = vx * 0.08 / max(n, 1e-6), vy * 0.08 / max(n, 1e-6)
                cmd = [vx, vy, 0.0]
            self.step(1, {"base_velocity": cmd})
        self.stop(20)
        if heading is not None:
            self.turn_to(heading)

    # -------------------------------------------------------------- arms / hands / body
    def arms(self, q14, seconds: float = 1.0) -> None:
        q0 = np.array(self.sensors()["joint_positions"][15:29])
        q1 = np.asarray(q14, float)
        n = max(1, int(seconds / self.env.control_dt))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            self.step(1, {"arm_q": (q0 + a * (q1 - q0)).tolist()})

    def hands(self, q14, ticks: int = 20) -> None:
        self.step(ticks, {"hand_q": list(q14)})

    def height(self, h: float, ticks: int = 60) -> None:
        self.step(ticks, {"pelvis_height": float(h)})
