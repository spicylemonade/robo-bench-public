"""Small rotation helpers. Quaternions are MuJoCo-style (w, x, y, z)."""

from __future__ import annotations

import math

import mujoco
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, then pitch, then roll)."""
    return quat_mul(quat_mul(quat_yaw(yaw), quat_axis_angle([0, 1, 0], pitch)), quat_axis_angle([1, 0, 0], roll))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, float).reshape(9))
    return q


def yaw_of_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_of_mat(m: np.ndarray) -> float:
    m = np.asarray(m).reshape(3, 3)
    return math.atan2(m[1, 0], m[0, 0])


def rotvec_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking q_current to q_target."""
    qe = quat_mul(q_target, quat_conj(q_current))
    if qe[0] < 0:
        qe = -qe
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def gravity_in_body(q: np.ndarray) -> np.ndarray:
    """Projection of world gravity direction (0,0,-1) into the body frame of orientation q."""
    w, x, y, z = q
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """MuJoCo camera ``xyaxes`` for a camera at ``pos`` looking at ``target``."""
    pos = np.asarray(pos, float)
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    return [*right.tolist(), *up2.tolist()]


def smoothstep(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)
