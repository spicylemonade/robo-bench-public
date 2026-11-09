"""Panda reference kinematics from the published Franka DH table (the same numbers ROBOT.md gives agents in motor
mode) plus a small numerical IK. Used by motor-level reference solutions (oracles); agents write their own."""

from __future__ import annotations

import math

import numpy as np

DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _tf(a: float, d: float, al: float, th: float) -> np.ndarray:
    ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk(q: np.ndarray) -> np.ndarray:
    """4x4 pose of the TCP (between the fingertips) in the world frame (base at the origin)."""
    T = np.eye(4)
    for (a, d, al), qi in zip(DH, q):
        T = T @ _tf(a, d, al, qi)
    T = T @ _tf(0, 0.107, 0, 0)
    c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    T = T @ R
    Tt = np.eye(4)
    Tt[2, 3] = 0.1034
    return T @ Tt


def down_rot(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """TCP orientation pointing down, fingers closing along (-sin yaw, cos yaw); optional tilts about the TCP's own
    y (pitch) and x (roll) axes - the same convention as the skill API's move_to."""
    Rx = np.diag([1.0, -1.0, -1.0])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ Rx
    if pitch:
        cp, sp = math.cos(pitch), math.sin(pitch)
        R = R @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        R = R @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return R


def _rot_err(R_t: np.ndarray, R: np.ndarray) -> np.ndarray:
    E = R_t @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def ik(p: np.ndarray, R_t: np.ndarray | None, q0: np.ndarray, iters: int = 300, rot_w: float = 0.7,
       posture: np.ndarray | None = HOME, tol: float = 2e-4) -> tuple[np.ndarray, float]:
    """Damped least squares from q0. Returns (q, position error in metres)."""
    q = np.array(q0, float)
    err = 1.0
    for _ in range(iters):
        T = fk(q)
        e_p = np.asarray(p, float) - T[:3, 3]
        e = e_p if R_t is None else np.concatenate([e_p, rot_w * _rot_err(R_t, T[:3, :3])])
        err = float(np.linalg.norm(e_p))
        if err < tol and (R_t is None or np.linalg.norm(e[3:]) < 2e-3):
            break
        J = np.zeros((len(e), 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = 1e-6
            T2 = fk(q + dq)
            J[:3, i] = (T2[:3, 3] - T[:3, 3]) / 1e-6
            if R_t is not None:
                J[3:, i] = rot_w * _rot_err(T2[:3, :3], T[:3, :3]) / 1e-6
        step = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
        if posture is not None:
            step += 0.002 * (posture - q)
        q = np.clip(q + np.clip(step, -0.2, 0.2), Q_LO + 0.01, Q_HI - 0.01)
    if posture is not None and err > tol:  # polish without the posture bias
        return ik(p, R_t, q, iters=60, rot_w=rot_w, posture=None, tol=tol)
    return q, err
