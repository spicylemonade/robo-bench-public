"""Damped-least-squares inverse kinematics on a scratch MjData (never touches the live sim)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from roboticsbench.core.geometry import mat_to_quat, rotvec_error


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float
    converged: bool


class SiteIK:
    """IK for one site over a subset of hinge/slide joints.

    ``posture`` (optional) is a nullspace attractor, ``rot_weight`` trades orientation vs position.
    """

    def __init__(self, model: mujoco.MjModel, site: str, joints: list[str], *, posture: np.ndarray | None = None,
                 damping: float = 1e-4, rot_weight: float = 0.5, posture_gain: float = 0.05):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        ids = [model.joint(j).id for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in ids])
        self.lo = model.jnt_range[ids, 0].copy()
        self.hi = model.jnt_range[ids, 1].copy()
        margin = 0.02
        self.lo += margin
        self.hi -= margin
        self.posture = None if posture is None else np.asarray(posture, float)
        self.damping = damping
        self.rot_weight = rot_weight
        self.posture_gain = posture_gain
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def site_pose(self, qpos_full: np.ndarray, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = self.d
        d.qpos[:] = qpos_full
        if q is not None:
            d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.site].copy(), mat_to_quat(d.site_xmat[self.site])

    def solve(self, qpos_full: np.ndarray, pos: np.ndarray, quat: np.ndarray | None = None, *,
              q_init: np.ndarray | None = None, iters: int = 30, tol_pos: float = 5e-4,
              tol_rot: float = 5e-3, max_step: float = 0.25) -> IKResult:
        m, d = self.m, self.d
        d.qpos[:] = qpos_full
        q = (np.asarray(q_init, float) if q_init is not None else qpos_full[self.qadr]).copy()
        pe = re = np.inf
        n = len(q)
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            perr = pos - d.site_xpos[self.site]
            pe = float(np.linalg.norm(perr))
            if quat is not None:
                rerr = rotvec_error(quat, mat_to_quat(d.site_xmat[self.site]))
                re = float(np.linalg.norm(rerr))
            else:
                rerr = None
                re = 0.0
            if pe < tol_pos and re < tol_rot:
                break
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
            if rerr is not None:
                J = np.vstack([self._jp[:, self.dadr], self.rot_weight * self._jr[:, self.dadr]])
                e = np.concatenate([perr, self.rot_weight * rerr])
            else:
                J = self._jp[:, self.dadr]
                e = perr
            JJt = J @ J.T
            lam = self.damping + 1e-3 * float(np.dot(e, e))
            dq = J.T @ np.linalg.solve(JJt + lam * np.eye(J.shape[0]), e)
            if self.posture is not None:
                Jpinv = J.T @ np.linalg.inv(JJt + 1e-6 * np.eye(J.shape[0]))
                N = np.eye(n) - Jpinv @ J
                dq += N @ (self.posture_gain * (self.posture - q))
            nrm = float(np.max(np.abs(dq)))
            if nrm > max_step:
                dq *= max_step / nrm
            q = np.clip(q + dq, self.lo, self.hi)
        return IKResult(q=q, pos_err=pe, rot_err=re, converged=bool(pe < 5 * tol_pos + 2e-3 and re < 0.05 + 10 * tol_rot))
