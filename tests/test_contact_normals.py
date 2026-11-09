"""A weight touching a balance pan must be pushed up, never down (2026-10-09: MuJoCo's native CCD inverted the normal of
tilted box-on-cylinder contacts, the pan pushed the weight down through itself; scenes where a cylinder can touch a
box / mesh / cylinder now collide through libccd)."""

import math

import mujoco
import numpy as np
import pytest

from roboticsbench.core.env import Env
from roboticsbench.tasks.arm.balance_scale import PAN_COL_H, W_HALF


@pytest.mark.parametrize("task,libccd", [("arm/balance_scale", True), ("humanoid/ninja_course", True),
                                          ("arm/lego_manual", False)])
def test_cylinder_scenes_collide_through_libccd(task, libccd):
    env = Env(task, seed=0, control="motor")
    try:
        assert bool(env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD) == libccd
    finally:
        env.close()


def test_tilted_weight_on_pan_is_pushed_up():
    env = Env("arm/balance_scale", seed=0, control="motor")
    m, d = env.model, env.data
    try:
        pan, box = m.geom("pan_right").id, m.geom("w200").id
        qadr = m.jnt_qposadr[m.body_jntadr[m.body("w200").id]]
        corners = np.array([[sx * W_HALF[0], sy * W_HALF[1], sz * W_HALF[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        mujoco.mj_forward(m, d)
        top = d.geom_xpos[pan][2] + PAN_COL_H
        centre = d.geom_xpos[pan][:2].copy()
        q, R = np.zeros(4), np.zeros(9)
        checked, wrong = 0, []
        # the configuration that failed (7.4 deg about a skewed axis) and its neighbourhood
        for tilt in (2.0, 4.0, 7.4, 9.0, 12.0, 20.0):
            for axis in ((1, 2, 0.3), (2, -1, 0.5), (0.3, 1, 0.8), (1, 0.2, -0.4), (1, 1, 0)):
                for off in ((0.0, 0.0), (0.01, -0.02), (-0.03, 0.02)):
                    mujoco.mju_axisAngle2Quat(q, np.array(axis, float) / np.linalg.norm(axis), math.radians(tilt))
                    mujoco.mju_quat2Mat(R, q)
                    low = (corners @ R.reshape(3, 3).T)[:, 2].min()
                    d.qpos[qadr:qadr + 3] = [centre[0] + off[0], centre[1] + off[1], top - low - 0.0008]
                    d.qpos[qadr + 3:qadr + 7] = q
                    mujoco.mj_forward(m, d)
                    for i in range(d.ncon):
                        c = d.contact[i]
                        if {int(c.geom1), int(c.geom2)} != {pan, box}:
                            continue
                        up = c.frame[2] if c.geom1 == pan else -c.frame[2]  # normal points from geom1 to geom2
                        checked += 1
                        if up < 0.5:
                            wrong.append((tilt, axis, off, round(float(up), 2)))
        assert checked >= 50
        assert not wrong, f"{len(wrong)}/{checked} pan contacts push the weight down: {wrong[:5]}"
    finally:
        env.close()
