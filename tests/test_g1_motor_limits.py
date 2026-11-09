"""The G1's motors follow a torque-speed curve: a full-torque, zero-gain command on a fallen robot must not spin its
joints up until the physics diverges (2026-10-09)."""

import numpy as np

from roboticsbench.core.env import Env
from roboticsbench.core.motor import MotorInterface
from roboticsbench.robots.g1 import motor_torque_limit


def test_torque_speed_curve():
    tau_max, dq_max = np.array([100.0] * 4), np.array([20.0] * 4)
    tau = np.array([100.0, 100.0, 100.0, -100.0])
    dq = np.array([5.0, 17.0, 25.0, 25.0])  # slow, fading, past no-load speed, braking past it
    out = motor_torque_limit(tau, dq, tau_max, dq_max)
    assert out[0] == 100.0 and 0.0 < out[1] < 100.0 and out[2] == 0.0 and out[3] == -100.0


def test_full_torque_thrashing_does_not_diverge():
    env = Env("humanoid/ninja_course", seed=0, control="motor")
    try:
        env.task.time_limit = 1e9
        motor = MotorInterface(env, "motor")
        motor.apply({"kp": [0.0] * 29, "kd": [0.0] * 29})
        env.run_for(3.0)  # collapse
        sign = np.where(np.arange(29) % 2, 1.0, -1.0)
        for k in range(6):  # alternate full torque on every joint, no damping
            motor.apply({"tau": (sign * (-1) ** k * motor.tau_max).tolist()})
            env.run_for(0.5)
        assert env.running, env.fail_reason
    finally:
        env.close()
