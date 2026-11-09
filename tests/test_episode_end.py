"""Humanoid episodes never end early: the agent keeps the full time after a fall or a broken rule (2026-10-09). Arm tasks
still end at once on a broken rule."""

from roboticsbench.core.env import Env
from roboticsbench.core.motor import MotorInterface
from roboticsbench.tasks.base import Evaluation


def _run_out(env):
    while env.tick(1):
        pass


def test_humanoid_fall_does_not_end_the_episode():
    env = Env("humanoid/ninja_course", seed=0, control="motor")
    try:
        env.task.time_limit = 4.0
        MotorInterface(env, "motor").apply({"kp": [0.0] * 29, "kd": [0.0] * 29})  # limp: the robot collapses
        _run_out(env)
        res = env.result()
        assert env.robot.is_fallen() and res["falls"] >= 1
        assert res["status"] == "timeout" and res["sim_time"] >= 4.0
    finally:
        env.close()


def _break_rule_at(env, t_break):
    evaluate = env.task.evaluate

    def broken(e):
        ev = evaluate(e)
        if e.t >= t_break:
            return Evaluation(success=False, progress=ev.progress, failed=True, reason="broke a task rule")
        return ev
    env.task.evaluate = broken


def test_humanoid_broken_rule_runs_to_the_time_limit_and_reports_the_rule():
    env = Env("humanoid/ninja_course", seed=0, control="motor")
    try:
        env.task.time_limit = 2.0
        _break_rule_at(env, 0.5)
        env.run_for(1.0)
        assert env.running  # the rule broke at 0.5 s: the clock keeps going
        _run_out(env)
        res = env.result()
        assert res["status"] == "failure" and res["reason"] == "broke a task rule" and res["sim_time"] >= 2.0
    finally:
        env.close()


def test_arm_broken_rule_ends_the_episode_at_once():
    env = Env("arm/balance_scale", seed=0, control="motor")
    try:
        env.task.time_limit = 10.0
        _break_rule_at(env, 0.5)
        _run_out(env)
        res = env.result()
        assert res["status"] == "failure" and res["sim_time"] < 0.6
    finally:
        env.close()
