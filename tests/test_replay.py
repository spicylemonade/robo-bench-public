"""Videos of real-time sandbox episodes are re-rendered from the trace: the replay must reproduce the episode."""

import json

from roboticsbench.core.env import Env
from roboticsbench.core.motor import MotorInterface
from roboticsbench.harness.replay import replay_video


def test_replay_reproduces_a_motor_episode(tmp_path):
    env = Env("arm/balance_scale", seed=0, control="motor")
    motor = MotorInterface(env, "motor")
    q0 = list(env.robot.q_goal)
    for k, dq in enumerate((0.3, -0.2, 0.15)):  # a few commands at irregular ticks, like a real agent
        env.tick(17 + 9 * k)
        q = list(q0)
        q[0] += dq
        q[3] += dq / 2
        motor.apply({"joint_positions": q, "gripper": {"width": 0.04 * k, "force": 20}})
    env.tick(40)
    trace = {"task": "arm/balance_scale", "seed": 0, "control": "motor", "final_tick": int(env.ticks),
             "end_tick": None, "motor_commands": motor.command_log, "result": env.result()}
    qpos = env.data.qpos.copy()
    env.close()
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace, default=float))
    rep = replay_video(path, tmp_path / "v.mp4", fps=4, size=(160, 120))
    assert rep["commands_applied"] == 3 and rep["ticks"] == trace["final_tick"] and rep["match"]
    assert (tmp_path / "v.mp4").stat().st_size > 1000
    # and the state itself, not only the progress figure
    env2 = Env("arm/balance_scale", seed=0, control="motor")
    m2 = MotorInterface(env2, "motor")
    cmds = iter(motor.command_log)
    nxt = next(cmds, None)
    while env2.ticks < trace["final_tick"]:
        while nxt is not None and nxt["tick"] <= env2.ticks:
            m2.apply(nxt["command"])
            nxt = next(cmds, None)
        env2.tick(1)
    assert abs(env2.data.qpos - qpos).max() < 1e-9
    env2.close()
