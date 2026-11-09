"""The robot datasheet given to motor-control agents (ROBOT.md / robot_spec.json) must match the real interface."""

import json
import os
import subprocess

import numpy as np
import pytest

from roboticsbench.core.env import Env
from roboticsbench.core.motor import MotorInterface


def _flat_keys(d: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flat_keys(v, f"{prefix}{k}.")
        else:
            out.add(prefix + k)
    return out


@pytest.mark.parametrize("task", ["humanoid/ninja_course", "arm/balance_scale"])
def test_spec_matches_sensors_and_joints(task):
    env = Env(task, 0, control="motor")
    mi = MotorInterface(env, "motor")
    spec = mi.spec()
    try:
        documented = {s["key"] for s in spec["sensors"]}
        assert documented == _flat_keys(mi.sensors()), "sensor table out of sync with MotorInterface.sensors()"
        assert [j["name"] for j in spec["joints"]] == mi.joint_names
        for j in spec["joints"]:
            assert abs(np.linalg.norm(j["axis"]) - 1.0) < 1e-2
        json.dumps(spec)  # serialisable as robot_spec.json
    finally:
        env.close()


def test_sign_conventions():
    env = Env("humanoid/ninja_course", 0, control="motor")
    g1 = {j["name"]: j for j in MotorInterface(env, "motor").spec()["joints"]}
    env.close()
    assert g1["left_knee_joint"]["plus_0p1rad_cm"][2] > 0      # + bends the knee: the foot comes up
    assert g1["left_hip_pitch_joint"]["plus_0p1rad_cm"][0] < 0  # + swings the leg back
    assert g1["left_hip_roll_joint"]["plus_0p1rad_cm"][1] > 0   # + moves the left leg outwards (left)
    assert g1["waist_pitch_joint"]["plus_0p1rad_cm"][0] > 0     # + leans forward
    env = Env("arm/balance_scale", 0, control="motor")
    panda = {j["name"]: j for j in MotorInterface(env, "motor").spec()["joints"]}
    env.close()
    assert panda["joint1"]["plus_0p1rad_cm"][1] > 0             # + turns the arm to the left


def test_workspace_python_launcher_and_spec(tmp_path):
    from roboticsbench.harness.workspace import make_workspace

    info = {"goal_md": "g", "manual_md": "m", "motor_spec": {"robot": "g1"}, "robot_model_xml": "<mujoco/>"}
    ws = make_workspace(tmp_path / "ws", info, "http://127.0.0.1:1", python="python3")
    assert json.loads((ws / "robot_spec.json").read_text()) == {"robot": "g1"}
    assert os.access(ws / "python", os.X_OK)
    out = subprocess.run([str(ws / "python"), "-c", "print(6 * 7)"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "42"


def test_humanoid_agents_see_only_the_head_camera(tmp_path):
    """The humanoid perceives the world only through its own head camera: the tasks' chase / map / drone views are
    for the video and the live view, never for the agent (Env.obs_cameras, the referee's look, ROBOT.md, GOAL.md)."""
    import re

    from roboticsbench.harness.referee import Referee
    from roboticsbench.tasks import TASKS

    external = re.compile(r"'(overview|bird|chase|lane|pins|goal|throw|side)'|overview camera|chase camera|bird camera")
    for name in sorted(t for t in TASKS if t.startswith("humanoid/")):
        env = Env(name, 0, control="motor")
        try:
            assert env.obs_cameras == ("head",), name
            for vision in (False, True):
                assert not external.findall(env.task.instruction_for(vision)), name
        finally:
            env.close()
    ref = Referee("humanoid/ninja_course", 0, tmp_path / "ep", vision_only=True, control="motor")
    try:
        assert ref.info()["cameras"] == ["head"]
        assert not ref.look(["chase"])["ok"] and not ref.look(["lane"])["ok"]
        assert set(ref.look(width=64, height=48)["images"]) == {"head"}
        assert [c["name"] for c in ref.info()["motor_spec"]["cameras"]] == ["head"]
    finally:
        ref.finalize()
        ref.env.close()
