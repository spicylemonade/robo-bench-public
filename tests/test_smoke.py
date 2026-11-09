"""Fast structural checks for every task + the agent interfaces (the oracles are exercised by `rbench validate`)."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills
from roboticsbench.tasks import LOAD_ERRORS, list_tasks

TASKS = list_tasks(include_shelved=True)  # shelved tasks stay working (back burner, not deleted)


def test_all_task_modules_load():
    assert not LOAD_ERRORS, "\n".join(f"{k}:\n{v}" for k, v in LOAD_ERRORS.items())
    assert TASKS == ["arm/balance_scale", "arm/lego_manual", "humanoid/ninja_course"]


def test_benchmark_set():
    from roboticsbench.eval.suites import SUITES
    from roboticsbench.tasks.base import SHELVED

    assert list_tasks() == SUITES["public"] and not SHELVED


@pytest.mark.parametrize("task", TASKS)
def test_task_builds_and_is_well_formed(task):
    env = Env(task, seed=0)
    try:
        t = env.task
        assert t.instruction and len(t.instruction) > 20
        assert t.category and t.difficulty in ("easy", "medium", "hard", "expert")
        assert 0 < t.time_limit <= 900 and 0 < t.max_actions <= 500
        assert env.running and env.progress < 1.0
        sk0 = make_skills(env)
        if t.allow_object_list:
            assert sk0.list_objects().ok
        json.dumps(env.observe(images=False))  # agents receive task status + proprio through this
        json.dumps(env.result())
        ev = t.evaluate(env)
        assert not ev.success and not ev.failed and ev.progress < 1.0
        for cam in env.obs_cameras:
            img = env.render(cam, 160, 120)
            assert img.shape == (120, 160, 3) and img.mean() > 5
        sk = make_skills(env)
        names = [tool["name"] for tool in sk.tools()]
        assert "done" in names and ("list_objects" in names) == t.allow_object_list
        for tool in sk.tools():
            assert tool["parameters"]["type"] == "object" and tool["description"]
        env.tick(10)
        assert all(math.isfinite(x) for x in env.data.qpos)
    finally:
        env.close()


@pytest.mark.parametrize("task", TASKS[:3])
def test_layout_is_deterministic_per_seed(task):
    a, b = Env(task, seed=3), Env(task, seed=3)
    try:
        assert a.task.instruction == b.task.instruction
        assert json.dumps(a.task.objects(a)) == json.dumps(b.task.objects(b))
    finally:
        a.close()
        b.close()


def test_robot_cli_value_parsing():
    from roboticsbench.harness.robot_cli import _value

    assert _value("1.5") == 1.5 and _value("-2") == -2 and _value("true") is True
    assert _value("left") == "left" and _value("[1, 2]") == [1, 2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _req(method, url, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.loads(f.read())


def test_referee_roundtrip(tmp_path):
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    p = subprocess.Popen([sys.executable, "-m", "roboticsbench.harness.referee", "--task", "arm/balance_scale",
                          "--seed", "1", "--port", str(port), "--ledger", str(tmp_path), "--control", "skills"],
                         env={**os.environ, "RBENCH_OPERATOR_TOKEN": "tok"})
    try:
        for _ in range(300):
            try:
                _req("GET", url + "/health")
                break
            except OSError:
                time.sleep(0.1)
        info = _req("GET", url + "/info")
        assert "./robot" in info["manual_md"] and info["tools"]
        bad = _req("POST", url + "/act", {"skill": "move_to", "args": {"nope": 2}})
        assert not bad["result"]["ok"] and bad["state"]["actions_used"] == 0  # invalid calls are free
        look = _req("POST", url + "/look", {"width": 64, "height": 48})
        assert look["ok"] and look["images"]
        with pytest.raises(urllib.error.HTTPError):
            _req("POST", url + "/operator/finalize", {})  # no token
        res = _req("POST", url + "/operator/finalize", {}, headers={"X-Operator-Token": "tok"})
        assert not res["success"] and res["score"] < 1.0 and res["n_actions"] == 0
        assert (tmp_path / "referee_result.json").exists() and (tmp_path / "referee_trace.json").exists()
        _req("POST", url + "/operator/shutdown", {}, headers={"X-Operator-Token": "tok"})
    finally:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def _start_referee(tmp_path, task, control):
    port = _free_port()
    p = subprocess.Popen([sys.executable, "-m", "roboticsbench.harness.referee", "--task", task, "--seed", "0",
                          "--port", str(port), "--ledger", str(tmp_path), "--control", control],
                         env={**os.environ, "RBENCH_OPERATOR_TOKEN": "tok"})
    url = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            _req("GET", url + "/health")
            break
        except OSError:
            time.sleep(0.1)
    return p, url


@pytest.mark.parametrize("task,control", [("arm/balance_scale", "motor"), ("humanoid/ninja_course", "motor"),
                                          ("humanoid/ninja_course", "velocity")])
def test_motor_referee(tmp_path, task, control):
    p, url = _start_referee(tmp_path, task, control)
    try:
        info = _req("GET", url + "/info")
        assert info["control"] == control and info["motor_spec"]["joints"] and "<mujoco" in info["robot_model_xml"]
        assert "walk_to" not in info["manual_md"].split("## Commands")[1].split("## Joints")[0] or control != "motor"
        assert not _req("POST", url + "/act", {"skill": "wait", "args": {"seconds": 1}})["result"]["ok"]
        assert _req("POST", url + "/objects", {})["ok"]
        bad = _req("POST", url + "/step", {"command": {"nope": 1}, "ticks": 1})
        assert not bad["ok"] and "unknown command field" in bad["message"]
        if info["robot"] == "panda":
            cmd = {"joint_positions": [0.3, -0.785, 0, -2.356, 0, 1.571, 0.785], "gripper": {"width": 0.02}}
        elif control == "motor":
            cmd = {"joints": {"waist_yaw_joint": {"q": 0.05}}}
        else:
            cmd = {"base_velocity": [0.3, 0, 0]}
        rep = _req("POST", url + "/step", {"command": cmd, "ticks": 50})
        assert rep["ok"] and rep["episode_status"] == "running" and rep["sensors"]["time"] > 0.9, rep.get("message") or rep.get("episode_end_reason")
        res = _req("POST", url + "/operator/finalize", {}, headers={"X-Operator-Token": "tok"})
        assert res["control"] == control and res["n_motor_commands"] == 1
        from roboticsbench.eval.verify import verify

        assert verify(tmp_path)["ok"]
        _req("POST", url + "/operator/shutdown", {}, headers={"X-Operator-Token": "tok"})
    finally:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def test_g1_motor_stance_is_stable():
    env = Env("humanoid/ninja_course", seed=2, control="motor")
    try:
        env.run_for(20.0)
        assert env.running and env.robot.tilt_deg() < 10
    finally:
        env.close()
