"""The coding agent's working directory: GOAL.md, ROBOT.md, the live robot module robot.py, the ./robot CLI,
robot_client.py, robot_spec.json, the ./python launcher and views/."""

from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path
from typing import Any

from roboticsbench.harness import robot_cli, robot_live

AGENT_PROMPT = ("You are operating a robot in a physics simulation. The task is described in GOAL.md; ROBOT.md "
                "documents the robot and how you must control it (the ./robot command and robot_client.py). Read "
                "both files, then complete the task. Work autonomously - nobody will answer questions. Look at the "
                "camera images (saved under views/) and check the robot's state after every action. When the task "
                "is complete, run ./robot done.")


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_workspace(ws: str | os.PathLike, info: dict[str, Any], api_url: str, python: str = "python3") -> Path:
    """``python``: the interpreter (with numpy, scipy, mujoco, pillow) that ``./python`` runs - an explicit launcher
    because agent harnesses often run commands in login shells that put the system Python first on PATH."""
    ws = Path(ws)
    (ws / "views").mkdir(parents=True, exist_ok=True)
    (ws / "GOAL.md").write_text(info["goal_md"])
    (ws / "ROBOT.md").write_text(info["manual_md"])
    src = Path(robot_cli.__file__).read_text()
    _executable(ws / "robot", src)
    (ws / "robot_client.py").write_text(src)
    (ws / "robot.py").write_text(Path(robot_live.__file__).read_text())  # from robot import robot: live senses
    (ws / ".robot_api").write_text(api_url + "\n")
    _executable(ws / "python", '#!/bin/sh\nHERE=$(cd "$(dirname "$0")" && pwd)\n'  # `import robot` from anywhere
                f'PYTHONPATH="$HERE${{PYTHONPATH:+:$PYTHONPATH}}" exec {shlex.quote(python)} "$@"\n')
    if info.get("robot_model_xml"):  # motor control: the robot's own model (like the manufacturer's URDF)
        (ws / "robot_model.xml").write_text(info["robot_model_xml"])
    if info.get("motor_spec"):  # ... and its datasheet (everything ROBOT.md says about the robot, as JSON)
        (ws / "robot_spec.json").write_text(json.dumps(info["motor_spec"], indent=1))
    import base64

    for rel, b64 in (info.get("documents") or {}).items():  # task documents, e.g. an assembly manual
        path = (ws / rel).resolve()
        if ws.resolve() not in path.parents:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(b64))
    return ws
