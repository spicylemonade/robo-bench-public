"""A Python environment for the agent's own scripts in local runs, matching what the hosted runner's agent image offers
(numpy, scipy, mujoco, pillow) but WITHOUT the roboticsbench package (so the agent cannot import the simulator).

Created once with ``uv`` under ``~/.cache/roboticsbench/agent-venv``; its ``bin`` goes first on the agent's PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PACKAGES = ["numpy", "scipy", "mujoco", "pillow"]
VENV = Path(os.environ.get("RBENCH_AGENT_VENV", Path.home() / ".cache" / "roboticsbench" / "agent-venv"))


def agent_python_bin() -> str | None:
    """Directory to prepend to the agent's PATH, or None if the environment cannot be created (no uv)."""
    py = VENV / "bin" / "python"
    marker = VENV / ".rbench-ready"
    if marker.exists() and py.exists():
        return str(VENV / "bin")
    uv = shutil.which("uv")
    if not uv:
        return None
    try:
        VENV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([uv, "venv", "--quiet", "--python", "3.12", str(VENV)], check=True, timeout=600)
        subprocess.run([uv, "pip", "install", "--quiet", "--python", str(py), *PACKAGES], check=True, timeout=1200)
        marker.write_text("ok\n")
    except (subprocess.SubprocessError, OSError):
        return None
    return str(VENV / "bin")
