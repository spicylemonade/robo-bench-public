"""Replay verification: re-simulate an episode from its seed and recorded skill calls and check that the physics
reproduces the recorded outcome. MuJoCo, the IK and the GR00T policy (single-threaded ONNX on CPU) are
deterministic, so any mismatch means the record was altered or the code/assets differ from the scoring run.

    rbench verify runs/<run>/<episode_dir>        (reads referee_trace.json or trace.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills


def _load_trace(path: Path) -> dict[str, Any]:
    if path.is_dir():
        for name in ("referee_trace.json", "trace.json"):
            if (path / name).exists():
                return json.loads((path / name).read_text())
        raise FileNotFoundError(f"no referee_trace.json / trace.json in {path}")
    return json.loads(path.read_text())


def replay(trace: dict[str, Any]) -> dict[str, Any]:
    rec = trace["result"]
    control = trace.get("control", "skills")
    env = Env(trace["task"], seed=int(trace["seed"]), control=control)
    sk = make_skills(env)
    if control in ("motor", "velocity"):
        from roboticsbench.core.motor import MotorInterface

        mi = MotorInterface(env, control)
        for entry in trace.get("motor_commands", []):
            if not env.running:
                break
            if entry["tick"] > env.ticks:
                env.tick(entry["tick"] - env.ticks)
            if not env.running:
                break
            if entry.get("done"):
                sk.call("done", {})
            else:
                mi.apply(entry["command"])
        end_tick = trace.get("end_tick") or trace.get("final_tick") or env.ticks
        if env.running and end_tick > env.ticks:
            env.tick(end_tick - env.ticks)
    else:
        for entry in trace.get("skill_log", []):
            if not env.running:
                break
            sk.call(entry["skill"], entry.get("args") or {})
    if env.running:  # the recorded episode was closed by the referee / runner: same rule
        env.run_for(env.task.success_hold + 0.2)
        if env.running:
            env.end("incomplete", "replay: episode closed")
    out = env.result()
    env.close()
    checks = {
        "status": out["status"] == rec["status"] or (rec["status"] in ("incomplete", "error") and
                                                     out["status"] == "incomplete"),
        "success": out["success"] == rec["success"],
        "progress": abs(float(out["progress"]) - float(rec["progress"])) < 1e-6,
        "milestones": out["milestones"] == rec["milestones"],
        "sim_time": abs(float(out["sim_time"]) - float(rec["sim_time"])) < 0.05,
    }
    return {"ok": all(checks.values()), "checks": checks, "recorded": {k: rec.get(k) for k in checks},
            "replayed": {k: out.get(k) for k in checks}}


def verify(path: str | Path) -> dict[str, Any]:
    return replay(_load_trace(Path(path)))
