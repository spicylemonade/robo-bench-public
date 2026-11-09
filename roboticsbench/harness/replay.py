"""Re-render an episode's video from its referee trace, after the fact.

Real-time episodes in CPU-only sandboxes cannot afford live video capture: software rendering at 12 fps drags the
humanoid scenes below 1x real time. They run without video, and the episode is replayed afterwards: the trace logs
every motor command with the tick it took effect at (``MotorInterface.command_log``), and the simulation is
deterministic, so re-applying the commands at the same ticks reproduces the episode - with no clock to keep.

    python -m roboticsbench.harness.replay referee_trace.json video.mp4

The returned report compares the replay's outcome with the recorded one (``match``); a mismatch means the video
may diverge from what the agent did (it never affects scoring - the signed referee result is the record).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def replay_video(trace_path: str | Path, out_path: str | Path, fps: int = 12, size: tuple[int, int] = (480, 360),
                 max_wall_s: float | None = None) -> dict[str, Any]:
    from roboticsbench.core.env import Env
    from roboticsbench.core.motor import MotorInterface

    trace = json.loads(Path(trace_path).read_text())
    control = trace.get("control") or "motor"
    if control not in ("motor", "velocity"):
        raise ValueError(f"replay supports motor / velocity episodes, not {control!r}")
    env = Env(trace["task"], seed=int(trace["seed"]), control=control)
    env.task.time_limit = 1e9  # the recording decides where the replay stops
    motor = MotorInterface(env, control)
    cmds = sorted((c for c in trace.get("motor_commands") or [] if "command" in c), key=lambda c: c["tick"])
    final = int(trace.get("end_tick") or trace.get("final_tick") or env.ticks)
    wall0 = time.time()
    env.start_video(out_path, fps=fps, width=size[0], height=size[1])
    i, cut = 0, False
    try:
        while env.ticks < final and env.running:
            while i < len(cmds) and cmds[i]["tick"] <= env.ticks:
                motor.apply(cmds[i]["command"])
                i += 1
            env.tick(1)
            if max_wall_s and time.time() - wall0 > max_wall_s:
                cut = True
                break
    finally:
        env.stop_video()
    rec = trace.get("result") or {}
    res = env.result()
    env.close()
    keys = ("status", "progress")
    return {"video": str(out_path), "ticks": int(env.ticks), "final_tick": final, "commands_applied": i,
            "commands": len(cmds), "cut_short": cut, "wall_s": round(time.time() - wall0, 1),
            "recorded": {k: rec.get(k) for k in keys}, "replayed": {k: res.get(k) for k in keys},
            "match": not cut and i == len(cmds) and abs(float(res.get("progress", 0)) - float(rec.get("progress", 0))) < 1e-6}


if __name__ == "__main__":
    print(json.dumps(replay_video(sys.argv[1], sys.argv[2]), indent=1))
