"""Task validation: run a task's scripted oracle over several seeds, plus a do-nothing baseline.

A task is healthy when the oracle succeeds on (almost) every seed through the public skill API and the
null agent never succeeds.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_oracle(task: str, seed: int, video: str | None = None) -> dict[str, Any]:
    from roboticsbench.agents.scripted import LoggedSkills
    from roboticsbench.tasks.arm.oracle_utils import EpisodeOver

    from roboticsbench.tasks import TASKS

    t0 = time.time()
    oc = TASKS[task].oracle_control
    env = Env(task, seed=seed, control=oc if oc in ("motor", "velocity") else "skills")
    build_s = time.time() - t0
    if video:
        env.start_video(video)
    sk = make_skills(env)
    err = tb = None
    try:
        if oc in ("motor", "velocity"):
            from roboticsbench.agents.motor_oracle import G1VelocityOracle, PandaMotorOracle
            from roboticsbench.core.motor import MotorInterface

            cls = PandaMotorOracle if env.robot.name == "panda" else G1VelocityOracle
            env.task.oracle(cls(env, MotorInterface(env, oc), sk))
        else:
            env.task.oracle(LoggedSkills(sk))
    except EpisodeOver:
        pass
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc(limit=4)
    if env.running:
        sk.done()
    env.stop_video()
    r = env.result()
    r.update(wall_s=round(time.time() - t0, 2), build_s=round(build_s, 2), oracle_error=err,
             n_actions=len(sk.log), time_limit=env.task.time_limit, oracle_control=oc)
    if tb:
        r["traceback"] = tb
    env.close()
    return r


def run_null(task: str, seed: int) -> dict[str, Any]:
    env = Env(task, seed=seed)
    while env.running:
        env.tick(50)
    r = env.result()
    env.close()
    return r


def validate(task: str, seeds: list[int], *, video_dir: str | None = None, null: bool = True,
             log=print) -> dict[str, Any]:
    oks, rows = 0, []
    for s in seeds:
        video = f"{video_dir}/{task.replace('/', '__')}_s{s}.mp4" if video_dir else None
        r = run_oracle(task, s, video)
        rows.append(r)
        oks += r["success"]
        log(f"[oracle] seed={s} status={r['status']:8s} progress={r['progress']:.2f} sim={r['sim_time']:6.1f}s "
            f"wall={r['wall_s']:5.1f}s actions={r['n_actions']:3d} milestones={r['milestones']} reason={r['reason']!r}"
            + (f" ERROR={r['oracle_error']}" if r.get("oracle_error") else ""))
        if r.get("traceback"):
            log(r["traceback"])
    null_bad = 0
    if null:
        for s in seeds[:2]:
            r = run_null(task, s)
            null_bad += r["success"]
            log(f"[null]   seed={s} status={r['status']:8s} progress={r['progress']:.2f} reason={r['reason']!r}")
    sims = [r["sim_time"] for r in rows if r["success"]]
    return {"task": task, "oracle_success_rate": oks / max(1, len(seeds)), "null_successes": null_bad,
            "mean_oracle_sim_s": round(sum(sims) / len(sims), 1) if sims else None,
            "time_limit": rows[0]["time_limit"] if rows else None}
