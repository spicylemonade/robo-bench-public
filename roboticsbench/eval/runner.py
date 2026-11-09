"""Episode runner: build env -> run agent -> final evaluation -> write trace / result / video."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from roboticsbench.agents import Agent, AgentTrace, make_agent
from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills

SCORER_REVISION = "rb-v1"


def episode_id(task: str, seed: int) -> str:
    return f"{task.replace('/', '__')}__s{seed}"


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def sign(entry: dict[str, Any], secret: bytes | None) -> str | None:
    if not secret:
        return None
    payload = json.dumps({k: v for k, v in entry.items() if k != "sig"}, sort_keys=True, separators=(",", ":"),
                         default=_json_default).encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def run_episode(task: str, agent: str | Agent, seed: int = 0, *, out_dir: str | os.PathLike | None = None,
                video: bool = False, video_camera: str | None = None, video_size: tuple[int, int] = (640, 480),
                video_fps: int = 25,
                agent_kwargs: dict[str, Any] | None = None, env_kwargs: dict[str, Any] | None = None,
                skill_kwargs: dict[str, Any] | None = None, run_id: str = "", vision_only: bool = False) -> dict[str, Any]:
    wall0 = time.time()
    ag = make_agent(agent, **(agent_kwargs or {})) if isinstance(agent, str) else agent
    env = Env(task, seed=seed, **(env_kwargs or {}))
    skill_kwargs = dict(skill_kwargs or {})
    if vision_only:
        skill_kwargs["allow_object_list"] = False
    sk = make_skills(env, **skill_kwargs)
    eid = episode_id(task, seed) + ("__vision" if vision_only else "")
    ep_dir = Path(out_dir) / eid if out_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=True)
    if video and ep_dir:
        env.start_video(ep_dir / "video.mp4", camera=video_camera, width=video_size[0], height=video_size[1],
                        fps=video_fps)
    error = None
    try:
        trace = ag.run(env, sk)
    except Exception as e:  # noqa: BLE001
        error = traceback.format_exc(limit=8)
        trace = AgentTrace(agent=getattr(ag, "name", "agent"), model=getattr(ag, "model", None),
                           stop_reason=f"agent crashed: {type(e).__name__}: {e}")
    infra_ok = "infra_error" not in trace.notes and error is None
    if env.running:
        # agent stopped early: give the scene a moment, then score the state as it is
        env.run_for(env.task.success_hold + 0.2)
        if env.running:
            env.end("incomplete", f"agent stopped before completing the task ({trace.stop_reason})")
    if not infra_ok and env.status != "success":
        env.status = "error"
    video_path = env.stop_video()
    res = env.result()
    res.update({
        "run_id": run_id,
        "episode_id": eid,
        "variant": "skills" + ("+vision" if vision_only else ""),
        "agent": ag.describe() if hasattr(ag, "describe") else {"agent": str(agent)},
        "model": getattr(ag, "model", None),
        "category": env.task.category,
        "difficulty": env.task.difficulty,
        "robot": env.task.robot,
        "score": 1.0 if res["success"] else float(res["progress"]),
        "n_actions": len(sk.log),
        "declared_done": sk.declared_done,
        "agent_stop_reason": trace.stop_reason,
        "usage": trace.usage,
        "n_model_turns": len(trace.turns),
        "wall_s": round(time.time() - wall0, 2),
        "infra_ok": infra_ok,
        "error": error,
        "scorer_revision": SCORER_REVISION,
        "video": str(video_path.name) if video_path else None,
    })
    if ep_dir:
        tr = {"instruction": env.task.instruction, "task": task, "seed": seed, "result": res,
              "skill_log": sk.log, "agent_turns": trace.turns, "agent_notes": trace.notes,
              "tools": sk.tools()}
        tpath = ep_dir / "trace.json"
        from roboticsbench.scrub import scrub_obj, secret_values

        secrets = secret_values()
        res, _ = scrub_obj(res, secrets)
        tr["result"] = res
        tr, _ = scrub_obj(tr, secrets)
        tbytes = json.dumps(tr, indent=1, default=_json_default).encode()
        tmp = tpath.with_suffix(".json.part")
        tmp.write_bytes(tbytes)
        tmp.replace(tpath)
        res["trace_sha256"] = hashlib.sha256(tbytes).hexdigest()
        secret = os.environ.get("RBENCH_LEDGER_SECRET", "").encode() or None
        res["sig"] = sign(res, secret)
        (ep_dir / "result.json").write_text(json.dumps(res, indent=1, default=_json_default))
    env.close()
    return res
