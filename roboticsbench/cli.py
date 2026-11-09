"""``rbench`` command line.

    rbench list [--robot g1|panda] [--json]
    rbench validate <task|suite> [--seeds 0-4] [--video-dir out]
    rbench run <task> --agent <spec> [--seed 0] [--out runs/dev] [--video]
    rbench bench --agent <spec> [--tasks all] [--seeds 0-2] [--workers 8] [--out runs] [--video]
    rbench agent <task> --harness claude-code|codex|cmd:<template> [--model M] [--seed 0] [--video]
    rbench agents --harness claude-code --model M [--tasks all] [--seeds 0-2] [--workers 4]
    rbench report <run_dir>
    rbench verify <episode_dir> ...             # replay the recorded skill calls, check the outcome
    rbench view <task> [--seed 0]              # interactive MuJoCo viewer (on macOS run it with `mjpython -m roboticsbench.cli view ...`)
    rbench serve [--host 0.0.0.0] [--port 8765] # HTTP API so external agents can drive episodes
    rbench fetch-assets

Agent specs (bench/run): oracle | null | random | llm:<provider>/<model> | policy:<module>:<Class> | remote:<url>
(e.g. llm:anthropic/claude-opus-5, llm:openai/gpt-5.1, llm:openrouter/google/gemini-3-pro).
Coding agents (agent/agents): Claude Code, Codex or any CLI drive the robot through a ./robot command in a
workspace; the simulator runs in a separate referee process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REALTIME_TIME_LIMIT_S = 180.0  # real-time agent runs: 3 minutes per task, from the agent's launch
DEFAULT_EPISODE_BUDGET_USD = 25.0  # hard per-episode spend cap for harnesses that support one (claude-code)


def _agent_kwargs(a: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if a.agent.startswith("llm:"):
        if a.effort:
            kw["effort"] = None if a.effort == "none" else a.effort
        if a.max_actions:
            kw["max_actions"] = a.max_actions
        if a.cameras:
            kw["cameras"] = tuple(a.cameras.split(","))
        if a.image_size:
            w, h = (int(x) for x in a.image_size.lower().split("x"))
            kw["image_size"] = (w, h)
    for item in a.agent_arg or []:
        k, _, v = item.partition("=")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return kw


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent", default="oracle", help="agent spec (see rbench --help)")
    p.add_argument("--effort", default=None, help="LLM reasoning effort (low|medium|high|max|none)")
    p.add_argument("--max-actions", type=int, default=0, help="LLM tool-call budget (default: task's)")
    p.add_argument("--cameras", default=None, help="comma-separated cameras shown to the LLM")
    p.add_argument("--image-size", default=None, help="LLM image size, e.g. 640x480")
    p.add_argument("--agent-arg", action="append", help="extra agent kwarg key=value (JSON values allowed)")
    p.add_argument("--video", action="store_true", help="record an MP4 per episode")
    p.add_argument("--video-camera", default=None)
    p.add_argument("--vision", action="store_true", help="vision-only variant: no list_objects()")


def cmd_list(a: argparse.Namespace) -> int:
    from roboticsbench.tasks import LOAD_ERRORS, TASKS

    robot = {"humanoid": "g1", "arm": "panda"}.get(a.robot, a.robot)
    from roboticsbench.tasks.base import SHELVED

    rows = [c for n, c in sorted(TASKS.items()) if (robot is None or c.robot == robot)
            and (a.dev or not n.startswith("dev/")) and (a.shelved or n not in SHELVED)]
    if a.markdown:
        print("| task | robot | category | difficulty | control modes | description |")
        print("|---|---|---|---|---|---|")
        for c in rows:
            flag = " *(vision only)*" if not c.allow_object_list else ""
            print(f"| `{c.name}` | {'G1 humanoid' if c.robot == 'g1' else 'Panda arm'} | {c.category} | "
                  f"{c.difficulty} | {', '.join(c.controls)} | {c.summary}{flag} |")
        return 0
    if a.json:
        print(json.dumps([{"name": c.name, "robot": c.robot, "category": c.category, "difficulty": c.difficulty,
                           "time_limit": c.time_limit, "max_actions": c.max_actions,
                           "vision_only": not c.allow_object_list, "summary": c.summary} for c in rows], indent=1))
    else:
        for c in rows:
            flag = " [vision]" if not c.allow_object_list else ""
            ctl = "" if set(c.controls) == {"skills", "motor", "velocity"} else f" [{'/'.join(c.controls)} only]"
            print(f"{c.name:32s} {c.robot:6s} {c.category:20s} {c.difficulty:7s} {c.summary}{flag}{ctl}")
        print(f"\n{len(rows)} tasks")
    for mod, err in LOAD_ERRORS.items():
        print(f"!! failed to load {mod}:\n{err}", file=sys.stderr)
    return 0


def _validate_one(job: tuple) -> tuple[dict[str, Any], list[str]]:
    from roboticsbench.eval.validate import validate

    task, seeds, video_dir, null = job
    lines: list[str] = []
    try:
        return validate(task, seeds, video_dir=video_dir, null=null, log=lines.append), lines
    except Exception as e:  # noqa: BLE001
        return {"task": task, "oracle_success_rate": 0.0, "null_successes": 0, "mean_oracle_sim_s": None,
                "time_limit": None, "error": f"{type(e).__name__}: {e}"}, lines


def cmd_validate(a: argparse.Namespace) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    from roboticsbench.eval.suites import resolve_tasks
    from roboticsbench.eval.validate import parse_seeds

    seeds = parse_seeds(a.seeds)
    worst = 0
    summary = []
    jobs = [(t, seeds, a.video_dir, not a.no_null) for t in resolve_tasks(a.task)]
    if a.workers > 1 and len(jobs) > 1:
        ex = cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
        outs = ex.map(_validate_one, jobs)
    else:
        ex, outs = None, map(_validate_one, jobs)
    for res, lines in outs:
        if len(jobs) == 1 or a.verbose:
            print("\n".join(lines))
        print(json.dumps(res))
        summary.append(res)
        if res["oracle_success_rate"] < a.min_success or res["null_successes"]:
            worst = 1
    if ex is not None:
        ex.shutdown()
    if len(summary) > 1:
        print("\nsummary:")
        for r in summary:
            print(f"  {r['task']:32s} oracle {100 * r['oracle_success_rate']:5.1f}%  null {r['null_successes']}  "
                  f"sim {r['mean_oracle_sim_s']}/{r['time_limit']}s" + (f"  ERROR {r['error']}" if r.get("error") else ""))
    return worst


def cmd_run(a: argparse.Namespace) -> int:
    from roboticsbench.eval.runner import run_episode

    out = a.out or f"runs/dev-{time.strftime('%Y%m%d-%H%M%S')}"
    res = run_episode(a.task, a.agent, a.seed, out_dir=out, video=a.video, video_camera=a.video_camera,
                      agent_kwargs=_agent_kwargs(a), vision_only=a.vision)
    print(json.dumps({k: res.get(k) for k in ("task", "seed", "status", "score", "reason", "milestones", "sim_time",
                                               "n_actions", "wall_s", "usage")}, indent=1))
    print(f"episode files: {Path(out) / res['episode_id']}")
    return 0 if res["success"] else 1


def _bench_worker(job: tuple) -> dict[str, Any]:
    task, agent, seed, out, video, video_camera, kw, run_id, vision = job
    from roboticsbench.eval.runner import run_episode

    try:
        return run_episode(task, agent, seed, out_dir=out, video=video, video_camera=video_camera, agent_kwargs=kw,
                           run_id=run_id, vision_only=vision)
    except Exception as e:  # noqa: BLE001 - a crashed episode must not kill the run
        return {"task": task, "seed": seed, "status": "error", "success": False, "score": 0.0, "progress": 0.0,
                "infra_ok": False, "error": f"{type(e).__name__}: {e}", "model": agent, "n_actions": 0,
                "sim_time": 0.0, "wall_s": 0.0}


def cmd_bench(a: argparse.Namespace) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    from roboticsbench.eval.report import write_report
    from roboticsbench.eval.results import leaderboard_text, summarize
    from roboticsbench.eval.suites import resolve_tasks
    from roboticsbench.eval.validate import parse_seeds

    tasks = resolve_tasks(a.tasks)
    if a.agent.startswith("llm:"):  # the direct-API agent drives the skill API
        tasks = _supported(tasks, "skills")
    seeds = parse_seeds(a.seeds)
    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S-") + a.agent.replace(":", "-").replace("/", "-")[:48]
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    kw = _agent_kwargs(a)
    jobs = [(t, a.agent, s, str(out), a.video, a.video_camera, kw, run_id, a.vision) for t in tasks for s in seeds]
    print(f"run {run_id}: {len(jobs)} episodes ({len(tasks)} tasks x {len(seeds)} seeds), agent={a.agent}, "
          f"workers={a.workers}")
    results = []
    with open(out / "results.jsonl", "w") as f, \
            cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_bench_worker, j) for j in jobs]
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(f"  {r['task']:32s} s{r['seed']} {r['status']:10s} score={r['score']:.2f} "
                  f"actions={r.get('n_actions', 0):3d} sim={r.get('sim_time', 0):6.1f}s wall={r.get('wall_s', 0):6.1f}s")
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(leaderboard_text(summary))
    print("report:", write_report(out, results))
    return 0


def _supported(tasks: list[str], control: str) -> list[str]:
    """Drop tasks that cannot be solved in this control mode (e.g. throwing needs motor control)."""
    from roboticsbench.tasks import TASKS

    keep = [t for t in tasks if control in TASKS[t].controls]
    skipped = sorted(set(tasks) - set(keep))
    if skipped:
        print(f"skipping {len(skipped)} task(s) not solvable with --control {control}: {', '.join(skipped)}")
    return keep


def _agent_episode(job: tuple) -> dict[str, Any]:
    from roboticsbench.harness.run import run_agent_episode

    task, seed, a_dict, out, run_id = job
    realtime = not a_dict.get("turn_based", False) and a_dict["control"] != "skills"
    time_limit = a_dict.get("time_limit") or (REALTIME_TIME_LIMIT_S if realtime else None)
    try:
        return run_agent_episode(task, seed, a_dict["harness"], model=a_dict["model"], effort=a_dict["effort"],
                                 out_dir=out, run_id=run_id, wall_limit_s=a_dict["wall_limit"],
                                 budget_usd=a_dict["budget_usd"], video=a_dict["video"],
                                 max_actions=a_dict["max_actions"] or None, vision_only=a_dict["vision"],
                                 control=a_dict["control"], live_port=a_dict.get("live", 0) or 0,
                                 realtime=realtime, time_limit_s=time_limit,
                                 log=print if a_dict.get("live") else (lambda *_: None))
    except Exception as e:  # noqa: BLE001
        return {"task": task, "seed": seed, "status": "error", "success": False, "score": 0.0, "progress": 0.0,
                "infra_ok": False, "error": f"{type(e).__name__}: {e}", "n_actions": 0, "sim_time": 0.0,
                "wall_s": 0.0, "model": f"{a_dict['harness']}/{a_dict['model']}"}


def _agent_opts(a: argparse.Namespace) -> dict[str, Any]:
    return {"harness": a.harness, "model": a.model, "effort": a.effort, "wall_limit": a.wall_limit,
            "budget_usd": a.budget_usd, "video": a.video, "max_actions": a.max_actions, "vision": a.vision,
            "control": a.control, "live": getattr(a, "live", 0), "turn_based": a.turn_based,
            "time_limit": a.time_limit}


def cmd_agent(a: argparse.Namespace) -> int:
    out = a.out or f"runs/agent-{time.strftime('%Y%m%d-%H%M%S')}"
    res = _agent_episode((a.task, a.seed, _agent_opts(a), out, ""))
    print(json.dumps({k: res.get(k) for k in ("task", "seed", "status", "score", "reason", "milestones", "sim_time",
                                               "n_actions", "n_observations", "agent_stop_reason", "wall_s", "usage",
                                               "error")}, indent=1, default=str))
    print(f"episode files: {Path(out) / res.get('episode_id', '')}")
    return 0 if res.get("success") else 1


def cmd_agents(a: argparse.Namespace) -> int:
    import concurrent.futures as cf

    from roboticsbench.eval.report import write_report
    from roboticsbench.eval.results import leaderboard_text, summarize
    from roboticsbench.eval.suites import resolve_tasks
    from roboticsbench.eval.validate import parse_seeds

    tasks = _supported(resolve_tasks(a.tasks), a.control)
    seeds = parse_seeds(a.seeds)
    label = f"{a.harness}-{a.model or 'default'}-{a.control}".replace(":", "-").replace("/", "-")[:56]
    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S-") + label
    out = Path(a.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    opts = _agent_opts(a)
    jobs = [(t, s, opts, str(out), run_id) for t in tasks for s in seeds]
    print(f"run {run_id}: {len(jobs)} episodes, harness={a.harness} model={a.model} workers={a.workers}")
    results = []
    with open(out / "results.jsonl", "w") as f, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in cf.as_completed([ex.submit(_agent_episode, j) for j in jobs]):
            r = fut.result()
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            cost = (r.get("usage") or {}).get("cost_usd")
            print(f"  {r['task']:32s} s{r['seed']} {r['status']:10s} score={r['score']:.2f} "
                  f"actions={r.get('n_actions', 0):3d} wall={r.get('wall_s', 0):6.1f}s"
                  + (f" ${cost:.2f}" if cost is not None else "") + (f" ERROR {r['error'][:80]}" if r.get("error") else ""))
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(leaderboard_text(summary))
    print("report:", write_report(out, results))
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from roboticsbench.eval.verify import verify

    bad = 0
    for path in a.paths:
        res = verify(path)
        bad += not res["ok"]
        print(json.dumps({"episode": str(path), **res}, default=str))
    return 1 if bad else 0


def cmd_report(a: argparse.Namespace) -> int:
    from roboticsbench.eval.report import write_report

    print(write_report(a.run_dir))
    return 0


def cmd_view(a: argparse.Namespace) -> int:
    import mujoco.viewer

    from roboticsbench.core.env import Env
    from roboticsbench.skills import make_skills

    env = Env(a.task, seed=a.seed)
    print(env.task.instruction)
    if a.oracle:
        import threading

        from roboticsbench.agents.scripted import LoggedSkills

        sk = make_skills(env)
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            env.tick_callbacks.append(lambda e: (v.sync(), time.sleep(max(0.0, e.control_dt / a.speed - 0.002))))
            t = threading.Thread(target=lambda: env.task.oracle(LoggedSkills(sk)), daemon=True)
            t.start()
            while v.is_running() and t.is_alive():
                time.sleep(0.1)
            print(env.result())
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            while v.is_running() and env.running:
                t0 = time.time()
                env.tick()
                v.sync()
                time.sleep(max(0.0, env.control_dt / a.speed - (time.time() - t0)))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from roboticsbench.server import serve

    serve(host=a.host, port=a.port)
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    from roboticsbench.assets.fetch import fetch

    fetch(force=a.force)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list tasks")
    p.add_argument("--robot", default=None, help="g1|panda|humanoid|arm")
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--dev", action="store_true", help="include dev/ (tutorial, smoke-test) tasks")
    p.add_argument("--shelved", action="store_true", help="include the shelved tasks (out of the benchmark)")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("validate", help="run oracle + null baselines for a task / suite")
    p.add_argument("task", help="task name, suite name, glob (arm/*) or 'all'")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--min-success", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=1, help="validate several tasks in parallel")
    p.add_argument("--verbose", action="store_true", help="print per-seed lines for every task")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("run", help="run one episode")
    p.add_argument("task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    _add_agent_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("bench", help="run many episodes locally in parallel")
    p.add_argument("--tasks", default="all")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="runs")
    p.add_argument("--run-id", default="")
    _add_agent_args(p)
    p.set_defaults(fn=cmd_bench)

    def _harness_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--harness", required=True, help="claude-code | codex | cmd:<command template>")
        p.add_argument("--model", default=None, help="model passed to the harness (default: harness default)")
        p.add_argument("--effort", default=None, help="reasoning effort passed to the harness")
        p.add_argument("--wall-limit", type=float, default=1800.0, help="wall-clock seconds per episode")
        p.add_argument("--budget-usd", type=float, default=DEFAULT_EPISODE_BUDGET_USD,
                       help=f"per-episode spend cap in USD for claude-code (default {DEFAULT_EPISODE_BUDGET_USD:g}; "
                            "0 = uncapped). Vals security rule: agent runs need hard cost caps")
        p.add_argument("--max-actions", type=int, default=0, help="robot action budget (default: task's)")
        p.add_argument("--video", action="store_true")
        p.add_argument("--vision", action="store_true", help="vision-only variant: no list_objects()")
        p.add_argument("--turn-based", action="store_true",
                       help="pause the simulation between the agent's calls (default for motor / velocity control: "
                            "real time - the world keeps running while the agent thinks)")
        p.add_argument("--time-limit", type=float, default=0.0,
                       help=f"time budget per episode in seconds (default: {REALTIME_TIME_LIMIT_S:.0f} s of real time "
                            "counted from the agent's launch in real-time runs; the task's own limit when "
                            "turn-based)")
        p.add_argument("--control", default="motor", choices=["motor", "velocity", "skills"],
                       help="motor: joint-level motor commands only (default); velocity: G1 legs via the walking "
                            "policy + joint-level arms; skills: the high-level skill API")

    p = sub.add_parser("agent", help="run a coding agent (Claude Code, Codex, ...) on one episode")
    p.add_argument("task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--live", type=int, default=0, metavar="PORT",
                   help="watch the episode live in a browser at http://127.0.0.1:PORT/ (sim view, status, agent feed)")
    _harness_args(p)
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("agents", help="run a coding agent over many episodes")
    p.add_argument("--tasks", default="all")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs")
    p.add_argument("--run-id", default="")
    _harness_args(p)
    p.set_defaults(fn=cmd_agents)

    p = sub.add_parser("verify", help="re-simulate recorded episodes and check their outcome")
    p.add_argument("paths", nargs="+", help="episode directories (or trace files)")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("report", help="(re)write report.html for a run directory")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("view", help="interactive viewer (macOS: run under mjpython)")
    p.add_argument("task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle", action="store_true", help="play the scripted oracle")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (x real time)")
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser("serve", help="HTTP API for external agents")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("fetch-assets", help="download robot assets")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
