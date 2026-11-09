"""Run a coding agent (Claude Code, Codex, ...) on one RoboticsBench episode.

    referee (subprocess, owns the sim)  <-- HTTP -->  ./robot CLI  <--  coding agent (subprocess, cwd = workspace)

The orchestrator starts the referee for (task, seed), writes the agent workspace, launches the harness headless,
waits until the agent exits / the episode ends (+ grace period) / the wall-clock limit, then finalizes the episode
and writes ``result.json`` (same schema as the other runners, plus harness usage).

Real-time runs: the clock starts the moment the agent is launched - reading the docs, thinking and writing code all
count against the time budget (``time_limit_s``, e.g. 3 minutes per task), and the world never waits.

Isolation: in local runs the agent runs as the same OS user and could read files outside its workspace, so local
scores are for development. Official runs use the hosted runner, where the agent is an unprivileged user that
cannot read the simulator code, the referee or the ledger.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from roboticsbench.harness.agents import child_env, make_harness
from roboticsbench.harness.workspace import AGENT_PROMPT, make_workspace


def episode_id(task: str, seed: int) -> str:
    return f"{task.replace('/', '__')}__s{seed}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http(method: str, url: str, body: dict | None = None, headers: dict | None = None,
          timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _kill_tree(p: subprocess.Popen, grace: float = 5.0) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        p.terminate()
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        p.wait(timeout=grace)


def run_agent_episode(task: str, seed: int, harness: str, *, model: str | None = None, effort: str | None = None,
                      out_dir: str | os.PathLike = "runs/agents", run_id: str = "", wall_limit_s: float = 1800.0,
                      budget_usd: float | None = None, video: bool = False, grace_s: float | None = None,
                      isolated: bool = False, agent_user: str | None = None, max_actions: int | None = None,
                      workspace_root: str | os.PathLike | None = None, agent_env: dict[str, str] | None = None,
                      vision_only: bool = False, control: str = "motor", live_port: int = 0,
                      realtime: bool = False, time_limit_s: float | None = None,
                      python: str = sys.executable, log=print) -> dict[str, Any]:
    """``agent_user``: run the agent as this (unprivileged) OS user via ``runuser`` - requires running as root.
    ``workspace_root``: put the agent workspace there instead of inside the ledger directory (so an isolated agent
    cannot touch the ledger); it is copied next to the ledger afterwards. ``time_limit_s`` replaces the task's time
    limit (in real-time runs it counts from the agent's launch). ``grace_s``: how long an agent may keep running
    after the episode ended (default 90 s in real-time runs - enough to end its turn, so the harness reports token
    usage - else 20 s)."""
    realtime = bool(realtime) and control != "skills"
    grace = grace_s if grace_s is not None else (90.0 if realtime else 20.0)
    h = make_harness(harness)
    if not h.available():
        raise RuntimeError(f"harness {harness!r} is not installed ({h.binary!r} not on PATH)")
    variant = control + ("+vision" if vision_only else "")
    eid = episode_id(task, seed) + ("" if control == "skills" else f"__{control}") + ("__vision" if vision_only else "")
    ep = Path(out_dir) / eid
    ws = (Path(workspace_root) / eid) if workspace_root else ep / "workspace"
    ep.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    token = secrets.token_hex(16)
    url = f"http://127.0.0.1:{port}"
    op = {"X-Operator-Token": token}
    ref_cmd = [python, "-m", "roboticsbench.harness.referee", "--task", task, "--seed", str(seed), "--port", str(port),
               "--ledger", str(ep)]
    if video:
        ref_cmd.append("--video")
    if max_actions:
        ref_cmd += ["--max-actions", str(max_actions)]
    if vision_only:
        ref_cmd.append("--vision")
    ref_cmd += ["--control", control]
    if realtime:
        ref_cmd.append("--realtime")
    if time_limit_s:
        ref_cmd += ["--time-limit", str(float(time_limit_s))]
    live_token = secrets.token_urlsafe(12) if live_port else ""
    if live_port:
        ref_cmd += ["--live-port", str(live_port), "--live-title",
                    f"{task} (seed {seed}) - {harness}{' ' + model if model else ''}{' ' + effort if effort else ''}"
                    + (" - camera-only" if vision_only else "") + (" - real time" if realtime else "")]
        log(f"live view: http://127.0.0.1:{live_port}/{live_token}/")
        (ep / "live_url.txt").write_text(f"http://127.0.0.1:{live_port}/{live_token}/\n")
    wall0 = time.time()
    ref_log = open(ep / "referee.log", "w")
    ref = subprocess.Popen(ref_cmd, stdout=ref_log, stderr=subprocess.STDOUT,
                           env={**os.environ, "RBENCH_OPERATOR_TOKEN": token, "RBENCH_LIVE_TOKEN": live_token})
    agent = None
    stop_reason, infra_error = "", None
    result: dict[str, Any] = {}
    try:
        for _ in range(600):  # the first build of a scene can take a few seconds
            if ref.poll() is not None:
                raise RuntimeError(f"referee exited during start-up (see {ep / 'referee.log'})")
            try:
                _http("GET", url + "/health", timeout=2)
                break
            except OSError:
                time.sleep(0.2)
        info = _http("GET", url + "/info")
        pybin = None
        if not agent_user:  # local runs: numpy/scipy/mujoco for the agent's scripts, without roboticsbench
            from roboticsbench.harness.agentenv import agent_python_bin

            pybin = agent_python_bin()
        make_workspace(ws, info, url, python=os.path.join(pybin, "python") if pybin else "python3")
        if agent_user:  # official sandboxed runs: the agent owns only its workspace
            subprocess.run(["chown", "-R", f"{agent_user}:{agent_user}", str(ws)], check=True)
        # motor control: the launch prompt itself carries the task, the live interface and the robot's key data, so
        # the agent can act at once (GOAL.md / ROBOT.md / robot_spec.json hold the same and more)
        prompt = info.get("brief_md") or AGENT_PROMPT
        launch = h.launch(prompt, ws.resolve(), model=model, effort=effort, budget_usd=budget_usd,
                          isolated=isolated)
        for rel, text in launch.files.items():  # harness settings / prompt file: readable, inside the workspace
            path = (ws / rel).resolve()
            if ws.resolve() not in path.parents:
                raise ValueError(f"harness file outside the workspace: {rel}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            path.chmod(0o644)
            if agent_user:
                subprocess.run(["chown", "-R", f"{agent_user}:{agent_user}", str(path.parent)], check=True)
        cmd = (["runuser", "-u", agent_user, "--"] if agent_user else []) + launch.cmd
        log(f"[{eid}] {harness} {model or ''} -> {ws}")
        with open(ep / "agent_stdout.jsonl", "w") as so, open(ep / "agent_stderr.log", "w") as se:
            env = child_env({**launch.env, **(agent_env or {})})
            if pybin:  # also first on PATH (login shells may reorder it - ./python always works)
                env["PATH"] = pybin + os.pathsep + env.get("PATH", "")
                env.pop("VIRTUAL_ENV", None)
                env.pop("PYTHONPATH", None)
            if agent_user:
                env["HOME"] = str(Path("~" + agent_user).expanduser())
                env["USER"] = env["LOGNAME"] = agent_user
                env.pop("PYTHONPATH", None)
            if realtime:  # the clock starts as the agent gets in
                _http("POST", url + "/operator/start", {}, headers=op, timeout=30)
            stdin = subprocess.DEVNULL
            if launch.stdin is not None:
                (ep / "agent_prompt.md").write_text(launch.stdin)
                stdin = open(ep / "agent_prompt.md")
            agent = subprocess.Popen(cmd, cwd=ws, stdin=stdin, stdout=so, stderr=se,
                                     env=env, start_new_session=True)
            if stdin is not subprocess.DEVNULL:
                stdin.close()  # the child holds its own copy
            t0 = time.time()
            ended_at = None
            while True:
                rc = agent.poll()
                if rc is not None:
                    stop_reason = f"agent exited (code {rc})"
                    break
                if time.time() - t0 > wall_limit_s:
                    stop_reason = f"wall-clock limit ({wall_limit_s:.0f} s)"
                    break
                try:
                    status = _http("GET", url + "/health", timeout=600)["episode_status"]
                except OSError:
                    status = "unknown"
                if status != "running":
                    ended_at = ended_at or time.time()
                    if time.time() - ended_at > grace:
                        stop_reason = f"episode ended ({status})"
                        break
                time.sleep(1.0)
            _kill_tree(agent)
        result = _http("POST", url + "/operator/finalize", {"reason": f"agent stopped: {stop_reason}"}, headers=op)
    except Exception as e:  # noqa: BLE001 - recorded as an infra failure
        infra_error = f"{type(e).__name__}: {e}"
        if agent is not None:
            _kill_tree(agent)
        try:
            result = _http("POST", url + "/operator/finalize", {"reason": f"harness error: {infra_error}"}, headers=op)
        except Exception:  # noqa: BLE001
            result = {"task": task, "seed": seed, "status": "error", "success": False, "progress": 0.0, "score": 0.0}
    finally:
        try:
            _http("POST", url + "/operator/shutdown", {}, headers=op, timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            ref.wait(timeout=30)
        except subprocess.TimeoutExpired:
            ref.kill()
        ref_log.close()

    if workspace_root and ws.exists():  # keep the agent's files (views, scripts) with the episode record
        import shutil

        shutil.copytree(ws, ep / "workspace", dirs_exist_ok=True, ignore=shutil.ignore_patterns(".robot_api"))
    # Vals security rule: no injected key may survive in logs, traces or the workspace copy
    from roboticsbench.scrub import scrub_obj, scrub_tree, secret_values

    redactions = scrub_tree(ep, extra=agent_env)
    parsed = h.parse(ep / "agent_stdout.jsonl") if agent is not None else {}
    no_activity = all(not result.get(k) for k in ("n_skill_calls", "n_observations", "n_step_calls",
                                                    "n_motor_commands"))
    # an agent that crashed or could not start (exited by itself, non-zero, never touched the robot) is an infra
    # failure; one we stopped because the episode / wall clock ran out while it was still thinking is not - it is a
    # model that never acted, and scores 0 like any other failure
    exited_by_itself = stop_reason.startswith("agent exited")
    if (infra_error is None and no_activity and exited_by_itself and agent is not None
            and agent.returncode not in (0, None)):
        infra_error = f"agent made no robot calls and exited with code {agent.returncode} (see agent_stderr.log)"
    if parsed.get("is_error") and no_activity:
        infra_error = infra_error or f"agent error: {parsed.get('final_message', '')[:300]}"
    label = f"{harness}/{model}" if model else harness
    result.update({
        "run_id": run_id, "episode_id": eid, "task": task, "seed": seed,
        "variant": variant, "control": control,
        "agent": {"agent": "coding-agent", "harness": harness, "model": model, "effort": effort,
                  "harness_version": parsed.get("harness_version"), "isolated": isolated},
        "model": label,
        "usage": parsed.get("usage") or {},
        "n_model_turns": parsed.get("num_turns") or parsed.get("turns"),
        "served_models": parsed.get("served_models"),
        "agent_detail": parsed.get("detail") or {},  # harness-specific diagnostics (e.g. Antigravity upstream calls)
        "agent_stop_reason": stop_reason,
        "agent_final_message": parsed.get("final_message", ""),
        "wall_s": round(time.time() - wall0, 2),
        "infra_ok": infra_error is None,
        "error": infra_error,
    })
    if infra_error and result.get("status") != "success":
        result["status"] = "error"
    result.setdefault("score", 1.0 if result.get("success") else float(result.get("progress", 0.0)))
    result, n_result = scrub_obj(result, secret_values(agent_env))
    result["secrets_redacted"] = sum(redactions.values()) + n_result
    if result["secrets_redacted"]:
        log(f"[{eid}] redacted {result['secrets_redacted']} secret occurrence(s): {sorted(redactions)}")
    (ep / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result
