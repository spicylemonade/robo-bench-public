"""Referee: owns ONE episode (task + seed fixed at start-up), exposes the allowlisted robot API to an agent over
HTTP on localhost, enforces the budgets, and writes the (optionally HMAC-signed) ledger. The agent never gets
simulator access - only the skill API, camera images and public state.

    python -m roboticsbench.harness.referee --task arm/lego_manual --seed 0 --port 8766 --ledger runs/x/ep [--video]

Control modes (``--control``): ``skills`` (high-level skill API), ``motor`` (joint-level motor commands only, see
:mod:`roboticsbench.core.motor`) or ``velocity`` (G1: walking policy driven by body velocity, joints for the rest).

Agent endpoints (JSON):
    GET  /health
    GET  /info                      task, robot, control mode, cameras, budgets, skills or motor spec, GOAL/ROBOT.md
    GET  /state                     simulated time, robot state (skills) or sensors (motor), task + episode status
    GET  /stream                    motor modes: the /state body once per control tick, one JSON line each (the
                                    robot's live senses, used by robot.py); not counted as observations
    POST /act   {skill, args}       skills mode: run one skill (a robot action unless look / list_objects)
    POST /step  {command?, ticks}   motor modes: apply a motor command (persistent) and advance `ticks` x 20 ms
    POST /objects                   list_objects() when the task allows it (all modes)
    POST /look  {cameras?, width?, height?}   -> {images: {camera: base64 JPEG}}
    POST /done                      declare completion (the episode ends after a short settle and is scored)
Operator endpoints (header ``X-Operator-Token`` = $RBENCH_OPERATOR_TOKEN at start-up):
    POST /operator/start                real-time mode: start the clock now (the orchestrator calls it as it launches
                                        the agent; otherwise the agent's first robot call starts it)
    POST /operator/finalize {reason?}   end a still-running episode as incomplete, write the ledger, return result
    POST /operator/shutdown
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills

OBSERVATION_SKILLS = {"look", "list_objects"}
MAX_STEP_TICKS = 250  # 5 s per /step call
MAX_STEP_CALLS = 200_000
SCORER_REVISION = "rb-v1"


class Referee:
    def __init__(self, task: str, seed: int, ledger_dir: str | os.PathLike, *, max_actions: int | None = None,
                 max_observations: int = 3000, width: int = 640, height: int = 480, video: bool = False,
                 video_fps: int = 12, video_size: tuple[int, int] = (480, 360), vision_only: bool = False,
                 control: str = "skills", live_port: int = 0, live_title: str = "", realtime: bool = False,
                 time_limit: float | None = None):
        if control not in ("skills", "motor", "velocity"):
            raise ValueError(f"unknown control mode {control!r}")
        self.control = control
        self.env = Env(task, seed=seed, control=control)
        if control not in self.env.task.controls:
            raise ValueError(f"task {task} cannot be solved with {control!r} control (supports: "
                             f"{', '.join(self.env.task.controls)})")
        if time_limit:  # a per-run time budget replaces the task's own limit (e.g. 3 min of real time per task)
            if not time_limit > 0:
                raise ValueError("time_limit must be positive")
            self.env.task.time_limit = float(time_limit)
        self.vision_only = bool(vision_only)
        self.env.vision_only = self.vision_only  # tasks hide ground-truth state from their status in this mode
        self.sk = make_skills(self.env, allow_object_list=False if vision_only else None)
        self.motor = None
        self.step_calls = 0
        if control != "skills":
            from roboticsbench.core.motor import MotorInterface

            self.motor = MotorInterface(self.env, control)
            if control == "motor":
                max_observations = max(max_observations, 50_000)
        # real-time mode: once the clock starts (the orchestrator starts it as it launches the agent, else the agent's
        # first robot call does) the simulation runs continuously at 1x wall-clock speed in a background thread and
        # never waits for the agent (the last motor command persists)
        self.realtime = bool(realtime)
        if self.realtime and self.motor is None:
            raise ValueError("real-time mode needs motor or velocity control")
        self.lock = threading.RLock()
        self.tick_cond = threading.Condition(self.lock)
        self._runner: threading.Thread | None = None
        self._stop_runner = False
        self.done_at_tick: int | None = None
        self.rt_max_lag = 0.0
        self.task_name, self.seed = task, int(seed)
        self.max_actions = int(max_actions or self.env.task.max_actions)
        self.max_observations = int(max_observations)
        self.width, self.height = int(width), int(height)
        self.actions = 0
        self.observations = 0
        self.events: list[dict[str, Any]] = []
        self.ledger = Path(ledger_dir)
        self.ledger.mkdir(parents=True, exist_ok=True)
        self.result: dict[str, Any] | None = None
        self.wall0 = time.time()
        self.video = video
        if video:
            self.env.start_video(self.ledger / "video.mp4", fps=video_fps, width=video_size[0], height=video_size[1])
        self.tools = self.sk.tools()
        self._info: dict[str, Any] | None = None
        self.live = None
        if live_port:  # local live view of the episode (see harness/live.py)
            from roboticsbench.harness.live import LiveView

            self.live = LiveView(self.ledger, live_port, title=live_title,
                                 token=os.environ.pop("RBENCH_LIVE_TOKEN", ""))
            self.live.attach(self.env, self)

    @property
    def variant(self) -> str:
        return self.control + ("+vision" if self.vision_only else "")

    # ------------------------------------------------------------------ agent API
    def info(self) -> dict[str, Any]:
        """Static episode information (computed once: the documents can take a moment to render)."""
        if self._info is None:
            self._info = self._make_info()
        return self._info

    def _make_info(self) -> dict[str, Any]:
        from roboticsbench.agents.llm.prompts import (goal_markdown, manual_markdown, motor_brief, motor_goal_markdown,
                                                      motor_manual)

        env = self.env
        out = {"task": self.task_name, "robot": env.robot.name, "instruction": env.task.instruction_for(self.vision_only),
               "control": self.control, "variant": self.variant, "realtime": self.realtime,
               "cameras": list(env.obs_cameras), "time_limit_s": env.task.time_limit,
               "object_list_available": self.sk.allow_object_list,
               "documents": {k: base64.b64encode(v).decode() for k, v in env.task.documents(env).items()}}
        if self.motor is None:
            out.update({"max_actions": self.max_actions, "tools": self.tools,
                        "goal_md": goal_markdown(env, self.max_actions, self.tools),
                        "manual_md": manual_markdown(env, self.tools, tuple(env.obs_cameras), self.max_actions)})
        else:
            from roboticsbench.core.motor import robot_model_xml

            spec = self.motor.spec()
            out.update({"motor_spec": spec, "max_step_ticks": MAX_STEP_TICKS,
                        "robot_model_xml": robot_model_xml(env.robot.name),
                        "goal_md": motor_goal_markdown(env, self.control, self.sk.allow_object_list,
                                                       realtime=self.realtime),
                        "manual_md": motor_manual(env, self.control, spec, tuple(env.obs_cameras),
                                                  self.sk.allow_object_list, realtime=self.realtime),
                        "brief_md": motor_brief(env, self.control, spec, tuple(env.obs_cameras),
                                                self.sk.allow_object_list, realtime=self.realtime)})
        return out

    def state(self) -> dict[str, Any]:
        env = self.env
        if self.motor is not None:
            obs = {"sensors": self.motor.sensors(), "time_left": round(env.time_left, 3),
                   "task": env.observe(images=False)["task"]}
        else:
            obs = env.observe(images=False)
            obs["actions_used"] = self.actions
            obs["max_actions"] = self.max_actions
        obs["episode_status"] = env.status
        if env.fail_reason:
            obs["episode_end_reason"] = env.fail_reason
        return obs

    def step(self, command: dict[str, Any] | None, ticks: Any) -> dict[str, Any]:
        """Motor modes: apply ``command`` (if given; it persists) and advance ``ticks`` control ticks."""
        env = self.env
        if self.motor is None:
            return {"ok": False, "message": "this episode uses the skill API (/act); /step is for motor control"}
        self.step_calls += 1
        if self.step_calls > MAX_STEP_CALLS:
            return {"ok": False, "message": f"step budget exhausted ({MAX_STEP_CALLS} calls)", **self.state()}
        try:
            n = int(ticks)
        except (TypeError, ValueError):
            return {"ok": False, "message": "ticks must be an integer", **self.state()}
        if not 0 <= n <= MAX_STEP_TICKS:
            return {"ok": False, "message": f"ticks must be in [0, {MAX_STEP_TICKS}]", **self.state()}
        if not env.running:
            return {"ok": False, "message": f"episode already ended ({env.status}: {env.fail_reason})",
                    **self.state()}
        if command:
            try:
                self.motor.apply(command)
            except ValueError as e:
                return {"ok": False, "message": f"bad command: {e}", **self.state()}
        n_events = len(self.motor.events)
        if n:
            if self.realtime:
                self._wait_ticks(n)
            else:
                env.tick(n)
                self.tick_cond.notify_all()  # stream readers (the caller holds the lock)
        self._maybe_write_ledger()
        return {"ok": True, "events": self.motor.events[n_events:], **self.state()}

    def objects(self) -> dict[str, Any]:
        if not self.sk.allow_object_list:
            return {"ok": False, "message": "list_objects is not available in this run (vision only)"}
        err = self._observe_budget()
        if err:
            return {"ok": False, "message": err}
        res = self.sk.list_objects()
        return {"ok": True, "t_sim": round(self.env.t, 3), "objects": res.data["objects"]}

    def _observe_budget(self) -> str | None:
        self.observations += 1
        if self.observations > self.max_observations:
            return f"observation budget exhausted ({self.max_observations} state/look/objects requests)"
        return None

    def act(self, skill: str, args: dict[str, Any] | None) -> dict[str, Any]:
        env = self.env
        t0 = time.time()
        if skill in ("look", "done"):
            return self.look() if skill == "look" else self.done()
        if self.motor is not None:
            return self._reply({"ok": False, "message": f"skills are not available: this episode uses "
                                                        f"{self.control} control (POST /step)"})
        invalid = self.sk.check(skill, args)
        if invalid:  # typos do not touch the robot and do not use the budget
            self.events.append({"kind": "invalid", "skill": skill, "args": args or {}, "message": invalid})
            return self._reply({"ok": False, "message": invalid})
        if skill in OBSERVATION_SKILLS:
            err = self._observe_budget()
            if err:
                return self._reply({"ok": False, "message": err})
        elif env.running:
            if self.actions >= self.max_actions:
                return self._reply({"ok": False, "message": f"action budget exhausted ({self.max_actions})"})
            self.actions += 1
        res = self.sk.call(skill, args or {})
        self.events.append({"kind": "act", "skill": skill, "args": args or {}, "ok": res.ok,
                            "t_sim": round(env.t, 3), "wall_s": round(time.time() - t0, 3)})
        if env.running and self.actions >= self.max_actions and skill not in OBSERVATION_SKILLS:
            self._end("incomplete", f"robot action budget ({self.max_actions}) exhausted")
        self._maybe_write_ledger()
        out = res.to_dict()
        if skill == "list_objects" and res.ok:
            out["message"] = "see data.objects"
        return self._reply(out)

    def look(self, cameras: list[str] | None = None, width: int | None = None,
             height: int | None = None, depth: bool = False) -> dict[str, Any]:
        """Camera images (JPEG) with each camera's calibration at this instant (pose + intrinsics, see
        ``Env.camera_calibration``); with ``depth`` also depth maps (16-bit PNG, millimetres along the optical axis)."""
        return self.look_begin(cameras, width, height, depth)()

    def look_begin(self, cameras: list[str] | None = None, width: int | None = None,
                   height: int | None = None, depth: bool = False) -> Callable[[], dict[str, Any]]:
        """:meth:`look` in two halves: this one (under the lock) photographs the scene now; the returned function
        renders and encodes the images and needs no lock, so a real-time episode keeps running meanwhile."""
        err = self._observe_budget()
        if err:
            return lambda: {"ok": False, "message": err}
        cams = [c for c in (cameras or self.env.obs_cameras)]
        bad = [c for c in cams if c not in self.env.obs_cameras]
        if bad:
            return lambda: {"ok": False, "message": f"unknown camera(s) {bad}; available: "
                                                    f"{list(self.env.obs_cameras)}"}
        env = self.env
        w = int(min(max(width or self.width, 64), 1280))
        h = int(min(max(height or self.height, 48), 960))
        rgb = {c: env.capture(c, w, h) for c in cams}
        dep = {c: env.capture(c, w, h, depth=True) for c in cams} if depth else {}
        out = {"ok": True, "t_sim": round(env.t, 3), "calibration": {c: env.camera_calibration(c, w, h) for c in cams}}
        self.events.append({"kind": "look", "cameras": cams, "depth": bool(depth), "t_sim": round(env.t, 3)})

        def finish() -> dict[str, Any]:
            from roboticsbench.agents.llm.providers import encode_jpeg

            out["images"] = {c: base64.b64encode(encode_jpeg(f.result())).decode() for c, f in rgb.items()}
            if dep:
                out["depth_png_mm"] = {c: _depth_png(f.result()) for c, f in dep.items()}
            return out

        return finish

    def done(self) -> dict[str, Any]:
        env = self.env
        if self.motor is not None and env.running:
            self.motor.command_log.append({"tick": int(env.ticks), "done": True})
        if self.realtime and env.running:
            # same settle as SkillAPI.done (replays match), but on the free-running clock
            self.sk.declared_done = True
            settle = max(env.task.success_hold, 0.2) + 0.1
            self.done_at_tick = int(env.ticks) + max(1, int(round(settle / env.control_dt)))
            self.start_clock()
            while env.running and self._runner is not None and self._runner.is_alive():
                self.tick_cond.wait(timeout=2.0)
            self.events.append({"kind": "done", "t_sim": round(env.t, 3)})
            self._maybe_write_ledger()
            return self._reply({"ok": env.status == "success", "message": f"episode finished: {env.status}" +
                                (f" ({env.fail_reason})" if env.fail_reason else "")})
        res = self.sk.call("done", {}) if self.env.running else None
        self.events.append({"kind": "done", "t_sim": round(self.env.t, 3)})
        self._maybe_write_ledger()
        if res is None:
            return self._reply({"ok": False, "message": f"episode already ended ({self.env.status})"})
        return self._reply(res.to_dict())

    def _reply(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"result": result, "state": self.state()}

    # ------------------------------------------------------------------ operator
    def _end(self, status: str, reason: str) -> None:
        env = self.env
        if env.running and getattr(self, "end_tick", None) is None:
            self.end_tick = int(env.ticks)  # recorded for replay verification
        if env.running:  # settle briefly (a success may complete), then close the episode
            env.run_for(env.task.success_hold + 0.2)
            if env.running:
                env.end(status, reason)

    def finalize(self, reason: str = "agent stopped before completing the task") -> dict[str, Any]:
        self._stop_runner = True
        r = self._runner
        if r is not None and r is not threading.current_thread():
            r.join(timeout=10)
        with self.lock:
            self._end("incomplete", reason)
            self._maybe_write_ledger()
            self.tick_cond.notify_all()  # streams see the end
        return self.result or {}

    # ------------------------------------------------------------------ real-time clock
    def start_clock(self) -> None:
        """Real-time mode: start the free-running simulation (operator start, or the agent's first robot call)."""
        if self.realtime and self._runner is None and self.env.running and not self._stop_runner:
            self._runner = threading.Thread(target=self._run, name="rbench-realtime", daemon=True)
            self._runner.start()

    def _run(self) -> None:
        env, dt = self.env, self.env.control_dt
        wall0, n = time.time(), 0
        while True:
            with self.lock:
                if self._stop_runner or not env.running:
                    self.tick_cond.notify_all()
                    return
                env.tick(1)
                n += 1
                if env.running and time.time() - wall0 > env.task.time_limit + 5.0:  # backstop if physics lagged
                    env.end("timeout", "time limit reached (wall clock)")
                if self.done_at_tick is not None and env.running and env.ticks >= self.done_at_tick:
                    env.end("failure", "agent declared done but the task is not complete")
                if not env.running:
                    self._maybe_write_ledger()
                self.tick_cond.notify_all()
            lag = time.time() - (wall0 + n * dt)
            if lag < 0:
                time.sleep(-lag)  # keep 1x real time
            else:
                self.rt_max_lag = max(self.rt_max_lag, lag)  # the physics could not keep up for a moment

    def _wait_ticks(self, n: int) -> None:
        """Block (lock released) until the free-running clock has advanced ``n`` ticks or the episode ended."""
        self.start_clock()
        target = self.env.ticks + n
        while self.env.running and self.env.ticks < target and self._runner is not None and self._runner.is_alive():
            self.tick_cond.wait(timeout=2.0)

    def _maybe_write_ledger(self) -> None:
        if self.env.running or self.result is not None:
            return
        env = self.env
        if self.video:
            env.stop_video()
        res = env.result()
        res.update({
            "episode_id": f"{self.task_name.replace('/', '__')}__s{self.seed}",
            "category": env.task.category, "difficulty": env.task.difficulty, "robot": env.task.robot,
            "score": 1.0 if res["success"] else float(res["progress"]),
            "n_actions": self.actions, "n_skill_calls": len(self.sk.log), "n_observations": self.observations,
            "max_actions": self.max_actions, "declared_done": self.sk.declared_done,
            "variant": self.variant, "control": self.control, "vision_only": self.vision_only,
            "n_step_calls": self.step_calls,
            "n_motor_commands": len(self.motor.command_log) if self.motor is not None else 0,
            "referee_wall_s": round(time.time() - self.wall0, 2), "scorer_revision": SCORER_REVISION,
            "video": "video.mp4" if self.video else None,
            "realtime": self.realtime, "rt_max_lag_s": round(self.rt_max_lag, 3),
        })
        from roboticsbench.scrub import scrub_obj, secret_values

        secrets = secret_values()
        res, _ = scrub_obj(res, secrets)
        trace = {"task": self.task_name, "seed": self.seed, "instruction": env.task.instruction, "result": res,
                 "control": self.control, "skill_log": self.sk.log, "events": self.events, "tools": self.tools,
                 "final_tick": int(env.ticks), "end_tick": getattr(self, "end_tick", None)}
        if self.motor is not None:
            trace["motor_commands"] = self.motor.command_log
            trace["motor_events"] = self.motor.events
        trace, _ = scrub_obj(trace, secrets)  # agent-supplied strings (commands, args) end up in here
        tbytes = json.dumps(trace, indent=1, default=_default).encode()
        _atomic_write(self.ledger / "referee_trace.json", tbytes)
        res["trace_sha256"] = hashlib.sha256(tbytes).hexdigest()
        from roboticsbench.eval.runner import sign

        secret = os.environ.get("RBENCH_LEDGER_SECRET", "").encode() or None
        res["sig"] = sign(res, secret)
        _atomic_write(self.ledger / "referee_result.json", json.dumps(res, indent=1, default=_default).encode())
        self.result = res


def _depth_png(depth_m: Any) -> str:
    """Depth in metres -> base64 16-bit PNG in millimetres."""
    import io

    import numpy as np
    from PIL import Image

    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(mm).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def make_handler(ref: Referee, operator_token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rbench-referee"

        def log_message(self, fmt: str, *args: Any) -> None:
            if os.environ.get("RBENCH_REFEREE_VERBOSE"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, default=_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream(self) -> None:
            """One JSON line with the /state body per control tick until the episode ends or the client leaves."""
            if ref.motor is None:
                return self._send(404, {"error": "/stream is for the motor control modes"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    with ref.lock:
                        while last is not None and ref.env.ticks == last and ref.env.running:
                            ref.tick_cond.wait(timeout=1.0)
                        last = ref.env.ticks
                        body = ref.state()
                        running = ref.env.running
                    self.wfile.write(json.dumps(body, default=_default).encode() + b"\n")
                    self.wfile.flush()
                    if not running:
                        return
            except OSError:  # the client went away
                return

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n)) if n else {}

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                return self._send(200, {"ok": True, "episode_status": ref.env.status})
            if path == "/info":
                with ref.lock:  # does not start the real-time clock
                    body = ref.info()
                return self._send(200, body)
            if path == "/stream":
                return self._stream()
            if path == "/state":
                with ref.lock:
                    ref.start_clock()
                    err = ref._observe_budget()
                    body = ref.state() if not err else {"error": err}
                return self._send(200 if not err else 429, body)
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._body()
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"invalid JSON: {e}"})
            if path.startswith("/operator/"):
                if not operator_token or self.headers.get("X-Operator-Token") != operator_token:
                    return self._send(403, {"error": "forbidden"})
                if path == "/operator/start":
                    with ref.lock:
                        ref.start_clock()
                    return self._send(200, {"ok": True, "realtime": ref.realtime, "t_sim": round(ref.env.t, 3)})
                if path == "/operator/finalize":
                    return self._send(200, ref.finalize(body.get("reason") or "agent stopped before completing "
                                                                             "the task"))
                if path == "/operator/shutdown":
                    self._send(200, {"ok": True})
                    self.server.shutting_down = True  # type: ignore[attr-defined]
                    return None
                return self._send(404, {"error": "not found"})
            if path == "/look":  # photographed under the lock, rendered + encoded outside it
                with ref.lock:
                    ref.start_clock()
                    finish = ref.look_begin(body.get("cameras"), body.get("width"), body.get("height"),
                                            bool(body.get("depth")))
                return self._send(200, finish())
            calls = {
                "/act": lambda: ref.act(str(body.get("skill", "")), body.get("args") or {}),
                "/step": lambda: ref.step(body.get("command"), body.get("ticks", 1)),
                "/objects": ref.objects,
                "/done": ref.done,
            }
            if path in calls:
                with ref.lock:  # the real-time clock thread steps the physics under the same lock
                    ref.start_clock()
                    out = calls[path]()
                return self._send(200, out)
            self._send(404, {"error": "not found"})

    return Handler


def serve(ref: Referee, host: str, port: int, operator_token: str) -> None:
    # one thread per request: in real-time mode a /step or /done waits for the clock (lock released) while e.g. a
    # second agent process asks for camera images, and the operator can always reach /operator/finalize
    srv = ThreadingHTTPServer((host, port), make_handler(ref, operator_token))
    srv.daemon_threads = True
    srv.timeout = 0.5  # handle_request() returns regularly, so a shutdown request is noticed
    srv.shutting_down = False  # type: ignore[attr-defined]
    print(json.dumps({"referee": f"http://{host}:{srv.server_address[1]}", "task": ref.task_name, "seed": ref.seed}),
          flush=True)
    try:
        while not srv.shutting_down:  # type: ignore[attr-defined]
            srv.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        ref.finalize("referee shut down")
        ref.env.close()
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--ledger", required=True, help="directory for referee_result.json / referee_trace.json / video")
    ap.add_argument("--max-actions", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--vision", action="store_true", help="vision-only variant: no list_objects()")
    ap.add_argument("--control", default="skills", choices=["skills", "motor", "velocity"])
    ap.add_argument("--live-port", type=int, default=0, help="serve a live view of the episode on this local port")
    ap.add_argument("--live-title", default="")
    ap.add_argument("--realtime", action="store_true",
                    help="free-running real-time simulation from /operator/start or the agent's first robot call "
                         "(motor modes)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="episode time budget in seconds, replacing the task's own limit (0 = the task's)")
    a = ap.parse_args(argv)
    token = os.environ.pop("RBENCH_OPERATOR_TOKEN", "")
    ref = Referee(a.task, a.seed, a.ledger, max_actions=a.max_actions or None, video=a.video, width=a.width,
                  height=a.height, vision_only=a.vision, control=a.control, live_port=a.live_port,
                  live_title=a.live_title, realtime=a.realtime, time_limit=a.time_limit or None)
    serve(ref, a.host, a.port, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
