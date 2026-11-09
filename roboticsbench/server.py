"""Minimal HTTP/JSON API so external agents (any language / framework) can run RoboticsBench episodes.

    rbench serve --port 8765        (binds 127.0.0.1 by default; set RBENCH_SERVER_TOKEN to require a bearer token)

Endpoints (all JSON):

    GET    /tasks                                   -> [{name, robot, category, difficulty, summary, ...}]
    POST   /episodes {task, seed, images?, cameras?, width?, height?}
                                                     -> {episode_id, instruction, tools, observation}
    GET    /episodes/{id}?images=1                  -> {observation}
    POST   /episodes/{id}/call {name, args, images?} -> {result: {ok, message, data?}, observation}
    POST   /episodes/{id}/done                      -> {result} (declares completion, then scores)
    GET    /episodes/{id}/result                    -> final/partial score record
    DELETE /episodes/{id}                           -> final score record (and frees the episode)

``tools`` are JSON-schema tool definitions (the same ones the built-in LLM agent gets). Observations contain the
simulated time, robot proprioception, public task status and, when ``images`` is true, base64 JPEG camera images.
Simulated time only advances inside tool calls, so the world waits while the client thinks.

The server is single-threaded on purpose (MuJoCo rendering contexts, determinism); run several servers on
different ports for parallel episodes.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from roboticsbench.core.env import Env
from roboticsbench.skills import make_skills

MAX_EPISODES = 16


class Episode:
    def __init__(self, task: str, seed: int, cameras: list[str] | None, width: int, height: int):
        self.env = Env(task, seed=seed)
        self.sk = make_skills(self.env)
        self.cameras = tuple(cameras or self.env.obs_cameras)
        self.width, self.height = width, height

    def observation(self, images: bool) -> dict[str, Any]:
        from roboticsbench.agents.llm.providers import encode_jpeg

        env = self.env
        obs = env.observe(images=False)
        obs["actions_used"] = len(self.sk.log)
        obs["max_actions"] = env.task.max_actions
        if images:
            obs["images"] = {c: base64.b64encode(encode_jpeg(env.render(c, self.width, self.height))).decode()
                             for c in self.cameras}
            obs["image_format"] = "jpeg/base64"
        return obs

    def result(self, final: bool = False) -> dict[str, Any]:
        env = self.env
        if final and env.running:  # same rule as the batch runner: settle briefly, then score as incomplete
            env.run_for(env.task.success_hold + 0.2)
            if env.running:
                env.end("incomplete", "client closed the episode before completing the task")
        r = env.result()
        r["score"] = 1.0 if r["success"] else r["progress"]
        r["n_actions"] = len(self.sk.log)
        return r


class RBHandler(BaseHTTPRequestHandler):
    episodes: dict[str, Episode] = {}
    token = os.environ.get("RBENCH_SERVER_TOKEN", "")

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        if os.environ.get("RBENCH_SERVER_VERBOSE"):
            super().log_message(fmt, *args)

    # -------------------------------------------------------------- plumbing
    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body, default=_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n))

    def _auth(self) -> bool:
        if not self.token:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {self.token}":
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def _episode(self, eid: str) -> Episode | None:
        ep = self.episodes.get(eid)
        if ep is None:
            self._send(404, {"error": f"unknown episode {eid}"})
        return ep

    # --------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802
        if not self._auth():
            return
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/tasks":
            from roboticsbench.tasks import TASKS
            from roboticsbench.tasks.base import SHELVED

            return self._send(200, [{"name": c.name, "robot": c.robot, "category": c.category,
                                     "difficulty": c.difficulty, "summary": c.summary, "time_limit": c.time_limit,
                                     "max_actions": c.max_actions, "vision_only": not c.allow_object_list,
                                     "benchmark": not (n in SHELVED or n.startswith("dev/"))}
                                    for n, c in sorted(TASKS.items())])
        m = re.fullmatch(r"/episodes/([\w-]+)(/result)?", url.path)
        if m:
            ep = self._episode(m.group(1))
            if ep is None:
                return None
            if m.group(2):
                return self._send(200, ep.result())
            return self._send(200, {"observation": ep.observation(q.get("images", ["0"])[0] in ("1", "true"))})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth():
            return
        url = urlparse(self.path)
        try:
            body = self._body()
        except json.JSONDecodeError as e:
            return self._send(400, {"error": f"invalid JSON: {e}"})
        if url.path == "/episodes":
            if len(self.episodes) >= MAX_EPISODES:
                return self._send(429, {"error": "too many open episodes; DELETE finished ones"})
            try:
                ep = Episode(body["task"], int(body.get("seed", 0)), body.get("cameras"),
                             int(body.get("width", 640)), int(body.get("height", 480)))
            except KeyError as e:
                return self._send(400, {"error": f"bad request: {e}"})
            eid = uuid.uuid4().hex[:12]
            self.episodes[eid] = ep
            return self._send(200, {"episode_id": eid, "instruction": ep.env.task.instruction,
                                    "robot": ep.env.robot.name, "cameras": list(ep.cameras),
                                    "time_limit": ep.env.task.time_limit, "max_actions": ep.env.task.max_actions,
                                    "tools": ep.sk.tools(),
                                    "observation": ep.observation(bool(body.get("images", True)))})
        m = re.fullmatch(r"/episodes/([\w-]+)/(call|done)", url.path)
        if m:
            ep = self._episode(m.group(1))
            if ep is None:
                return None
            name = "done" if m.group(2) == "done" else body.get("name", "")
            res = ep.sk.call(name, body.get("args") or {})
            return self._send(200, {"result": res.to_dict(),
                                    "observation": ep.observation(bool(body.get("images", True))),
                                    "episode_status": ep.env.status})
        self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._auth():
            return
        m = re.fullmatch(r"/episodes/([\w-]+)", urlparse(self.path).path)
        if not m:
            return self._send(404, {"error": "not found"})
        ep = self.episodes.pop(m.group(1), None)
        if ep is None:
            return self._send(404, {"error": "unknown episode"})
        res = ep.result(final=True)
        ep.env.close()
        self._send(200, res)


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


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = HTTPServer((host, port), RBHandler)
    print(f"RoboticsBench server on http://{host}:{port}  (token {'required' if RBHandler.token else 'off'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
