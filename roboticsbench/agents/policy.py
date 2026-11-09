"""Custom robot policies (VLAs, RL controllers, ...) acting through the low-level interface.

Local policy: any object with

    action_space: str            # e.g. "ee_delta" (Panda) or "wbc" (G1)
    control_hz: float            # decision rate
    cameras: tuple[str, ...]     # optional, images the policy wants
    def reset(self, info: dict) -> None          # info = {"instruction", "robot", "action_space", "state_layout"}
    def act(self, obs: dict) -> np.ndarray       # one action (dim,) or an action chunk (k, dim)

Load with ``policy:<module.path>:<ClassName>`` (kwargs forwarded).

Remote policy server (``remote:<url>``): HTTP JSON protocol, so any model can be served from anywhere (e.g. a GPU
box or a cloud container):

    POST {url}/reset  {"instruction", "robot", "action_space", "state_layout", "control_hz"}  -> {"ok": true}
    POST {url}/act    {"state": [...], "images": {cam: base64-jpeg}, "instruction", "time"}   -> {"actions": [[...], ...]}
"""

from __future__ import annotations

import base64
import importlib
import json
import urllib.request
from typing import Any

import numpy as np

from roboticsbench.agents.base import Agent, AgentTrace
from roboticsbench.core.lowlevel import LowLevelControl


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self, policy: Any, *, image_size: tuple[int, int] = (256, 256), privileged: bool = False,
                 max_steps: int | None = None):
        self.policy = policy
        self.model = type(policy).__name__
        self.image_size = image_size
        self.privileged = privileged
        self.max_steps = max_steps

    def run(self, env, sk) -> AgentTrace:
        pol = self.policy
        ctl = LowLevelControl(env, getattr(pol, "action_space", None), getattr(pol, "control_hz", 10.0),
                              cameras=getattr(pol, "cameras", None), image_size=self.image_size,
                              privileged=self.privileged)
        pol.reset({"instruction": env.task.instruction, "robot": env.robot.name, "action_space": ctl.space,
                   "state_layout": ctl.state_layout(), "control_hz": ctl.control_hz})
        tr = AgentTrace(agent=self.name, model=self.model)
        obs = ctl.observe()
        steps = 0
        while env.running and (self.max_steps is None or steps < self.max_steps):
            act = np.asarray(pol.act(obs), float)
            chunk = act[None] if act.ndim == 1 else act
            for a in chunk:
                res = ctl.step(a)
                steps += 1
                obs = res.obs
                if res.terminated or res.truncated:
                    break
        tr.stop_reason = f"episode ended ({env.status})" if not env.running else "step budget"
        tr.usage["policy_steps"] = steps
        return tr


def load_policy_agent(spec: str, **kw: Any) -> PolicyAgent:
    mod, _, cls = spec.partition(":")
    policy_cls = getattr(importlib.import_module(mod), cls)
    agent_kw = {k: kw.pop(k) for k in ("image_size", "privileged", "max_steps") if k in kw}
    return PolicyAgent(policy_cls(**kw), **agent_kw)


class RemotePolicy:
    """Client for the HTTP policy protocol above."""

    def __init__(self, url: str, action_space: str | None = None, control_hz: float = 10.0,
                 cameras: tuple[str, ...] | None = None, timeout: float = 120.0, headers: dict | None = None):
        self.url = url.rstrip("/")
        self.action_space = action_space
        self.control_hz = control_hz
        self.cameras = cameras
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def reset(self, info: dict) -> None:
        self._post("/reset", info)

    def act(self, obs: dict) -> np.ndarray:
        from roboticsbench.agents.llm.providers import encode_jpeg

        body = {"state": obs["state"].tolist(), "instruction": obs["instruction"], "time": obs["time"],
                "images": {c: base64.b64encode(encode_jpeg(im)).decode() for c, im in obs["images"].items()}}
        return np.asarray(self._post("/act", body)["actions"], float)


class RemotePolicyAgent(PolicyAgent):
    name = "remote"

    def __init__(self, url: str, **kw: Any):
        pol_kw = {k: kw.pop(k) for k in ("action_space", "control_hz", "cameras", "timeout", "headers") if k in kw}
        super().__init__(RemotePolicy(url, **pol_kw), **kw)
        self.model = url
