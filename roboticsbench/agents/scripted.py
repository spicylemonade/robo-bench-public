"""Non-LLM reference agents: the task oracle, a do-nothing baseline and a random-skill baseline."""

from __future__ import annotations

import inspect
import json

import numpy as np

from roboticsbench.agents.base import Agent, AgentTrace


class LoggedSkills:
    """Proxy that routes direct skill-method calls (``sk.move_to(0.5, 0, 0.1)``) through ``sk.call`` so oracle
    episodes are logged, captioned and counted exactly like agent episodes."""

    def __init__(self, sk):
        self._sk = sk
        self._names = {t["name"] for t in sk.tools()}

    def __getattr__(self, name):
        attr = getattr(self._sk, name)
        if name not in self._names or not callable(attr):
            return attr
        sig = inspect.signature(attr)

        def call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            return self._sk.call(name, {k: v for k, v in bound.arguments.items() if v is not None})

        return call


class OracleAgent(Agent):
    """Runs the task's scripted reference solution (public skill API only)."""

    name = "oracle"

    def run(self, env, sk) -> AgentTrace:
        from roboticsbench.tasks.arm.oracle_utils import EpisodeOver

        tr = AgentTrace(agent=self.name)
        try:
            if env.task.oracle_control in ("motor", "velocity"):
                from roboticsbench.agents.motor_oracle import G1VelocityOracle, PandaMotorOracle
                from roboticsbench.core.motor import MotorInterface

                cls = PandaMotorOracle if env.robot.name == "panda" else G1VelocityOracle
                env.task.oracle(cls(env, MotorInterface(env, env.task.oracle_control), sk))
            else:
                env.task.oracle(LoggedSkills(sk))
            tr.stop_reason = "oracle finished"
        except EpisodeOver:
            tr.stop_reason = "episode ended"
        except Exception as e:  # noqa: BLE001 - recorded, the runner scores what happened
            tr.stop_reason = f"oracle error: {type(e).__name__}: {e}"
        return tr


class NullAgent(Agent):
    """Does nothing until the time limit (sanity baseline: must never succeed)."""

    name = "null"

    def run(self, env, sk) -> AgentTrace:
        while env.running:
            env.tick(50)
        return AgentTrace(agent=self.name, stop_reason="time limit")


class RandomAgent(Agent):
    """Calls random skills with random (bounded) arguments."""

    name = "random"

    def __init__(self, max_actions: int = 30, seed: int = 0):
        self.max_actions = max_actions
        self.seed = seed

    def run(self, env, sk) -> AgentTrace:
        rng = np.random.default_rng(self.seed)
        tools = [t for t in sk.tools() if t["name"] not in ("done", "look", "list_objects")]
        tr = AgentTrace(agent=self.name)
        for _ in range(self.max_actions):
            if not env.running:
                break
            t = tools[rng.integers(len(tools))]
            args = {}
            for k, spec in t["parameters"]["properties"].items():
                if "enum" in spec:
                    args[k] = str(rng.choice(spec["enum"]))
                elif spec.get("type") == "number":
                    args[k] = float(rng.uniform(-1.0, 1.0)) if "deg" not in k else float(rng.uniform(-90, 90))
                elif spec.get("type") == "integer":
                    args[k] = int(rng.integers(0, 5))
            if t["name"] == "wait":
                args["seconds"] = 1.0
            res = sk.call(t["name"], args)
            tr.notes.append(json.dumps({"skill": t["name"], "ok": res.ok}))
        tr.stop_reason = "action budget"
        return tr
