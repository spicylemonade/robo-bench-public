"""Skill API: the high-level, tool-callable action interface (used by LLM agents, coding agents
and the scripted oracles). Every skill runs closed-loop controllers on the live physics for
some simulated time and returns a :class:`SkillResult`."""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from roboticsbench.core.env import Env


@dataclass
class SkillResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message, **({"data": self.data} if self.data else {})}


def skill(description: str, **params: dict[str, Any]) -> Callable:
    """Mark a method as an agent-callable skill. ``params`` maps argument name -> JSON-schema
    fragment (add ``"optional": True`` for optional args; defaults come from the signature)."""

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            if pname in ("self", "sk"):  # task-defined skills take (self, sk, ...)
                continue
            spec = dict(params.get(pname, {"type": "number"}))
            optional = spec.pop("optional", p.default is not inspect.Parameter.empty)
            if p.default is not inspect.Parameter.empty and p.default is not None:
                spec.setdefault("default", p.default)
            props[pname] = spec
            if not optional:
                required.append(pname)
        fn._skill = {"name": fn.__name__, "description": description,
                     "parameters": {"type": "object", "properties": props, "required": required}}
        return fn

    return deco


class SkillError(Exception):
    pass


class SkillAPI:
    """Base class. Subclasses add robot-specific skills; tasks may add extra skills by name."""

    robot_kind = "robot"

    def __init__(self, env: "Env", *, allow_object_list: bool | None = None, allow_done: bool = True):
        self.env = env
        self.robot = env.robot
        self.log: list[dict[str, Any]] = []
        self.allow_object_list = env.task.allow_object_list if allow_object_list is None else allow_object_list
        self.allow_done = allow_done
        self.declared_done = False

    # ------------------------------------------------------------ registry
    def _skill_methods(self) -> dict[str, Callable]:
        out = {}
        for name in dir(type(self)):
            fn = getattr(type(self), name, None)
            if callable(fn) and hasattr(fn, "_skill"):
                out[name] = getattr(self, name)
        for name in self.env.task.extra_skills():
            fn = getattr(self.env.task, name, None)
            if fn is not None and hasattr(fn, "_skill"):
                wrapped = (lambda f: (lambda **kw: f(self, **kw)))(fn)
                wrapped._skill = fn._skill  # type: ignore[attr-defined]
                out[name] = wrapped
        if not self.allow_object_list:
            out.pop("list_objects", None)
        if not self.allow_done:
            out.pop("done", None)
        return out

    def tools(self) -> list[dict[str, Any]]:
        return [dict(m._skill) for m in self._skill_methods().values()]

    def check(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        """Validation error for a call (unknown skill / bad arguments), or None if it would run."""
        methods = self._skill_methods()
        if name not in methods:
            return f"unknown skill {name!r}. Available: {', '.join(sorted(methods))}"
        try:
            _coerce_args(dict(args or {}), methods[name]._skill["parameters"])
        except (TypeError, ValueError) as e:
            return f"bad arguments for {name}: {e}"
        return None

    def call(self, name: str, args: dict[str, Any] | None = None) -> SkillResult:
        """Validated, logged dispatch (what agents go through)."""
        args = dict(args or {})
        methods = self._skill_methods()
        env = self.env
        t_start, wall = env.t, time.time()
        if name not in methods:
            res = SkillResult(False, f"unknown skill {name!r}. Available: {', '.join(sorted(methods))}")
        elif not env.running and name not in ("list_objects", "look"):  # read-only calls still answer
            res = SkillResult(False, f"episode already ended ({env.status}: {env.fail_reason})")
        else:
            spec = methods[name]._skill["parameters"]
            try:
                clean = _coerce_args(args, spec)
                env.caption = f"{name}({', '.join(f'{k}={_fmt(v)}' for k, v in clean.items())})"
                res = methods[name](**clean)
            except SkillError as e:
                res = SkillResult(False, str(e))
            except (TypeError, ValueError) as e:
                res = SkillResult(False, f"bad arguments for {name}: {e}")
            if env.status == "success" and not res.ok:  # e.g. the goal was reached mid-motion
                res = SkillResult(True, res.message, res.data)
        entry = {"skill": name, "args": args, "ok": res.ok, "message": res.message,
                 "t_start": round(t_start, 3), "t_end": round(env.t, 3), "wall_s": round(time.time() - wall, 3)}
        self.log.append(entry)
        return res

    # ---------------------------------------------------------- helpers
    def _outcome_suffix(self) -> str:
        env = self.env
        if env.status == "running":
            return ""
        return f" [EPISODE ENDED: {env.status}{(' - ' + env.fail_reason) if env.fail_reason else ''}]"

    def _wait(self, seconds: float) -> None:
        self.env.run_for(seconds)

    # ------------------------------------------------------- common skills
    @skill("Let simulated time pass without commanding anything new (robot keeps balancing / holding).",
           seconds={"type": "number", "description": "0.1 - 10 seconds"})
    def wait(self, seconds: float) -> SkillResult:
        seconds = float(np.clip(seconds, 0.02, 10.0))
        self._wait(seconds)
        return SkillResult(True, f"waited {seconds:.2f}s" + self._outcome_suffix())

    @skill("Return the current camera images and robot/task state without moving.")
    def look(self) -> SkillResult:
        return SkillResult(True, "observation captured")

    @skill("List the task-relevant objects with their exact 3D positions (simulated perfect perception).")
    def list_objects(self) -> SkillResult:
        from roboticsbench.core.jsonutil import jsonable

        objs = jsonable(self.env.task.objects(self.env))
        return SkillResult(True, json.dumps(objs), {"objects": objs})

    @skill("Declare that you believe the task is complete. Ends the episode after a short settle.")
    def done(self) -> SkillResult:
        env = self.env
        self.declared_done = True
        env.run_for(max(env.task.success_hold, 0.2) + 0.1)
        if env.running:
            env.end("failure", "agent declared done but the task is not complete")
        return SkillResult(env.status == "success", f"episode finished: {env.status}" +
                           (f" ({env.fail_reason})" if env.fail_reason else ""))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _coerce_args(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties", {})
    out = {}
    unknown = set(args) - set(props)
    if unknown:
        raise ValueError(f"unexpected argument(s): {sorted(unknown)}")
    for k in schema.get("required", []):
        if k not in args or args[k] is None:
            raise ValueError(f"missing required argument {k!r}")
    for k, v in args.items():
        if v is None:
            continue
        t = props[k].get("type")
        if t == "number":
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
        elif t == "integer":
            v = int(v)
        elif t == "boolean":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
        elif t == "string":
            v = str(v)
            enum = props[k].get("enum")
            if enum and v not in enum:
                raise ValueError(f"{k} must be one of {enum}")
        elif t == "array":
            v = list(v)
        out[k] = v
    return out
