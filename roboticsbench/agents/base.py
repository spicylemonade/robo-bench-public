"""Agent interface.

An agent drives one episode: ``run(env, sk)`` issues skill calls (or low-level actions) until the
episode ends or the agent gives up. The runner evaluates the final state afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from roboticsbench.core.env import Env
    from roboticsbench.skills.base import SkillAPI


@dataclass
class AgentTrace:
    """Agent-side record of an episode (the env/skill logs hold the physical side)."""

    agent: str
    model: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)  # one entry per model request
    notes: list[str] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, float] = field(default_factory=dict)

    def add_usage(self, **kw: float) -> None:
        for k, v in kw.items():
            if v:
                self.usage[k] = self.usage.get(k, 0) + v


class Agent:
    name = "agent"
    model: str | None = None

    def run(self, env: "Env", sk: "SkillAPI") -> AgentTrace:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"agent": self.name, "model": self.model}
