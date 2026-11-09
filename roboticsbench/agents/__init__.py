"""Agent factory.

Specs:
    oracle | null | random
    llm:<provider>/<model>            e.g. llm:anthropic/claude-opus-5, llm:openrouter/google/gemini-3-pro
    policy:<module>:<Class>           a custom low-level policy (see roboticsbench.agents.policy)
    remote:<ws-or-http-url>           a remote policy server (see roboticsbench.agents.policy)
"""

from __future__ import annotations

from typing import Any

from roboticsbench.agents.base import Agent, AgentTrace


def make_agent(spec: str, **kw: Any) -> Agent:
    if spec == "oracle":
        from roboticsbench.agents.scripted import OracleAgent

        return OracleAgent()
    if spec == "null":
        from roboticsbench.agents.scripted import NullAgent

        return NullAgent()
    if spec == "random":
        from roboticsbench.agents.scripted import RandomAgent

        return RandomAgent(**kw)
    if spec.startswith("llm:"):
        from roboticsbench.agents.llm.agent import LLMAgent

        return LLMAgent(spec[4:], **kw)
    if spec.startswith("policy:"):
        from roboticsbench.agents.policy import load_policy_agent

        return load_policy_agent(spec[7:], **kw)
    if spec.startswith("remote:"):
        from roboticsbench.agents.policy import RemotePolicyAgent

        return RemotePolicyAgent(spec[7:], **kw)
    raise KeyError(f"unknown agent spec {spec!r}")


__all__ = ["Agent", "AgentTrace", "make_agent"]
