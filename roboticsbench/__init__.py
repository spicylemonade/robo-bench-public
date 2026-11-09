"""RoboticsBench: physics-accurate humanoid + arm tasks for LLM agents and robot policies."""

from __future__ import annotations

__version__ = "0.1.0"


def make(task: str, seed: int = 0, **kwargs):
    """Create an :class:`~roboticsbench.core.env.Env` for ``task``."""
    from roboticsbench.core.env import Env

    return Env(task, seed=seed, **kwargs)


def list_tasks(robot: str | None = None) -> list[str]:
    from roboticsbench.tasks.base import list_tasks as _lt

    return _lt(robot)
