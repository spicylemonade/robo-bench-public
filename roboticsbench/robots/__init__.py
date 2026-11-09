from __future__ import annotations

from roboticsbench.robots.base import Robot


def make_robot(name: str, **kwargs) -> Robot:
    if name == "g1":
        from roboticsbench.robots.g1 import G1

        return G1(**kwargs)
    if name == "panda":
        from roboticsbench.robots.panda import Panda

        return Panda(**kwargs)
    raise KeyError(f"unknown robot {name!r}")


__all__ = ["Robot", "make_robot"]
