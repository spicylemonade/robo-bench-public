"""Robot interface used by :class:`roboticsbench.core.env.Env`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mujoco
import numpy as np


class Robot(ABC):
    """A robot owns its MJCF, its low-level controllers and its proprioceptive state.

    Life-cycle: ``build_spec`` -> (task adds props) -> compile -> ``bind`` -> ``reset`` ->
    for every control tick: ``pre_tick`` once, then ``substep`` before every ``mj_step``.
    """

    name: str = "robot"
    physics_dt: float = 0.002
    control_dt: float = 0.02
    default_cameras: tuple[str, ...] = ()
    video_camera: str = ""
    # the only cameras an agent may use (None = any camera the task offers); the others stay for the video / live view
    agent_cameras: tuple[str, ...] | None = None

    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

    @abstractmethod
    def build_spec(self) -> mujoco.MjSpec:
        """Return a fresh spec containing the robot (and robot-mounted sites/cameras)."""

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.body_ids = self._subtree_bodies()
        self.geom_ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] in self.body_ids}

    @abstractmethod
    def reset(self, rng: np.random.Generator, **pose: Any) -> None:
        """Write the initial robot state into ``self.data`` (after ``mj_resetData``)."""

    def pre_tick(self) -> None:
        """Once per control tick (policy inference, command smoothing)."""

    def substep(self) -> None:
        """Before every physics step (PD torque computation)."""

    @abstractmethod
    def proprio(self) -> dict[str, Any]:
        """Proprioceptive state (JSON-serialisable)."""

    def is_fallen(self) -> bool:
        return False

    # ---------------------------------------------------------------- helpers
    @property
    def root_body(self) -> str:
        raise NotImplementedError

    def _subtree_bodies(self) -> set[int]:
        m = self.model
        root = m.body(self.root_body).id
        out = {root}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = m.body_parentid[p]
        return out
