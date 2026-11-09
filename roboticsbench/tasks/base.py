"""Task definition API + registry.

A task builds its props into the scene, places them on reset, monitors the episode every
control tick (milestones, rule violations) and reports an :class:`Evaluation`. Each task also
ships an ``oracle`` -- a scripted reference solution that uses *only the public skill API* --
which is how we prove every task is solvable through the same interface agents get.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from roboticsbench.core.scene import SceneBuilder

if TYPE_CHECKING:  # pragma: no cover
    from roboticsbench.core.env import Env
    from roboticsbench.robots.base import Robot


@dataclass
class Evaluation:
    success: bool = False
    progress: float = 0.0  # partial credit in [0, 1]
    failed: bool = False  # unrecoverable failure (fell, broke a rule, ...)
    reason: str = ""
    milestones: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class Milestones:
    """Latched, ordered partial-credit ladder."""

    def __init__(self, ladder: list[tuple[str, float]]):
        self.ladder = ladder
        self.credit = {n: c for n, c in ladder}
        self.reached: dict[str, float] = {}

    def reset(self) -> None:
        self.reached.clear()

    def hit(self, name: str, t: float) -> None:
        if name not in self.reached:
            self.reached[name] = round(float(t), 3)

    def has(self, name: str) -> bool:
        return name in self.reached

    @property
    def progress(self) -> float:
        return max([self.credit[n] for n in self.reached] or [0.0])

    def names(self) -> list[str]:
        return [n for n, _ in self.ladder if n in self.reached]


class Task:
    # identity / metadata -------------------------------------------------------
    name: ClassVar[str] = "base"
    robot: ClassVar[str] = "panda"
    category: ClassVar[str] = "misc"
    difficulty: ClassVar[str] = "medium"  # easy | medium | hard | expert
    summary: ClassVar[str] = ""
    time_limit: ClassVar[float] = 120.0  # simulated seconds
    max_actions: ClassVar[int] = 60  # suggested tool-call budget for LLM agents
    success_hold: ClassVar[float] = 0.5  # seconds the success condition must hold
    # False: a fall is not a failure and a broken task rule only rules out success - the episode runs on to the time
    # limit (or a success) either way, and a broken rule is reported when it ends (humanoid tasks, 2026-10-09)
    ends_on_failure: ClassVar[bool] = True
    allow_object_list: ClassVar[bool] = True  # may agents call list_objects() (perfect perception)?
    world: ClassVar[dict[str, Any]] = {}
    cameras: ClassVar[tuple[str, ...]] = ()  # observation cameras (default: robot defaults)
    video_camera: ClassVar[str] = ""
    # control modes in which the task is solvable ("skills" = high-level skill API, "motor" = joint-level motor
    # commands, "velocity" = G1 walking policy + joint-level arms). Throwing, for example, is motor-only.
    controls: ClassVar[tuple[str, ...]] = ("skills", "motor", "velocity")
    # the reference solution's interface: "skills" -> oracle(sk: SkillAPI); "motor" -> oracle(api) with
    # roboticsbench.agents.motor_oracle.PandaMotorOracle (public motor interface + published kinematics only)
    oracle_control: ClassVar[str] = "skills"

    def __init__(self, **options: Any):
        self.options = options
        self.instruction = ""
        self.milestones = Milestones([])
        self.info: dict[str, Any] = {}

    # -- construction ------------------------------------------------------------
    def make_robot(self) -> "Robot":
        from roboticsbench.robots import make_robot

        return make_robot(self.robot)

    def build(self, scene: SceneBuilder, rng: np.random.Generator) -> None:
        """Add props / cameras. Randomise layout from ``rng`` (seeded per episode)."""

    def bind(self, env: "Env") -> None:
        """Resolve model ids after compilation."""

    def robot_start(self, rng: np.random.Generator) -> dict[str, Any]:
        return {}

    def reset(self, env: "Env", rng: np.random.Generator) -> None:
        """Place objects (write qpos). Called after the robot has been reset."""

    # -- monitoring ---------------------------------------------------------------
    def step(self, env: "Env") -> None:
        """Called every control tick after physics; update milestones / detect violations."""

    def evaluate(self, env: "Env") -> Evaluation:
        raise NotImplementedError

    # -- agent-facing ---------------------------------------------------------------
    def objects(self, env: "Env") -> list[dict[str, Any]]:
        """Perception output for ``list_objects`` (name, colour, shape, position, size)."""
        return []

    def status(self, env: "Env") -> dict[str, Any]:
        """Task-specific public state shown to the agent (never the score internals)."""
        return {}

    def allows_low_posture(self, env: "Env") -> bool:
        """True where the robot may be low or tilted without counting as fallen (e.g. on a crawl mat)."""
        return False

    def extra_skills(self) -> list[str]:
        """Names of task-specific skills (methods on the skill API) to expose."""
        return []

    def instruction_for(self, vision_only: bool = False) -> str:
        """The instruction shown to the agent; camera-only runs may get a variant without ground-truth positions."""
        return self.instruction

    def documents(self, env: "Env") -> dict[str, bytes]:
        """Files handed to the agent with the task (e.g. an illustrated assembly manual): relative path -> bytes.
        Coding agents find them in their workspace; the direct-API agent gets images attached to its first message."""
        return {}

    # -- reference solution ---------------------------------------------------------
    def oracle(self, sk: Any) -> None:  # pragma: no cover - implemented per task
        raise NotImplementedError


# ----------------------------------------------------------------------- helpers
def body_pos(env: "Env", name: str) -> np.ndarray:
    return env.data.body(name).xpos.copy()


def body_quat(env: "Env", name: str) -> np.ndarray:
    return env.data.body(name).xquat.copy()


def body_yaw_deg(env: "Env", name: str) -> float:
    R = env.data.body(name).xmat.reshape(3, 3)
    return round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 1)


def body_up_angle_deg(env: "Env", name: str) -> float:
    """Angle between the body's local z axis and world up."""
    R = env.data.body(name).xmat.reshape(3, 3)
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))


def set_free_pose(env: "Env", body: str, pos, quat=(1, 0, 0, 0)) -> None:
    m, d = env.model, env.data
    j = m.body(body).jntadr[0]
    adr = m.jnt_qposadr[j]
    d.qpos[adr:adr + 3] = pos
    d.qpos[adr + 3:adr + 7] = quat
    d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0


def obj(name: str, shape: str, color: str, pos, size, **extra: Any) -> dict[str, Any]:
    out = {"name": name, "shape": shape, "color": color,
           "position": [round(float(x), 3) for x in pos], "size_m": size}
    out.update(extra)
    return out


def xy_dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


TASKS: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    if cls.name in TASKS:
        raise ValueError(f"duplicate task {cls.name}")
    TASKS[cls.name] = cls
    return cls


def make_task(name: str, **options: Any) -> Task:
    import roboticsbench.tasks  # noqa: F401  (populate registry)

    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name](**options)


# Tasks registered but outside the benchmark set (none in this export).
SHELVED: frozenset[str] = frozenset()


def list_tasks(robot: str | None = None, include_dev: bool = False, include_shelved: bool = False) -> list[str]:
    """Benchmark tasks (``dev/*`` tutorial / smoke-test tasks only with ``include_dev``, the ``SHELVED`` ones only
    with ``include_shelved``)."""
    import roboticsbench.tasks  # noqa: F401

    return sorted(n for n, c in TASKS.items() if (robot is None or c.robot == robot)
                  and (include_dev or not n.startswith("dev/")) and (include_shelved or n not in SHELVED))
