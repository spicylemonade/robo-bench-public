"""Importing this package registers every task module under tasks/humanoid and tasks/arm."""

from __future__ import annotations

import importlib
import pkgutil
import traceback
import warnings

from roboticsbench.tasks.base import TASKS, Evaluation, Task, list_tasks, make_task, register

_SKIP = {"common", "oracle_utils"}
LOAD_ERRORS: dict[str, str] = {}


def _load_all() -> None:
    for sub in ("humanoid", "arm", "dev"):
        pkg = importlib.import_module(f"roboticsbench.tasks.{sub}")
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name in _SKIP or mod.name.startswith("_"):
                continue
            name = f"roboticsbench.tasks.{sub}.{mod.name}"
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 - one broken task must not take down the benchmark
                LOAD_ERRORS[name] = traceback.format_exc(limit=3)
                warnings.warn(f"failed to load task module {name}", stacklevel=2)


_load_all()

__all__ = ["TASKS", "Evaluation", "Task", "list_tasks", "make_task", "register", "LOAD_ERRORS"]
