"""Named task suites. ``resolve_tasks("humanoid,arm/*")`` -> sorted task names."""

from __future__ import annotations

import fnmatch

SUITES: dict[str, list[str]] = {
    # the disclosed example tasks (the benchmark's other tasks are held out)
    "public": ["arm/balance_scale", "arm/lego_manual", "humanoid/ninja_course"],
}


def resolve_tasks(spec: str) -> list[str]:
    from roboticsbench.tasks.base import TASKS, list_tasks

    all_tasks = list_tasks()
    out: list[str] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "all":
            out += all_tasks
        elif part in ("humanoid", "g1"):
            out += list_tasks("g1")
        elif part in ("arm", "panda"):
            out += list_tasks("panda")
        elif part in SUITES:
            missing = [t for t in SUITES[part] if t not in TASKS]
            if missing:  # a suite may list tasks that are not (yet) available in this build
                import warnings

                warnings.warn(f"suite {part!r}: skipping unknown task(s) {missing}", stacklevel=2)
            out += [t for t in SUITES[part] if t in TASKS]
        elif any(ch in part for ch in "*?["):
            out += [t for t in all_tasks if fnmatch.fnmatch(t, part)]
        elif part.startswith("difficulty="):
            want = part.split("=", 1)[1]
            out += [t for t in all_tasks if TASKS[t].difficulty == want]
        elif part.startswith("category="):
            want = part.split("=", 1)[1]
            out += [t for t in all_tasks if TASKS[t].category == want]
        elif part in TASKS:
            out.append(part)
        else:
            raise KeyError(f"unknown task or suite {part!r}")
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]
